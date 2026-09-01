#pragma once

#include <atomic>
#include <thread>
#include <vector>
#include <memory>
#include <string>
#include <condition_variable>

#include "python.h"
#include "tetris.h"

// A fixed-size pool that runs body(i) for i in [0, n) across its workers, with the calling
// thread participating.  Work is handed out with an atomic counter because env step cost
// varies by an order of magnitude (an adjustment phase is ~10x cheaper than a placement,
// and cost also depends on tap speed / level), so static chunking would idle most workers.
class ParallelFor {
 public:
  struct Body {
    virtual void Run(int i) = 0;
    virtual ~Body() = default;
  };

  explicit ParallelFor(int num_threads);
  ~ParallelFor();

  ParallelFor(const ParallelFor&) = delete;
  ParallelFor& operator=(const ParallelFor&) = delete;

  int NumThreads() const { return num_threads_; }
  void Run(int n, Body& body);

 private:
  void WorkerLoop_(int id);
  void Drain_();

  int num_threads_;
  std::vector<std::thread> threads_;
  std::mutex mtx_;
  std::condition_variable start_cv_, done_cv_;
  Body* body_ = nullptr;
  int n_ = 0;
  uint64_t epoch_ = 0;
  int busy_ = 0;
  bool stop_ = false;
  alignas(64) std::atomic<int> next_{0};
};

// A vector of games stepped in lockstep.  Observations are written straight into buffers
// owned by the caller (pinned host memory when training on a GPU), so a batched step does
// no allocation, no copying and no per-env Python work.
class PythonTetrisBatch {
 public:
  PyObject_HEAD

  struct EnvState {
    double reward_acc = 0;
    bool prev_truncate = false;
    bool is_short = false;
    // Absolute line count at which the episode is *cut* -- a truncation, not a terminal --
    // 0 for "no cut".  This is how a run can be restricted to one level segment; see
    // training_env/game_param.py.  The milestone is separate and lives on the env itself.
    int line_cap = 0;
    // Whether this episode was started from a true level-18 start (empty board, 0 lines).
    // The dual variable's p_hat and every reported number must be measured on these alone,
    // or `c` constrains a distribution nobody plays (spec section 9).
    bool is_true_start = true;
  };

  // Result of stepping one env whose episode ended.
  struct EpisodeInfo {
    int index;
    bool is_short;      // a short curriculum game, cut once the board was cleaned up
    bool is_topout;     // TOP_OUT -- the constraint's event.  Otherwise MILESTONE or a cut.
    bool is_milestone;  // reached the milestone line count
    bool is_true_start;
    double reward;      // sum of r_t over the episode
    int score, lines, pieces, tetrises;
  };

  PythonTetrisBatch(int n, uint64_t seed, int num_threads);
  ~PythonTetrisBatch();

  int Size() const { return n_; }
  int NumThreads() const { return pool_ ? pool_->NumThreads() : 1; }
  PythonTetris& Env(int i) { return envs_[i]; }
  EnvState& Extra(int i) { return extra_[i]; }

  // obs is the base of the (num_obs_steps, n, ...) rollout buffer; rewards and over are
  // (n, num_steps, 2) and (n, num_steps, 2) -- [reward, cost] and [done, truncated].
  void SetBuffers(const StateView& obs, int num_obs_steps, float* rewards, bool* over, int num_steps);
  bool HasBuffers() const { return buffers_set_; }
  void SetObsStep(int step) { obs_step_ = step; }
  int NumObsSteps() const { return num_obs_steps_; }
  int NumSteps() const { return num_steps_; }

  StateView ObsAt(int i) const;
  void WriteObs(int i);
  // Steps every env, storing rewards/dones at slot `step` and the resulting observations at
  // slot `step + 1`.  Envs whose episode ended are returned and left un-reset: their
  // observation slot stays stale until ResetEnv is called.
  void Step(int step, const int32_t* actions, std::vector<EpisodeInfo>& finished);

  const std::string& Error() const { return error_; }

 private:
  void StepOne_(int i, int step, const int32_t* actions);

  int n_;
  std::vector<PythonTetris> envs_;
  std::vector<EnvState> extra_;
  std::unique_ptr<ParallelFor> pool_;

  bool buffers_set_ = false;
  StateView obs_{};
  int obs_step_ = 0;
  int num_obs_steps_ = 0;
  float* rewards_ = nullptr;
  bool* over_ = nullptr;
  int num_steps_ = 0;

  // Per-env episode-end flags, collected after the parallel section.
  std::vector<uint8_t> done_flags_;
  std::mutex error_mtx_;
  std::string error_;
};

extern PyTypeObject py_tetris_batch_class;
