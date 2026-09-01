#pragma once

#include <random>
#include "../core/tetris.h"
#include "python.h"
#include "state.h"

// One environment step, in the two channels the reward spec defines.
//
//   reward  r_t = base(n_lines) * (level + 1) / K, i.e. the NES score of the clear divided
//           by K = 22800 (one level-18 tetris).  Zero on any placement that clears nothing.
//   cost    1 on the transition into TOP_OUT, 0 otherwise.  This is the constraint channel:
//           its undiscounted return is P(top out before the milestone).
//
// There is nothing else.  Every shaping term this environment used to carry -- step
// rewards, aggression multipliers, burn penalties, phantom top-out probabilities, bottom-row
// bonuses -- was removed on purpose: each one biases the pace-survival frontier this
// environment exists to measure.  See NEW_DESIGN_SPEC.md section 6.
struct Reward {
  double reward, cost;
};

class PythonTetris {
 public:
  PyObject_HEAD

  // Absolute line count that ends an episode with no cost charged (spec section 1:
  // MILESTONE).  230 is level-29 entry from a level-18 start.
  static constexpr int kDefaultMilestone = kLineCap < 230 ? kLineCap : 230;

 private:
  // K in r_t = base(n_lines) * (level + 1) / K.  One level-18 tetris, 1200 * 19.  A *fixed*
  // constant, never the current level: normalizing by the current level would distort the
  // relative value of early versus late lines, which is part of what is being measured.
  static constexpr double kScoreNorm_ = 22800.0;

  // Episode terminates here; see kDefaultMilestone.
  int milestone_lines_ = kDefaultMilestone;

  // states
  std::mt19937_64 rng_;
  int next_piece_;
  int piece_count_;
#ifdef NO_ROTATION
  bool is_mirror_;
  bool nnb_;
#else
  bool skip_unique_initial_;
#endif // NO_ROTATION

  // The transition weights are small integers summing to a power of two, so the whole
  // distribution collapses to one table lookup on a single rng draw.  (std::discrete_distribution
  // heap-allocates and renormalizes on every construction, and this runs once per piece.)
  template <size_t N, class T>
  static constexpr std::array<std::array<uint8_t, N>, kPieces> MakeAliasTable_(const T& weights) {
    std::array<std::array<uint8_t, N>, kPieces> ret{};
    for (size_t p = 0; p < kPieces; p++) {
      size_t idx = 0;
      for (size_t next = 0; next < kPieces; next++) {
        for (int k = 0; k < weights[p][next]; k++) ret[p][idx++] = next;
      }
      if (idx != N) throw "transition weights must sum to N";
    }
    return ret;
  }

  int GenNextPiece_(int piece) {
#ifdef USE_PIECE_COUNT_RNG
    static constexpr auto kAlias = [] {
      std::array<std::array<std::array<uint8_t, 16>, kPieces>, 8> ret{};
      for (int i = 0; i < 8; i++) ret[i] = MakeAliasTable_<16>(kTransitionRealisticProbInt[i]);
      return ret;
    }();
    piece_count_ = (piece_count_ + 1) & 7;
    return kAlias[piece_count_][piece][rng_() & 15];
#else
    static constexpr auto kAlias = MakeAliasTable_<32>(kTransitionProbInt);
    return kAlias[piece][rng_() & 31];
#endif
  }

  // The two channels, and nothing else.  `score` is the environment's own NES score for
  // this placement (core/game.h: GameScore -> ScoreFromLevel), so r_t is exactly
  // base(n_lines) * (level + 1) / K with the NES 40/100/300/1200 table.
  //
  // NOTE ON THE LEVEL AT A TRANSITION.  GameScore awards a clear at the level *after* the
  // line counter advances, so the tetris that crosses 130 lines is worth 1200 * 20 rather
  // than 1200 * 19.  That is what the NES does and what a CTWC score readout shows, and it
  // is the reason the reward is taken from the environment's score rather than recomputed
  // here: the undiscounted return then equals the reported pace exactly, which is the whole
  // point of gamma = 1.  Section 2 of the spec describes this as the pre-advance level; the
  // two disagree by one tetris at each of the two transitions inside an episode.  If the
  // pre-advance convention is what you want, change GameScore -- not this -- so that the
  // return and the reported score stay the same number.
  Reward StepAndCalculateReward_(const Position& pos, int score, int lines) {
    // An illegal placement ends the game (consecutive_fail_), so it is a top-out: no score,
    // full cost.  The policy cannot produce one -- every illegal placement is masked to
    // -inf -- so this is a guard, not a shaping term.
    if (score == -1) return {0.0, 1.0};

    // Generate next piece
#ifdef NO_ROTATION
    next_piece_ = GenNextPiece_(next_piece_);
#else
    if (!tetris.IsAdj()) next_piece_ = GenNextPiece_(next_piece_);
#endif

    return {score / kScoreNorm_, IsTopOut() ? 1.0 : 0.0};
  }

#ifndef NO_ROTATION
  Reward CheckReducibleInitial_() {
    if (!skip_unique_initial_ || tetris.IsAdj() || tetris.IsOver()) return {0, 0};
    auto& move_list = tetris.GetPossibleMoveList();
    auto initial_mask = tetris.GetInitialMask();
    if (!move_list.non_adj.empty() || popcount(initial_mask) != 1) return {0, 0};
    Position pos = move_list.adj[ctz(initial_mask)].first;
    auto [score, lines] = tetris.InputPlacement(pos, next_piece_);
    return StepAndCalculateReward_(pos, score, lines);
  }
#endif

 public:
#ifdef NO_ROTATION
  TetrisNoro tetris;

  PythonTetris(size_t seed) : rng_(seed), piece_count_(0) {
    Reset(Board::Ones, 0, 0, true, false, false);
  }

  Position GetRealPosition(Position pos) {
    if (is_mirror_) pos.y = kMirrorCols[tetris.NowPiece()] - pos.y;
    return pos;
  }

  void ResetRandom(const Board& b) {
    int start_level = std::discrete_distribution<int>({
        15, 1, 1, 1, 2, 2, 2, 2, 4, 6, // 0-9
        4, 0, 0, 4, 0, 0, 4, 0, 0, // 10-18
        4, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 19-28
        8})(rng_);
    bool do_tuck = std::discrete_distribution({1, 1})(rng_);
    bool nnb = do_tuck ? std::discrete_distribution<int>({2, 1})(rng_) :
                         std::discrete_distribution<int>({1, 1})(rng_);
    bool is_mirror = std::discrete_distribution({1, 1})(rng_);
    Reset(b, 0, start_level, do_tuck, nnb, is_mirror);
  }

  void Reset(const Board& b, int lines, int start_level, bool do_tuck, bool nnb, bool is_mirror,
             int now_piece = -1, int next_piece = -1) {
    if (now_piece == -1 || next_piece == -1) {
      piece_count_ = std::uniform_int_distribution<int>(0, 8)(rng_);
      if (now_piece == -1) now_piece = std::uniform_int_distribution<int>(0, kPieces - 1)(rng_);
      next_piece = GenNextPiece_(now_piece);
    }
    nnb_ = nnb;
    is_mirror_ = is_mirror;
    tetris.Reset(b, lines, start_level, do_tuck, now_piece, next_piece);
    next_piece_ = GenNextPiece_(next_piece);
  }
#else // !NO_ROTATION
  Tetris tetris;

  PythonTetris(size_t seed) : rng_(seed), piece_count_(0) {
    constexpr Tap30Hz taps;
    Reset(Board::Ones, 0, taps.data(), 18);
  }

  Position GetRealPosition(Position pos) { return pos; }

  void Reset(const Board& b, int lines, const int tap_sequence[], int adj_delay,
             int now_piece = -1, int next_piece = -1, bool skip_unique_initial = false) {
    if (now_piece == -1 || next_piece == -1) {
      piece_count_ = std::uniform_int_distribution<int>(0, 8)(rng_);
      if (now_piece == -1) now_piece = std::uniform_int_distribution<int>(0, kPieces - 1)(rng_);
      next_piece = GenNextPiece_(now_piece);
    }
    tetris.Reset(b, lines, now_piece, next_piece, tap_sequence, adj_delay);
    next_piece_ = GenNextPiece_(next_piece);
    skip_unique_initial_ = skip_unique_initial;
    CheckReducibleInitial_();
  }

  Reward DirectPlacement(const Position& pos) {
    Position npos = GetRealPosition(pos);
    auto [score, lines] = tetris.DirectPlacement(npos, next_piece_);
    return StepAndCalculateReward_(npos, score, lines);
  }

#endif // !NO_ROTATION

  Reward InputPlacement(const Position& pos) {
    Position npos = GetRealPosition(pos);
    auto [score, lines] = tetris.InputPlacement(npos, next_piece_);
    auto reward = StepAndCalculateReward_(npos, score, lines);
#ifdef NO_ROTATION
    return reward;
#else
    if (!skip_unique_initial_) return reward;
    // A forced initial placement is replayed here rather than handed to the policy, so its
    // score counts on this step.  It cannot cause a game over, so the cost is the first
    // placement's.
    auto reward_2 = CheckReducibleInitial_();
    return {reward.reward + reward_2.reward, std::max(reward.cost, reward_2.cost)};
#endif
  }

  /// State generation
#ifdef NO_ROTATION
  void GetState(const StateView& state, int line_reduce = 0) const {
    ::GetState(tetris, state, nnb_, is_mirror_, line_reduce);
  }
#else // !NO_ROTATION
  void GetState(const StateView& state, int line_reduce = 0) const {
    ::GetState(tetris, state, line_reduce);
  }

  MultiState GetAdjStates(const Position& pos) const {
    if (tetris.IsAdj()) throw std::logic_error("should only called on non adj phase");
    Tetris n_tetris = tetris;
    n_tetris.InputPlacement(pos, 0);
    return ::GetAdjStates(n_tetris);
  }
#endif // !NO_ROTATION


  /// Episode boundaries (spec section 1).  Exactly two terminals, and they are told apart
  /// by the line counter, not by which flag the core happened to set:
  ///
  ///   MILESTONE  lines >= milestone            cost 0, V_R target 0
  ///   TOP_OUT    game over and not MILESTONE   cost 1, V_R target 0
  ///
  /// The milestone wins ties.  A placement that clears to exactly the milestone *and*
  /// leaves a board the next piece cannot spawn into is a milestone: the spawn that would
  /// have collided is past the end of the episode and never happens.
  ///
  /// Note that a rollout cut (the batch's line_cap, a short curriculum game, or the end of
  /// the buffer) is *truncation*, not either of these, and is handled by the batch.
  void SetMilestone(int lines) {
    milestone_lines_ = lines > 0 ? lines : kLineCap;
  }
  int Milestone() const { return milestone_lines_; }
  bool ReachedMilestone() const { return tetris.GetLines() >= milestone_lines_; }
  bool IsTopOut() const { return tetris.IsOver() && !ReachedMilestone(); }
  bool IsEpisodeOver() const { return tetris.IsOver() || ReachedMilestone(); }

#ifdef NO_ROTATION
  operator TetrisNoro() const { return tetris; }
#else
  operator Tetris() const { return tetris; }
#endif // NO_ROTATION
};

extern PyTypeObject py_tetris_class;
