#include "batch.h"

#include <algorithm>
#include <stdexcept>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL TETRIS_PY_ARRAY_SYMBOL_
#include <numpy/ndarrayobject.h>

#include "board.h"
#include "piece.h"

/// -------- ParallelFor --------

ParallelFor::ParallelFor(int num_threads) : num_threads_(std::max(1, num_threads)) {
  threads_.reserve(num_threads_ - 1);
  for (int i = 1; i < num_threads_; i++) {
    threads_.emplace_back([this, i] { WorkerLoop_(i); });
  }
}

ParallelFor::~ParallelFor() {
  {
    std::lock_guard<std::mutex> lck(mtx_);
    stop_ = true;
    epoch_++;
  }
  start_cv_.notify_all();
  for (auto& t : threads_) t.join();
}

void ParallelFor::Drain_() {
  int n = n_;
  Body* body = body_;
  for (int i = next_.fetch_add(1, std::memory_order_relaxed); i < n;
       i = next_.fetch_add(1, std::memory_order_relaxed)) {
    body->Run(i);
  }
}

void ParallelFor::WorkerLoop_(int) {
  uint64_t seen = 0;
  while (true) {
    {
      std::unique_lock<std::mutex> lck(mtx_);
      start_cv_.wait(lck, [this, &seen] { return epoch_ != seen; });
      seen = epoch_;
      if (stop_) return;
    }
    Drain_();
    {
      std::lock_guard<std::mutex> lck(mtx_);
      if (--busy_ == 0) done_cv_.notify_one();
    }
  }
}

void ParallelFor::Run(int n, Body& body) {
  if (num_threads_ == 1 || n <= 1) {
    for (int i = 0; i < n; i++) body.Run(i);
    return;
  }
  {
    std::lock_guard<std::mutex> lck(mtx_);
    n_ = n;
    body_ = &body;
    next_.store(0, std::memory_order_relaxed);
    busy_ = num_threads_ - 1;
    epoch_++;
  }
  start_cv_.notify_all();
  Drain_(); // the caller is a worker too
  std::unique_lock<std::mutex> lck(mtx_);
  done_cv_.wait(lck, [this] { return busy_ == 0; });
}

/// -------- PythonTetrisBatch --------

namespace {

// Split the seed the same way the old per-worker code did, so each env gets an
// independent stream.
inline uint64_t SplitMix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

} // namespace

PythonTetrisBatch::PythonTetrisBatch(int n, uint64_t seed, int num_threads) : n_(n) {
  envs_.reserve(n);
  for (int i = 0; i < n; i++) envs_.emplace_back(SplitMix64(seed + i));
  extra_.resize(n);
  done_flags_.resize(n);
  pool_ = std::make_unique<ParallelFor>(num_threads);
}

PythonTetrisBatch::~PythonTetrisBatch() = default;

void PythonTetrisBatch::SetBuffers(
    const StateView& obs, int num_obs_steps, float* rewards, bool* over, int num_steps) {
  obs_ = obs;
  num_obs_steps_ = num_obs_steps;
  obs_step_ = 0;
  rewards_ = rewards;
  over_ = over;
  num_steps_ = num_steps;
  buffers_set_ = true;
}

StateView PythonTetrisBatch::ObsAt(int i) const {
  size_t slot = static_cast<size_t>(obs_step_) * n_ + i;
  return {obs_.board + slot * (kBoardPlanes * kPlaneSize),
          obs_.meta + slot * kMetaSize,
          obs_.moves + slot * (kMovePlanes * kPlaneSize),
          obs_.move_meta + slot * kMoveMetaSize,
          obs_.meta_int + slot * kMetaIntSize};
}

void PythonTetrisBatch::WriteObs(int i) {
  envs_[i].GetState(ObsAt(i));
}

void PythonTetrisBatch::StepOne_(int i, int step, const int32_t* actions) {
  PythonTetris& env = envs_[i];
  EnvState& ex = extra_[i];
  int32_t action = actions[i];
  Position pos{static_cast<uint8_t>(action / 200), static_cast<uint8_t>(action / 10 % 20),
               static_cast<uint8_t>(action % 10)};
  Reward reward = env.InputPlacement(pos);
  ex.reward_acc += reward.reward;

  float* rw = rewards_ + (static_cast<size_t>(i) * num_steps_ + step) * 2;
  rw[0] = reward.reward;
  rw[1] = reward.cost;

  // done[0] = the episode ended here and the env is about to be reset.  done[1] separates
  // *truncation* -- a cut, where the trainer must bootstrap both value heads off the state
  // that follows -- from the two real terminals, where it must not (spec section 8).
  bool truncated = ex.prev_truncate;
  bool done = truncated || env.IsEpisodeOver();
  bool* ov = over_ + (static_cast<size_t>(i) * num_steps_ + step) * 2;
  ov[0] = done;
  ov[1] = truncated;

  // Two ways an episode gets cut short of a terminal: a short curriculum game, which ends
  // once the board is clean again, and a line_cap that pins the run to one level segment.
  // Neither is a top-out and neither is the milestone, so neither may be charged as one:
  // the flag set here makes the *next* step a truncation, which gives the trainer a real
  // successor state to bootstrap from before the env is reset.  That step is then dropped
  // from the loss (generator.py: skip_mask).
  ex.prev_truncate =
      (ex.is_short && env.tetris.RunLines() >= 6 && BoardIsClean(env.tetris.GetBoard())) ||
      (ex.line_cap > 0 && env.tetris.GetLines() >= ex.line_cap && !env.IsEpisodeOver());

  done_flags_[i] = done;
  if (!done) env.GetState(ObsAt(i));
}

void PythonTetrisBatch::Step(int step, const int32_t* actions, std::vector<EpisodeInfo>& finished) {
  struct StepBody : ParallelFor::Body {
    PythonTetrisBatch* self;
    int step;
    const int32_t* actions;
    StepBody(PythonTetrisBatch* self, int step, const int32_t* actions)
        : self(self), step(step), actions(actions) {}
    void Run(int i) override {
      try {
        self->StepOne_(i, step, actions);
      } catch (std::exception& e) {
        std::lock_guard<std::mutex> lck(self->error_mtx_);
        if (self->error_.empty()) self->error_ = e.what();
        self->done_flags_[i] = 0;
      }
    }
  } body(this, step, actions);

  error_.clear();
  obs_step_ = step + 1;
  pool_->Run(n_, body);

  finished.clear();
  for (int i = 0; i < n_; i++) {
    if (!done_flags_[i]) continue;
    PythonTetris& env = envs_[i];
    const EnvState& ex = extra_[i];
    // A truncated episode is neither terminal: prev_truncate is what ended it, and the env
    // never topped out nor reached the milestone on the step being reported.
    bool truncated = ex.prev_truncate;
    finished.push_back({i, ex.is_short, !truncated && env.IsTopOut(),
                        !truncated && env.ReachedMilestone(), ex.is_true_start, ex.reward_acc,
                        env.tetris.RunScore(), env.tetris.RunLines(), env.tetris.RunPieces(),
                        env.tetris.RunTetrises()});
  }
}

/// -------- Python bindings --------

namespace {

void BatchDealloc(PythonTetrisBatch* self) {
  self->~PythonTetrisBatch();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject* BatchNew(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  return (PyObject*)type->tp_alloc(type, 0);
}

int BatchInit(PythonTetrisBatch* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"num_envs", "seed", "num_threads", nullptr};
  int n = 0, num_threads = 0;
  unsigned long long seed = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "i|Ki", (char**)kwlist, &n, &seed, &num_threads)) {
    return -1;
  }
  if (n <= 0) {
    PyErr_SetString(PyExc_ValueError, "num_envs must be positive");
    return -1;
  }
  if (num_threads <= 0) {
    num_threads = std::max(1u, std::thread::hardware_concurrency());
  }
  num_threads = std::min(num_threads, n);
  new (self) PythonTetrisBatch(n, seed, num_threads);
  return 0;
}

// Validates an array and returns its data pointer, or nullptr with an exception set.
void* CheckArray(PyObject* obj, int typenum, int ndim, const npy_intp* dims, const char* name) {
  if (!PyArray_Check(obj)) {
    PyErr_Format(PyExc_TypeError, "%s must be a numpy array", name);
    return nullptr;
  }
  PyArrayObject* arr = (PyArrayObject*)obj;
  if (PyArray_TYPE(arr) != typenum) {
    PyErr_Format(PyExc_TypeError, "%s has wrong dtype", name);
    return nullptr;
  }
  if (!PyArray_IS_C_CONTIGUOUS(arr)) {
    PyErr_Format(PyExc_ValueError, "%s must be C-contiguous", name);
    return nullptr;
  }
  if (PyArray_NDIM(arr) != ndim) {
    PyErr_Format(PyExc_ValueError, "%s has wrong number of dimensions", name);
    return nullptr;
  }
  for (int i = 0; i < ndim; i++) {
    if (PyArray_DIM(arr, i) != dims[i]) {
      PyErr_Format(PyExc_ValueError, "%s has wrong shape in dimension %d", name, i);
      return nullptr;
    }
  }
  return PyArray_DATA(arr);
}

PyObject* Batch_SetBuffers(PythonTetrisBatch* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"obs", "rewards", "over", nullptr};
  PyObject *obs_obj, *reward_obj, *over_obj;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO", (char**)kwlist, &obs_obj, &reward_obj, &over_obj)) {
    return nullptr;
  }
  if (!PySequence_Check(obs_obj) || PySequence_Size(obs_obj) != 5) {
    PyErr_SetString(PyExc_TypeError, "obs must be a sequence of 5 arrays");
    return nullptr;
  }
  int n = self->Size();
  PyObject* items[5];
  for (int i = 0; i < 5; i++) {
    items[i] = PySequence_GetItem(obs_obj, i);
    if (!items[i]) {
      for (int j = 0; j < i; j++) Py_DECREF(items[j]);
      return nullptr;
    }
  }
  // obs arrays are (num_obs_steps, n, ...): the batch writes each observation straight into
  // its final slot in the rollout buffer.
  int obs_steps = 0;
  if (PyArray_Check(items[0]) && PyArray_NDIM((PyArrayObject*)items[0]) == 5) {
    obs_steps = PyArray_DIM((PyArrayObject*)items[0], 0);
  }
  StateView view{};
  bool ok = obs_steps > 0;
  if (!ok) {
    PyErr_SetString(PyExc_ValueError, "obs[0] must be a 5d array (steps, envs, planes, 20, 10)");
  } else {
    npy_intp d0[] = {obs_steps, n, kBoardPlanes, 20, 10};
    npy_intp d1[] = {obs_steps, n, kMetaSize};
    npy_intp d2[] = {obs_steps, n, kMovePlanes, 20, 10};
    npy_intp d3[] = {obs_steps, n, kMoveMetaSize};
    npy_intp d4[] = {obs_steps, n, kMetaIntSize};
    view.board = (uint8_t*)CheckArray(items[0], NPY_UINT8, 5, d0, "obs[0]");
    view.meta = (float*)CheckArray(items[1], NPY_FLOAT32, 3, d1, "obs[1]");
    view.moves = (uint8_t*)CheckArray(items[2], NPY_UINT8, 5, d2, "obs[2]");
    view.move_meta = (float*)CheckArray(items[3], NPY_FLOAT32, 3, d3, "obs[3]");
    view.meta_int = (int32_t*)CheckArray(items[4], NPY_INT32, 3, d4, "obs[4]");
    ok = view.board && view.meta && view.moves && view.move_meta && view.meta_int;
  }
  for (int i = 0; i < 5; i++) Py_DECREF(items[i]);
  if (!ok) return nullptr;

  if (!PyArray_Check(reward_obj) || PyArray_NDIM((PyArrayObject*)reward_obj) != 3) {
    PyErr_SetString(PyExc_ValueError, "rewards must be a 3d array (envs, steps, 2)");
    return nullptr;
  }
  int num_steps = PyArray_DIM((PyArrayObject*)reward_obj, 1);
  npy_intp dr[] = {n, num_steps, 2};
  npy_intp dov[] = {n, num_steps, 2};
  float* rewards = (float*)CheckArray(reward_obj, NPY_FLOAT32, 3, dr, "rewards");
  if (!rewards) return nullptr;
  bool* over = (bool*)CheckArray(over_obj, NPY_BOOL, 3, dov, "over");
  if (!over) return nullptr;
  static_assert(sizeof(bool) == 1, "npy_bool must match bool");
  if (obs_steps < num_steps + 1) {
    PyErr_SetString(PyExc_ValueError, "obs must have at least num_steps + 1 slots");
    return nullptr;
  }

  self->SetBuffers(view, obs_steps, rewards, over, num_steps);
  Py_RETURN_NONE;
}

PyObject* Batch_SetObsStep(PythonTetrisBatch* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"step", nullptr};
  int step;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", (char**)kwlist, &step)) return nullptr;
  if (step < 0 || step >= self->NumObsSteps()) {
    PyErr_SetString(PyExc_IndexError, "obs step out of range");
    return nullptr;
  }
  self->SetObsStep(step);
  Py_RETURN_NONE;
}

PyObject* Batch_Step(PythonTetrisBatch* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"step", "actions", nullptr};
  int step;
  PyObject* action_obj;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iO", (char**)kwlist, &step, &action_obj)) {
    return nullptr;
  }
  if (!self->HasBuffers()) {
    PyErr_SetString(PyExc_RuntimeError, "SetBuffers must be called before Step");
    return nullptr;
  }
  if (step < 0 || step >= self->NumSteps()) {
    PyErr_SetString(PyExc_IndexError, "step out of range");
    return nullptr;
  }
  npy_intp da[] = {self->Size()};
  int32_t* actions = (int32_t*)CheckArray(action_obj, NPY_INT32, 1, da, "actions");
  if (!actions) return nullptr;
  // the move map is indexed directly with no bounds check, so screen the whole array once
  for (int i = 0, n = self->Size(); i < n; i++) {
    if (actions[i] < 0 || actions[i] >= kActionSize) {
      PyErr_Format(PyExc_ValueError, "action %d out of range at index %d", (int)actions[i], i);
      return nullptr;
    }
  }

  static thread_local std::vector<PythonTetrisBatch::EpisodeInfo> finished;
  Py_BEGIN_ALLOW_THREADS
  self->Step(step, actions, finished);
  Py_END_ALLOW_THREADS
  if (!self->Error().empty()) {
    PyErr_SetString(PyExc_RuntimeError, self->Error().c_str());
    return nullptr;
  }

  PyObject* list = PyList_New(finished.size());
  if (!list) return nullptr;
  for (size_t i = 0; i < finished.size(); i++) {
    const auto& f = finished[i];
    PyObject* item = Py_BuildValue("(iOOOOdiiii)", f.index, f.is_short ? Py_True : Py_False,
                                   f.is_topout ? Py_True : Py_False,
                                   f.is_milestone ? Py_True : Py_False,
                                   f.is_true_start ? Py_True : Py_False, f.reward, f.score,
                                   f.lines, f.pieces, f.tetrises);
    if (!item) {
      Py_DECREF(list);
      return nullptr;
    }
    PyList_SET_ITEM(list, i, item);
  }
  return list;
}

PyObject* Batch_ResetEnv(PythonTetrisBatch* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {
    "index", "is_short", "now_piece", "next_piece", "lines", "board", "line_cap",
    "milestone", "is_true_start",
#ifdef NO_ROTATION
    "start_level", "do_tuck", "nnb", "mirror",
#else
    "tap_sequence", "adj_delay", "skip_unique_initial",
#endif
    nullptr
  };
  int index;
  int is_short = 0;
  PyObject *now_obj = nullptr, *next_obj = nullptr, *board_obj = nullptr;
  int lines = 0, line_cap = 0;
  int milestone = PythonTetris::kDefaultMilestone;
  int is_true_start = 1;
  Board board = Board::Ones;
#ifdef NO_ROTATION
  int start_level = 0, do_tuck = 1, nnb = 0, mirror = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "i|pOOiOiipippp", (char**)kwlist, &index, &is_short,
        &now_obj, &next_obj, &lines, &board_obj, &line_cap, &milestone, &is_true_start,
        &start_level, &do_tuck, &nnb, &mirror)) {
    return nullptr;
  }
#else
  PyObject* tap_sequence_obj = nullptr;
  int adj_delay = 18;
  int skip_unique_initial = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "i|pOOiOiipOip", (char**)kwlist, &index, &is_short,
        &now_obj, &next_obj, &lines, &board_obj, &line_cap, &milestone, &is_true_start,
        &tap_sequence_obj, &adj_delay, &skip_unique_initial)) {
    return nullptr;
  }
  std::array<int, 10> tap_sequence;
  if (tap_sequence_obj) {
    if (!ParseTapSequence(tap_sequence_obj, tap_sequence.data())) return nullptr;
  } else {
    constexpr Tap30Hz tap_table;
    memcpy(tap_sequence.data(), tap_table.data(), sizeof(tap_sequence));
  }
#endif
  if (index < 0 || index >= self->Size()) {
    PyErr_SetString(PyExc_IndexError, "env index out of range");
    return nullptr;
  }
  if (board_obj) {
    if (!PyObject_IsInstance(board_obj, (PyObject*)&py_board_class)) {
      PyErr_SetString(PyExc_TypeError, "Invalid board type.");
      return nullptr;
    }
    board = reinterpret_cast<PythonBoard*>(board_obj)->board;
  }
  int now_piece = -1, next_piece = -1;
  if (now_obj) {
    now_piece = ParsePieceID(now_obj);
    if (now_piece < 0) return nullptr;
    if (next_obj) {
      next_piece = ParsePieceID(next_obj);
      if (next_piece < 0) return nullptr;
    }
  }

  PythonTetris& env = self->Env(index);
  env.SetMilestone(milestone);
  try {
#ifdef NO_ROTATION
    env.Reset(board, lines, start_level, do_tuck, nnb, mirror, now_piece, next_piece);
#else
    env.Reset(board, lines, tap_sequence.data(), adj_delay, now_piece, next_piece, skip_unique_initial);
#endif
  } catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
  auto& ex = self->Extra(index);
  ex.reward_acc = 0;
  ex.prev_truncate = false;
  ex.is_short = is_short;
  ex.line_cap = line_cap;
  ex.is_true_start = is_true_start;
  bool over = env.IsEpisodeOver();
  if (!over && self->HasBuffers()) self->WriteObs(index);
  return PyBool_FromLong(over);
}

PyObject* Batch_Size(PythonTetrisBatch* self, PyObject* Py_UNUSED(ignored)) {
  return PyLong_FromLong(self->Size());
}

PyObject* Batch_NumThreads(PythonTetrisBatch* self, PyObject* Py_UNUSED(ignored)) {
  return PyLong_FromLong(self->NumThreads());
}

PyMethodDef py_tetris_batch_methods[] = {
    {"SetBuffers", (PyCFunction)Batch_SetBuffers, METH_VARARGS | METH_KEYWORDS,
     "Bind the destination buffers for observations, rewards and done flags"},
    {"SetObsStep", (PyCFunction)Batch_SetObsStep, METH_VARARGS | METH_KEYWORDS,
     "Choose which rollout slot ResetEnv writes observations to"},
    {"Step", (PyCFunction)Batch_Step, METH_VARARGS | METH_KEYWORDS,
     "Step every env; returns (index, is_short, is_topout, is_milestone, is_true_start, "
     "reward, score, lines, pieces, tetrises) for each episode that ended"},
    {"ResetEnv", (PyCFunction)Batch_ResetEnv, METH_VARARGS | METH_KEYWORDS,
     "Reset one env; returns whether the fresh game is already over"},
    {"Size", (PyCFunction)Batch_Size, METH_NOARGS, "Number of envs"},
    {"NumThreads", (PyCFunction)Batch_NumThreads, METH_NOARGS, "Number of worker threads"},
    {nullptr}};

} // namespace

// PythonTetrisBatch keeps its envs in a std::vector, which over-aligns them itself.
// If it ever gains one, this fires instead of segfaulting at random -- see
// AlignedAlloc in python.h.
static_assert(!kNeedsAlignedAlloc<PythonTetrisBatch>);

PyTypeObject py_tetris_batch_class = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "tetris.Batch",              // tp_name
    sizeof(PythonTetrisBatch),   // tp_basicsize
    0,                           // tp_itemsize
    (destructor)BatchDealloc,    // tp_dealloc
    0,                           // tp_vectorcall_offset
    0,                           // tp_getattr
    0,                           // tp_setattr
    0,                           // tp_as_async
    0,                           // tp_repr
    0,                           // tp_as_number
    0,                           // tp_as_sequence
    0,                           // tp_as_mapping
    0,                           // tp_hash
    0,                           // tp_call
    0,                           // tp_str
    0,                           // tp_getattro
    0,                           // tp_setattro
    0,                           // tp_as_buffer
    Py_TPFLAGS_DEFAULT,          // tp_flags
    "Batched Tetris environment", // tp_doc
    0,                           // tp_traverse
    0,                           // tp_clear
    0,                           // tp_richcompare
    0,                           // tp_weaklistoffset
    0,                           // tp_iter
    0,                           // tp_iternext
    py_tetris_batch_methods,     // tp_methods
    0,                           // tp_members
    0,                           // tp_getset
    0,                           // tp_base
    0,                           // tp_dict
    0,                           // tp_descr_get
    0,                           // tp_descr_set
    0,                           // tp_dictoffset
    (initproc)BatchInit,         // tp_init
    0,                           // tp_alloc
    BatchNew,                    // tp_new
};
