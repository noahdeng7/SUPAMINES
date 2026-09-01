#include "tetris.h"

#include <optional>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL TETRIS_PY_ARRAY_SYMBOL_
#include <numpy/ndarrayobject.h>

#include "board.h"
#include "piece.h"

namespace {

/// -------- helpers --------

bool CheckBoard(Board& board, PyObject* obj) {
  if (obj) {
    if (!PyObject_IsInstance(obj, (PyObject*)&py_board_class)) {
      PyErr_SetString(PyExc_TypeError, "Invalid board type.");
      return false;
    }
    board = reinterpret_cast<PythonBoard*>(obj)->board;
  }
  return true;
}

PyObject* PositionToTuple(const Position& pos) {
  return Py_BuildValue("(bbb)", pos.r, pos.x, pos.y);
}

PyObject* FrameSequenceToArray(const FrameSequence& seq) {
  npy_intp dims[] = {(long)seq.size()};
  PyObject* ret = PyArray_SimpleNew(1, dims, NPY_UINT8);
  static_assert(sizeof(seq[0]) == 1);
  memcpy(PyArray_DATA((PyArrayObject*)ret), seq.data(), seq.size());
  return ret;
}

#ifndef NO_ROTATION
std::optional<FrameSequence> ArrayToFrameSequence(PyObject* obj) {
  if (!PyArray_Check(obj)) {
    PyErr_SetString(PyExc_IndexError, "Invalid frame sequence");
    return std::nullopt;
  }
  PyArrayObject *arr = (PyArrayObject*)PyArray_FROM_OTF(obj, NPY_UINT8, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_FORCECAST);
  if (!arr) return std::nullopt;
  if (PyArray_NDIM(arr) != 1) {
    PyErr_SetString(PyExc_IndexError, "Array must be 1D");
    return std::nullopt;
  }
  FrameSequence seq(PyArray_DIM(arr, 0));
  memcpy(seq.data(), PyArray_DATA(arr), seq.size());
  Py_DECREF(arr);
  return seq;
}
#endif // !NO_ROTATION

// (reward, cost): the score channel and the constraint channel.  See tetris.h.
PyObject* GetRewardObj(Reward reward) {
  return Py_BuildValue("(dd)", reward.reward, reward.cost);
}

/// -------- impl --------

void TetrisDealloc(PythonTetris* self) {
  self->~PythonTetris();
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
  Py_TYPE(self)->tp_free((PyObject*)self);
#pragma GCC diagnostic pop
}

PyObject* TetrisNew(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  PythonTetris* self = (PythonTetris*)type->tp_alloc(type, 0);
  // leave initialization to __init__
  return (PyObject*)self;
}

int TetrisInit(PythonTetris* self, PyObject* args, PyObject* kwds) {
  static const char *kwlist[] = {"seed", nullptr};
  unsigned long long seed = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|K", (char**)kwlist, &seed)) {
    return -1;
  }
  new(self) PythonTetris(seed);
  return 0;
}

PyObject* Tetris_InputPlacement(PythonTetris* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"rotate", "x", "y", nullptr};
  Position pos;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "bbb", (char**)kwlist, &pos.r, &pos.x, &pos.y)) {
    return nullptr;
  }
  return GetRewardObj(self->InputPlacement(pos));
}

#ifndef NO_ROTATION
PyObject* Tetris_DirectPlacement(PythonTetris* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"rotate", "x", "y", nullptr};
  Position pos;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "bbb", (char**)kwlist, &pos.r, &pos.x, &pos.y)) {
    return nullptr;
  }
  return GetRewardObj(self->DirectPlacement(pos));
}
#endif // !NO_ROTATION

PyObject* Tetris_SetNextPiece(PythonTetris* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"piece", nullptr};
  PyObject* piece_obj;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &piece_obj)) {
    return nullptr;
  }
  int piece = ParsePieceID(piece_obj);
  if (piece < 0) return nullptr;
  self->tetris.SetNextPiece(piece);
  Py_RETURN_NONE;
}

PyObject* Tetris_SetLines(PythonTetris* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"lines", nullptr};
  int lines;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", (char**)kwlist, &lines)) {
    return nullptr;
  }
  try {
    self->tetris.SetLines(lines);
    Py_RETURN_NONE;
  } catch (std::range_error& e) {
    PyErr_SetString(PyExc_ValueError, e.what());
    return nullptr;
  }
}

PyObject* Tetris_SetMilestone(PythonTetris* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"lines", nullptr};
  int lines;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", (char**)kwlist, &lines)) {
    return nullptr;
  }
  self->SetMilestone(lines);
  Py_RETURN_NONE;
}

PyObject* Tetris_Reset(PythonTetris* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {
    "now_piece", "next_piece", "lines", "board", "milestone",
#ifdef NO_ROTATION
    "start_level", "do_tuck", "nnb", "mirror",
#else
    "tap_sequence", "adj_delay", "skip_unique_initial",
#endif
    nullptr
  };
  PyObject *now_obj = nullptr, *next_obj = nullptr, *board_obj = nullptr;
  int lines = 0;
  int milestone = PythonTetris::kDefaultMilestone;
  Board board = Board::Ones;
#ifdef NO_ROTATION
  int start_level = 0;
  int do_tuck = 1;
  int nnb = 0;
  int mirror = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOiOiippp", (char**)kwlist,
        &now_obj, &next_obj, &lines, &board_obj, &milestone,
        &start_level, &do_tuck, &nnb, &mirror)) {
    return nullptr;
  }
#else
  std::array<int, 10> tap_sequence;
  PyObject* tap_sequence_obj = nullptr;
  int adj_delay = 18;
  int skip_unique_initial = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOiOiOip", (char**)kwlist,
        &now_obj, &next_obj, &lines, &board_obj, &milestone,
        &tap_sequence_obj, &adj_delay, &skip_unique_initial)) {
    return nullptr;
  }
  if (tap_sequence_obj) {
    if (!ParseTapSequence(tap_sequence_obj, tap_sequence.data())) return nullptr;
  } else {
    constexpr Tap30Hz tap_table;
    memcpy(tap_sequence.data(), tap_table.data(), sizeof(tap_sequence));
  }
#endif
  if (!CheckBoard(board, board_obj)) return nullptr;
  int now_piece = -1, next_piece = -1;
  if (now_obj) {
    now_piece = ParsePieceID(now_obj);
    if (now_piece < 0) return nullptr;
    if (next_obj) {
      next_piece = ParsePieceID(next_obj);
      if (next_piece < 0) return nullptr;
    }
  }
  self->SetMilestone(milestone);
#ifdef NO_ROTATION
  self->Reset(board, lines, start_level, do_tuck, nnb, mirror, now_piece, next_piece);
#else
  self->Reset(board, lines, tap_sequence.data(), adj_delay, now_piece, next_piece, skip_unique_initial);
#endif
  Py_RETURN_NONE;
}

#ifdef NO_ROTATION
PyObject* Tetris_ResetRandom(PythonTetris* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {
    "params", "board",
    nullptr
  };
  int start_level = -1, do_tuck = -1, nnb = -1, mirror = -1;
  Board board = Board::Ones;
  PyObject* board_obj = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|(ippp)O", (char**)kwlist, &start_level, &do_tuck, &nnb, &mirror, &board_obj)) {
    return nullptr;
  }
  if (!CheckBoard(board, board_obj)) return nullptr;
  if (start_level == -1) {
    self->ResetRandom(board);
  } else {
    self->Reset(board, 0, start_level, do_tuck, nnb, mirror);
  }
  Py_RETURN_NONE;
}
#endif // NO_ROTATION

PyObject* Tetris_GetState(PythonTetris* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"line_reduce", nullptr};
  int line_reduce = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", (char**)kwlist, &line_reduce)) {
    return nullptr;
  }
  State state{};
  self->GetState(state.View(), line_reduce);
  return state.ToPython();
}

#ifndef NO_ROTATION
PyObject* Tetris_GetAdjStates(PythonTetris* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"rotate", "x", "y", nullptr};
  Position pos;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "bbb", (char**)kwlist, &pos.r, &pos.x, &pos.y)) {
    return nullptr;
  }
  return self->GetAdjStates(pos).ToPython();
}
#endif // !NO_ROTATION

static PyObject* Tetris_StateShapes(void*, PyObject* Py_UNUSED(ignored)) {
  PyObject *r1, *r2, *r3, *r4, *r5;
  r1 = Py_BuildValue("(iii)", kStateShapes[0][0], kStateShapes[0][1], kStateShapes[0][2]);
  r2 = Py_BuildValue("(i)", kStateShapes[1][0]);
  r3 = Py_BuildValue("(iii)", kStateShapes[2][0], kStateShapes[2][1], kStateShapes[2][2]);
  r4 = Py_BuildValue("(i)", kStateShapes[3][0]);
  r5 = Py_BuildValue("(i)", kStateShapes[4][0]);
  PyObject* ret = PyTuple_Pack(5, r1, r2, r3, r4, r5);
  Py_DECREF(r1);
  Py_DECREF(r2);
  Py_DECREF(r3);
  Py_DECREF(r4);
  Py_DECREF(r5);
  return ret;
}

PyObject* Tetris_StateTypes(void*, PyObject* Py_UNUSED(ignored)) {
  return Py_BuildValue("(sssss)", "uint8", "float32", "uint8", "float32", "int32");
}

PyObject* Tetris_GetSequence(PythonTetris* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"rotate", "x", "y", nullptr};
  Position pos;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "bbb", (char**)kwlist, &pos.r, &pos.x, &pos.y)) {
    return nullptr;
  }
  return FrameSequenceToArray(self->tetris.GetSequence(self->GetRealPosition(pos)));
}

#ifndef NO_ROTATION
PyObject* Tetris_IsAdjMove(PythonTetris* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"rotate", "x", "y", nullptr};
  Position pos;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "bbb", (char**)kwlist, &pos.r, &pos.x, &pos.y)) {
    return nullptr;
  }
  return PyBool_FromLong(self->tetris.IsAdjMove(pos));
}

PyObject* Tetris_IsNoAdjMove(PythonTetris* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"rotate", "x", "y", nullptr};
  Position pos;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "bbb", (char**)kwlist, &pos.r, &pos.x, &pos.y)) {
    return nullptr;
  }
  return PyBool_FromLong(self->tetris.IsNoAdjMove(pos));
}

PyObject* Tetris_GetAdjPremove(PythonTetris* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"pos_list", nullptr};
  Position pos[7];
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "((bbb)(bbb)(bbb)(bbb)(bbb)(bbb)(bbb))", (char**)kwlist,
        &pos[0].r, &pos[0].x, &pos[0].y,
        &pos[1].r, &pos[1].x, &pos[1].y,
        &pos[2].r, &pos[2].x, &pos[2].y,
        &pos[3].r, &pos[3].x, &pos[3].y,
        &pos[4].r, &pos[4].x, &pos[4].y,
        &pos[5].r, &pos[5].x, &pos[5].y,
        &pos[6].r, &pos[6].x, &pos[6].y)) {
    return nullptr;
  }
  auto [npos, seq] = self->tetris.GetAdjPremove(pos);
  PyObject *r1 = PositionToTuple(npos), *r2 = FrameSequenceToArray(seq);
  PyObject* ret = PyTuple_Pack(2, r1, r2);
  Py_DECREF(r1);
  Py_DECREF(r2);
  return ret;
}

PyObject* Tetris_FinishAdjSequence(PythonTetris* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"sequence", "intermediate_pos", "final_pos", nullptr};
  Position intermediate_pos, final_pos;
  PyObject* seq_obj;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O(bbb)(bbb)", (char**)kwlist,
        &seq_obj, &intermediate_pos.r, &intermediate_pos.x, &intermediate_pos.y,
        &final_pos.r, &final_pos.x, &final_pos.y)) {
    return nullptr;
  }
  try {
    auto seq = ArrayToFrameSequence(seq_obj).value();
    self->tetris.FinishAdjSequence(seq, intermediate_pos, final_pos);
    return FrameSequenceToArray(seq);
  } catch (std::bad_optional_access&) {
    return nullptr;
  }
}
#endif // !NO_ROTATION

PyObject* Tetris_IsOver(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyBool_FromLong((long)self->tetris.IsOver());
}

PyObject* Tetris_IsEpisodeOver(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyBool_FromLong((long)self->IsEpisodeOver());
}

PyObject* Tetris_IsTopOut(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyBool_FromLong((long)self->IsTopOut());
}

PyObject* Tetris_ReachedMilestone(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyBool_FromLong((long)self->ReachedMilestone());
}

PyObject* Tetris_GetMilestone(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyLong_FromLong(self->Milestone());
}

PyObject* Tetris_GetRunTetrises(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyLong_FromLong(self->tetris.RunTetrises());
}

PyObject* Tetris_GetBoard(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  PythonBoard* board = PyObject_New(PythonBoard, &py_board_class);
  new(board) PythonBoard(self->tetris.GetBoard());
  return reinterpret_cast<PyObject*>(board);
}

PyObject* Tetris_GetLines(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyLong_FromLong(self->tetris.GetLines());
}

PyObject* Tetris_GetPieces(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyLong_FromLong(self->tetris.GetPieces());
}

PyObject* Tetris_GetNowPiece(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyLong_FromLong(self->tetris.NowPiece());
}

PyObject* Tetris_GetNextPiece(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyLong_FromLong(self->tetris.NextPiece());
}

PyObject* Tetris_GetRunScore(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyLong_FromLong(self->tetris.RunScore());
}

PyObject* Tetris_GetRunLines(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyLong_FromLong(self->tetris.RunLines());
}

PyObject* Tetris_GetRunPieces(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyLong_FromLong(self->tetris.RunPieces());
}

PyObject* Tetris_GetRealPosition(PythonTetris* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"pos", nullptr};
  Position pos;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(bbb)", (char**)kwlist, &pos.r, &pos.x, &pos.y)) {
    return nullptr;
  }
  return PositionToTuple(self->GetRealPosition(pos));
}

PyObject* Tetris_IsNoro(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
#ifdef NO_ROTATION
  return PyBool_FromLong(1);
#else
  return PyBool_FromLong(0);
#endif // NO_ROTATION
}

PyObject* Tetris_IsTetrisOnly(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyBool_FromLong((int)kTetrisOnly);
}

PyObject* Tetris_LineCap(PythonTetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyLong_FromLong(kLineCap);
}

PyMethodDef py_tetris_class_methods[] = {
    {"IsOver", (PyCFunction)Tetris_IsOver, METH_NOARGS,
     "Check whether the game is over (top out, or past the build's line cap)"},
    {"IsEpisodeOver", (PyCFunction)Tetris_IsEpisodeOver, METH_NOARGS,
     "Check whether the episode ended -- either terminal, TOP_OUT or MILESTONE"},
    {"IsTopOut", (PyCFunction)Tetris_IsTopOut, METH_NOARGS,
     "Check whether the episode ended in TOP_OUT (the constraint's event)"},
    {"ReachedMilestone", (PyCFunction)Tetris_ReachedMilestone, METH_NOARGS,
     "Check whether the line counter reached the milestone"},
    {"GetMilestone", (PyCFunction)Tetris_GetMilestone, METH_NOARGS,
     "Get the milestone line count"},
    {"InputPlacement", (PyCFunction)Tetris_InputPlacement, METH_VARARGS | METH_KEYWORDS,
     "Input a placement and return the reward"},
#ifndef NO_ROTATION
    {"DirectPlacement", (PyCFunction)Tetris_DirectPlacement, METH_VARARGS | METH_KEYWORDS,
     "Input a placement (skip pre-adj) and return the reward"},
#endif // !NO_ROTATION
    {"SetNextPiece", (PyCFunction)Tetris_SetNextPiece, METH_VARARGS | METH_KEYWORDS,
     "Set the next piece"},
    {"SetLines", (PyCFunction)Tetris_SetLines, METH_VARARGS | METH_KEYWORDS,
     "Set lines"},
    {"SetMilestone", (PyCFunction)Tetris_SetMilestone, METH_VARARGS | METH_KEYWORDS,
     "Set the absolute line count that ends an episode with no cost (0 = the line cap)"},
    {"Reset", (PyCFunction)Tetris_Reset, METH_VARARGS | METH_KEYWORDS,
     "Reset game and assign pieces randomly"},
#ifdef NO_ROTATION
    {"ResetRandom", (PyCFunction)Tetris_ResetRandom, METH_VARARGS | METH_KEYWORDS,
     "Reset game and assign pieces randomly"},
#endif // !NO_ROTATION
    {"GetState", (PyCFunction)Tetris_GetState, METH_VARARGS | METH_KEYWORDS,
     "Get state tuple"},
    {"StateShapes", (PyCFunction)Tetris_StateShapes, METH_NOARGS | METH_STATIC,
     "Get shapes of state array (static)"},
    {"StateTypes", (PyCFunction)Tetris_StateTypes, METH_NOARGS | METH_STATIC,
     "Get types of state array (static)"},
    {"GetSequence", (PyCFunction)Tetris_GetSequence, METH_VARARGS | METH_KEYWORDS,
     "Get frame sequence to a particular position"},
#ifndef NO_ROTATION
    {"GetAdjStates", (PyCFunction)Tetris_GetAdjStates, METH_VARARGS | METH_KEYWORDS,
     "Get state tuple for every possible next piece"},
    {"IsAdjMove", (PyCFunction)Tetris_IsAdjMove, METH_VARARGS | METH_KEYWORDS,
     "Check if a move can have adjustments"},
    {"IsNoAdjMove", (PyCFunction)Tetris_IsNoAdjMove, METH_VARARGS | METH_KEYWORDS,
     "Check if a move cannot have adjustments"},
    {"GetAdjPremove", (PyCFunction)Tetris_GetAdjPremove, METH_VARARGS | METH_KEYWORDS,
     "Get pre-adjustment placement and frame sequence by possible final destinations"},
    {"FinishAdjSequence", (PyCFunction)Tetris_FinishAdjSequence, METH_VARARGS | METH_KEYWORDS,
     "Finish a pre-adjustment sequence"},
#endif // !NO_ROTATION
    {"GetRealPosition", (PyCFunction)Tetris_GetRealPosition, METH_VARARGS | METH_KEYWORDS,
     "Get real (possibly mirrored) position"},
    {"GetBoard", (PyCFunction)Tetris_GetBoard, METH_NOARGS, "Get board object"},
    {"GetLines", (PyCFunction)Tetris_GetLines, METH_NOARGS, "Get total lines"},
    {"GetPieces", (PyCFunction)Tetris_GetPieces, METH_NOARGS, "Get total pieces"},
    {"GetNowPiece", (PyCFunction)Tetris_GetNowPiece, METH_NOARGS, "Get current piece"},
    {"GetNextPiece", (PyCFunction)Tetris_GetNextPiece, METH_NOARGS, "Get next piece"},
    {"GetRunScore", (PyCFunction)Tetris_GetRunScore, METH_NOARGS, "Get score of this run"},
    {"GetRunLines", (PyCFunction)Tetris_GetRunLines, METH_NOARGS, "Get lines of this run"},
    {"GetRunPieces", (PyCFunction)Tetris_GetRunPieces, METH_NOARGS, "Get pieces of this run"},
    {"GetRunTetrises", (PyCFunction)Tetris_GetRunTetrises, METH_NOARGS,
     "Get the number of tetrises cleared in this run"},
    {"IsNoro", (PyCFunction)Tetris_IsNoro, METH_NOARGS | METH_STATIC, "Check if noro flag is on"},
    {"IsTetrisOnly", (PyCFunction)Tetris_IsTetrisOnly, METH_NOARGS | METH_STATIC, "Check if tetris only flag is on"},
    {"LineCap", (PyCFunction)Tetris_LineCap, METH_NOARGS | METH_STATIC, "Get line cap"},
    {nullptr}};

} // namespace

PyTypeObject py_tetris_class = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "tetris.Tetris",         // tp_name
    sizeof(PythonTetris),    // tp_basicsize
    0,                       // tp_itemsize
    (destructor)TetrisDealloc, // tp_dealloc
    0,                       // tp_print
    0,                       // tp_getattr
    0,                       // tp_setattr
    0,                       // tp_reserved
    0,                       // tp_repr
    0,                       // tp_as_number
    0,                       // tp_as_sequence
    0,                       // tp_as_mapping
    0,                       // tp_hash
    0,                       // tp_call
    0,                       // tp_str
    0,                       // tp_getattro
    0,                       // tp_setattro
    0,                       // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // tp_flags
    "Tetris class",          // tp_doc
    0,                       // tp_traverse
    0,                       // tp_clear
    0,                       // tp_richcompare
    0,                       // tp_weaklistoffset
    0,                       // tp_iter
    0,                       // tp_iternext
    py_tetris_class_methods, // tp_methods
    0,                       // tp_members
    0,                       // tp_getset
    0,                       // tp_base
    0,                       // tp_dict
    0,                       // tp_descr_get
    0,                       // tp_descr_set
    0,                       // tp_dictoffset
    (initproc)TetrisInit,    // tp_init
    // PythonTetris embeds an alignas(32) Board and PyObject_Malloc only guarantees 16, so
    // the default tp_alloc hands back a misaligned object every other time and the move
    // search's aligned AVX2 loads fault.  See AlignedAlloc in python.h.
    AlignedAlloc<PythonTetris>, // tp_alloc
    TetrisNew,               // tp_new
    AlignedFree<PythonTetris>, // tp_free
};
