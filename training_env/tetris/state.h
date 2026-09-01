#pragma once

#include <array>
#include <vector>
#include <cstdint>
#include "../core/tetris.h"

typedef struct _object PyObject;

// Board / move planes are strictly binary, so they are stored as uint8 rather than float32:
// 4x less memory to fill, 4x less host->device traffic, and 4x smaller rollout buffers.
// The model casts them to float on the GPU.
constexpr int kPlaneSize = 200; // 20 * 10

#ifdef NO_ROTATION
constexpr int kBoardPlanes = 2;
constexpr int kMovePlanes = 3;
constexpr int kMetaSize = 32;
constexpr int kMoveMetaSize = 31;
constexpr int kActionRotations = 1;
#else // !NO_ROTATION
constexpr int kBoardPlanes = 6;
constexpr int kMovePlanes = 18;
constexpr int kMetaSize = 32;
constexpr int kMoveMetaSize = 28;
constexpr int kActionRotations = 4;
#endif // !NO_ROTATION
constexpr int kMetaIntSize = 2;
// an action index is rotation * 200 + row * 10 + col
constexpr int kActionSize = kActionRotations * kPlaneSize;

// Destination for a single state. Points into caller-owned memory (the rollout staging
// buffers), so state generation writes its result exactly once, with no allocation.
struct StateView {
  uint8_t* board;      // kBoardPlanes * kPlaneSize
  float* meta;         // kMetaSize
  uint8_t* moves;      // kMovePlanes * kPlaneSize
  float* move_meta;    // kMoveMetaSize
  int32_t* meta_int;   // kMetaIntSize
};

struct State {
  std::array<uint8_t, kBoardPlanes * kPlaneSize> board;
  std::array<float, kMetaSize> meta;
  std::array<uint8_t, kMovePlanes * kPlaneSize> moves;
  std::array<float, kMoveMetaSize> move_meta;
  std::array<int32_t, kMetaIntSize> meta_int;

  StateView View() { return {board.data(), meta.data(), moves.data(), move_meta.data(), meta_int.data()}; }
  PyObject* ToPython() const;
};

extern std::vector<long> kStateShapes[5];

// X macro for state members
#define STATE_MEMBERS_ \
  X(board, 0, 3, NPY_UINT8) \
  X(meta, 1, 1, NPY_FLOAT32) \
  X(moves, 2, 3, NPY_UINT8) \
  X(move_meta, 3, 1, NPY_FLOAT32) \
  X(meta_int, 4, 1, NPY_INT32)

struct MultiState {
#define X(name, id, dims, typ) std::vector<decltype(State::name)> name;
  STATE_MEMBERS_
#undef X
  void reserve(size_t sz) {
#define X(name, id, dims, typ) name.reserve(sz);
    STATE_MEMBERS_
#undef X
  }
  void resize(size_t sz) {
#define X(name, id, dims, typ) name.resize(sz);
    STATE_MEMBERS_
#undef X
  }
  void push_back(const State& st) {
    resize(size() + 1);
#define X(name, id, dims, typ) memcpy(name.back().data(), st.name.data(), sizeof(State::name));
    STATE_MEMBERS_
#undef X
  }
  void merge(const MultiState& st) {
    size_t offset = size(), sz = size() + st.size();
    resize(sz);
#define X(name, id, dims, typ) memcpy(name.data() + offset, st.name.data(), st.size() * sizeof(State::name));
    STATE_MEMBERS_
#undef X
  }
  void swap(MultiState& st) {
#define X(name, id, dims, typ) name.swap(st.name);
    STATE_MEMBERS_
#undef X
  }
  size_t size() const { return board.size(); }
  PyObject* ToPython() const;
};

static constexpr int kMirrorCols[] = {9, 9, 9, 10, 9, 9, 10};
static constexpr int kMirrorPiece[] = {0, 5, 4, 3, 2, 1, 6};
double GetNoroLineRewardExp(int lines, int start_level, bool do_tuck, bool nnb);

void GetState(const TetrisNoro& tetris, const StateView& state, bool nnb, bool is_mirror, int line_reduce = 0);
void GetState(const Tetris& tetris, const StateView& state, int line_reduce = 0);
MultiState GetAdjStates(const Tetris& tetris);
