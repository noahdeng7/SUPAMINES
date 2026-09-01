#include "files.h"

#include "board.h"

namespace fs = std::filesystem;

uint64_t BoardCount(const fs::path& board_file) {
  return fs::file_size(board_file) / kBoardBytes;
}

bool MkdirForFile(fs::path path) {
  auto x = path.remove_filename();
  if (x.empty()) return true;
  return fs::create_directories(path.remove_filename());
}
