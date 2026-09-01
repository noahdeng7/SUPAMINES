#pragma once

#include <cstdint>
#include <filesystem>

// Board files are a flat array of CompactBoard; used to seed rollouts from a
// curriculum of starting positions (see training_env/tools/random_boards.cpp).
uint64_t BoardCount(const std::filesystem::path& board_file);

bool MkdirForFile(std::filesystem::path);
