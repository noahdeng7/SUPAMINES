PYTHON ?= python3

# Build the tetris extension in place (training_env/tetris/tetris*.so|.pyd).
# Variant knobs, e.g.:  make TETRIS_DEFINES="LINE_CAP=290 TETRIS_ONLY"
#                       make TETRIS_ARCH=native
ext:
	$(PYTHON) setup.py build_ext --inplace

# Editable install so `import training_env` works from any directory.
install:
	$(PYTHON) -m pip install -e . --no-build-isolation

# Optional: board-curriculum sampler for RL's --board-file (see training_env/tools/).
random_boards:
	$(CXX) -std=c++20 -O3 -march=native -o random_boards training_env/tools/random_boards.cpp

clean:
	rm -rf build training_env.egg-info
	rm -f training_env/tetris/*.so training_env/tetris/*.pyd random_boards

.PHONY: ext install random_boards clean
