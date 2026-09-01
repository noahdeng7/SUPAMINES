#!/usr/bin/env python3

import os
import tempfile
from pathlib import Path
from setuptools import Extension, setup
from setuptools.errors import CompileError, LinkError
from setuptools.command.build_ext import build_ext

import numpy

_EXT = 'training_env/tetris'
_CORE = 'training_env/core'
sources = [f'{_EXT}/{f}' for f in
           ('tetris.cpp', 'state.cpp', 'board.cpp', 'batch.cpp', 'supervised_reader.cpp', 'module.cpp')]
sources += [f'{_CORE}/{f}' for f in ('tetris.cpp', 'frame_sequence.cpp', 'files.cpp')]
# The C init symbol is PyInit_tetris, so the extension's last component must stay 'tetris'.
NAME = 'training_env.tetris.tetris'

# Game variant defines. Override with e.g. TETRIS_DEFINES="LINE_CAP=290 TETRIS_ONLY".
DEFINES = os.environ.get('TETRIS_DEFINES', 'LINE_CAP=430').split()
# The board is 4x uint64 (32B, 32B-aligned) and the move search is pdep/pext-heavy, so the
# ISA level dominates throughput.  x86-64-v3 == AVX2 + BMI1/2 + LZCNT + FMA (Haswell / Zen1+).
# Override with TETRIS_ARCH=native for a machine-specific build, or e.g. TETRIS_ARCH=x86-64-v2.
ARCH = os.environ.get('TETRIS_ARCH', 'x86-64-v3')


def _unix_flags():
    flags = ['-std=c++20', '-O3', '-DNDEBUG', '-fno-math-errno', '-fvisibility=hidden']
    if ARCH:
        flags.append(f'-march={ARCH}')
    return flags + [f'-D{d}' for d in DEFINES]


def _msvc_flags():
    flags = ['/std:c++20', '/O2', '/Ob3', '/DNDEBUG', '/fp:fast', '/EHsc', '/GS-']
    if ARCH:  # MSVC has no -march; /arch:AVX2 implies BMI1/2 + LZCNT + FMA
        flags.append('/arch:AVX2' if ARCH in ('native', 'x86-64-v3', 'x86-64-v4') else '/arch:SSE2')
    return flags + [f'/D{d}' for d in DEFINES]


class build_ext_ex(build_ext):
    extra_compile_args = {
        NAME: {
            'unix': _unix_flags(),
            'msvc': _msvc_flags(),
        }
    }
    extra_link_args = {
        NAME: {
            'unix': ['-O3'],
            'msvc': [],
        }
    }

    def _zstd_found(self) -> bool:
        code = r"""
        int ZSTD_versionNumber(void);
        int main(void) {
            return ZSTD_versionNumber() == 0;
        }
        """
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src = td_path / "check_zstd.c"
            src.write_text(code, encoding="utf-8")
            try:
                objects = self.compiler.compile([str(src)], output_dir=str(td_path))
                self.compiler.link_executable(
                    objects,
                    str(td_path / "check_zstd"),
                    libraries=["zstd"],
                )
            except (CompileError, LinkError, OSError):
                return False
        return True

    def build_extension(self, ext):
        zstd_found = self._zstd_found()
        if zstd_found:
            ext.libraries = list(ext.libraries or []) + ["zstd"]
        else:
            print("zstd not found; you won't be able to run training with supervised data")

        ctype = self.compiler.compiler_type
        extra_args = self.extra_compile_args.get(ext.name)
        if extra_args is not None:
            ext.extra_compile_args = extra_args.get(ctype, [])
        link_args = self.extra_link_args.get(ext.name)
        if link_args is not None:
            ext.extra_link_args = link_args.get(ctype, [])

        build_ext.build_extension(self, ext)

module = Extension(
    NAME,
    sources=sources,
    libraries=[],
    include_dirs=[numpy.get_include()],
)
setup(ext_modules=[module], cmdclass={'build_ext': build_ext_ex})
