#pragma once

#define PY_SSIZE_T_CLEAN
#if __has_include(<python3.8/Python.h>)
#include <python3.8/Python.h>
#elif __has_include(<python3.9/Python.h>)
#include <python3.9/Python.h>
#elif __has_include(<python3.10/Python.h>)
#include <python3.10/Python.h>
#elif __has_include(<python3.11/Python.h>)
#include <python3.11/Python.h>
#elif __has_include(<python3.12/Python.h>)
#include <python3.12/Python.h>
#elif __has_include(<python3.13/Python.h>)
#include <python3.13/Python.h>
#else
#include <Python.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>

// --- Over-aligned Python objects ---------------------------------------------------------
//
// `PyObject_Malloc` guarantees only 16-byte alignment, but `core/board.h`'s `Board` is
// `BoardTmpl<32>` and the move search issues aligned AVX2 loads against it.  A type that
// embeds one therefore cannot use the default `tp_alloc`: on GCC/Linux the allocations come
// back alternating 32- and 16-byte aligned, and every other object faults -- at the
// constructor, or mid-rollout, or anywhere else, which reads like heap corruption and sends
// you hunting in the wrong place.  `-march=x86-64-v3` is a floor the move search needs, so
// lowering the ISA is not an option.  MSVC happens not to trip this, which is why it shipped.
//
// `AlignedAlloc` over-allocates, rounds the payload up to `alignof(T)`, and stashes the base
// pointer in the word below the object so `AlignedFree` can recover it.  Wire the pair into
// the type object as `tp_alloc` / `tp_free`.  Types that do *not* need it carry a
// `static_assert(!kNeedsAlignedAlloc<...>)` instead, so the day one of them gains an
// over-aligned member the build fails rather than segfaulting.

// What PyObject_Malloc actually guarantees.
inline constexpr size_t kPyObjectAlign = 16;

template <class T>
inline constexpr bool kNeedsAlignedAlloc = alignof(T) > kPyObjectAlign;

template <class T>
PyObject* AlignedAlloc(PyTypeObject* type, Py_ssize_t nitems) {
  constexpr size_t kAlign = alignof(T) < alignof(void*) ? alignof(void*) : alignof(T);
  const size_t size = (size_t)type->tp_basicsize + (size_t)nitems * (size_t)type->tp_itemsize;
  // Worst case the payload starts sizeof(void*) + kAlign - 1 bytes into the block.
  void* base = PyMem_RawMalloc(size + sizeof(void*) + kAlign);
  if (base == nullptr) return PyErr_NoMemory();
  uintptr_t payload =
      ((uintptr_t)base + sizeof(void*) + kAlign - 1) & ~(uintptr_t)(kAlign - 1);
  ((void**)payload)[-1] = base;
  PyObject* obj = (PyObject*)payload;
  memset(obj, 0, size);
  if (type->tp_itemsize) {
    PyObject_InitVar((PyVarObject*)obj, type, nitems);
  } else {
    PyObject_Init(obj, type);
  }
  return obj;
}

template <class T>
void AlignedFree(void* ptr) {
  if (ptr == nullptr) return;
  PyMem_RawFree(((void**)ptr)[-1]);
}
