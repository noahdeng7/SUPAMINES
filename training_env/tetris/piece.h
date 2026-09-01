#pragma once

#include <cstring>
#include "python.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL TETRIS_PY_ARRAY_SYMBOL_
#include <numpy/ndarrayobject.h>

inline int ParsePieceID(PyObject* obj) {
  if (PyUnicode_Check(obj)) {
    if (PyUnicode_GET_LENGTH(obj) < 1) {
      PyErr_SetString(PyExc_KeyError, "Invalid piece symbol.");
      return -1;
    }
    switch (PyUnicode_READ_CHAR(obj, 0)) {
      case 'T': return 0;
      case 'J': return 1;
      case 'Z': return 2;
      case 'O': return 3;
      case 'S': return 4;
      case 'L': return 5;
      case 'I': return 6;
      default: {
        PyErr_SetString(PyExc_KeyError, "Invalid piece symbol.");
        return -1;
      }
    }
  } else if (PyLong_Check(obj)) {
    long x = PyLong_AsLong(obj);
    if (x < 0 || x >= 7) {
      PyErr_SetString(PyExc_IndexError, "Piece ID out of range.");
      return -1;
    }
    return x;
  } else {
    PyErr_SetString(PyExc_TypeError, "Invalid type for piece.");
    return -1;
  }
}

// Reads a 10-element tap sequence from a Python list or an int array.
// Returns false with a Python exception set on failure.
inline bool ParseTapSequence(PyObject* obj, int out[10]) {
  if (PyList_Check(obj) || PyTuple_Check(obj)) {
    PyObject* seq = PySequence_Fast(obj, "tap_sequence");
    if (!seq) return false;
    bool ok = PySequence_Fast_GET_SIZE(seq) == 10;
    if (!ok) {
      PyErr_SetString(PyExc_TypeError, "tap_sequence length should be 10");
    } else {
      for (int i = 0; i < 10; i++) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        if (!PyLong_Check(item)) {
          PyErr_SetString(PyExc_TypeError, "tap_sequence must contain integers");
          ok = false;
          break;
        }
        out[i] = PyLong_AsLong(item);
      }
    }
    Py_DECREF(seq);
    if (!ok) return false;
  } else if (PyArray_Check(obj)) {
    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(obj);
    if (PyArray_TYPE(array) != NPY_INT || !PyArray_ISCONTIGUOUS(array) || PyArray_SIZE(array) != 10) {
      PyErr_SetString(PyExc_TypeError, "tap_sequence must be 10 contiguous ints");
      return false;
    }
    memcpy(out, PyArray_DATA(array), 10 * sizeof(int));
  } else {
    PyErr_SetString(PyExc_TypeError, "tap_sequence must be a list or array");
    return false;
  }
  for (int i = 1; i < 10; i++) {
    if (out[i] - out[i - 1] < 2) {
      PyErr_SetString(PyExc_TypeError, "Invalid tap sequence");
      return false;
    }
  }
  return true;
}
