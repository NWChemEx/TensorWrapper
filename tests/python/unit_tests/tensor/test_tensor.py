# Copyright 2025 NWChemEx-Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import py_test_tensorwrapper.testing as testing

import tensorwrapper


class TestTensor(unittest.TestCase):
    def test_rank(self):
        with self.assertRaises(RuntimeError):
            self.defaulted.rank()
        self.assertEqual(self.scalar.rank(), 0)
        self.assertEqual(self.vector.rank(), 1)
        self.assertEqual(self.matrix.rank(), 2)
        self.assertEqual(self.scalar_from_cpp.rank(), 0)
        self.assertEqual(self.vector_from_cpp.rank(), 1)
        self.assertEqual(self.matrix_from_cpp.rank(), 2)

    def test_equality(self):
        self.assertTrue(self.scalar == self.scalar_from_cpp)
        self.assertTrue(self.vector == self.vector_from_cpp)
        self.assertTrue(self.matrix == self.matrix_from_cpp)

    def test_inequality(self):
        self.assertTrue(self.defaulted != self.scalar)

    def test_numpy(self):
        np_scalar = np.array(self.scalar)
        np_vector = np.array(self.vector)
        np_matrix = np.array(self.matrix)

        scalar_corr = np.array(42.0)
        vector_corr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        matrix_corr = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertFalse((np_scalar - scalar_corr).any())
        self.assertFalse((np_vector - vector_corr).any())
        self.assertFalse((np_matrix - matrix_corr).any())

    def setUp(self):
        self.defaulted = tensorwrapper.Tensor()
        self.scalar = tensorwrapper.Tensor(np.array(42.0))
        self.vector = tensorwrapper.Tensor(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
        self.matrix = tensorwrapper.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        self.scalar_from_cpp = testing.get_scalar()
        self.vector_from_cpp = testing.get_vector()
        self.matrix_from_cpp = testing.get_matrix()
