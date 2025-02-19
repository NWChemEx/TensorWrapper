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

import tensorwrapper
import numpy as np
import py_test_tensorwrapper.testing as testing
import unittest


class TestTensor(unittest.TestCase):

    def test_ctor(self):
        self.assertEqual(self.scalar.rank(), 0)
        self.assertEqual(self.vector.rank(), 1)
        self.assertEqual(self.matrix.rank(), 2)

    def test_numpy(self):
        np_scalar = np.array(self.scalar)
        np_vector = np.array(self.vector)
        np_matrix = np.array(self.matrix)

        scalar_corr = np.array(42)
        vector_corr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        matrix_corr = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertFalse((np_scalar - scalar_corr).any())
        self.assertFalse((np_vector - vector_corr).any())
        self.assertFalse((np_matrix - matrix_corr).any())

    def setUp(self):
        self.scalar = testing.get_scalar()
        self.vector = testing.get_vector()
        self.matrix = testing.get_matrix()
