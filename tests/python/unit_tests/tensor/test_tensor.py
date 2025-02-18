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
import unittest


class TestTensor(unittest.TestCase):

    def test_ctor(self):
        self.assertEqual(self.scalar.rank(), 0)
        self.assertEqual(self.vector.rank(), 1)
        self.assertEqual(self.matrix.rank(), 2)

    def setUp(self):
        self.scalar = tensorwrapper.Tensor(3.14)
        self.vector = tensorwrapper.Tensor([1.1, 2.2, 3.3])
        self.matrix = tensorwrapper.Tensor([[1.1, 2.2], [3.3, 4.4]])
