# Copyright 2026 NWChemEx-Project
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

try:
    import sigma
except ImportError:
    sigma = None

# The fixtures only exist when TensorWrapper was built with ENABLE_SIGMA
has_sigma = sigma is not None and hasattr(testing, "get_interval_matrix")


@unittest.skipUnless(has_sigma, "TensorWrapper was not built with sigma")
class TestTensorSigma(unittest.TestCase):
    def check_round_trip(self, t, corr):
        """Asserts that np.array(t) is an object array equal to corr"""
        arr = np.array(t)
        self.assertEqual(arr.dtype, object)
        self.assertEqual(arr.shape, np.shape(corr))
        for x, y in zip(arr.flat, np.asarray(corr, dtype=object).flat):
            self.assertIs(type(x), type(y))
            self.assertEqual(x, y)

    def test_interval_from_list(self):
        t = tensorwrapper.Tensor(self.intervals)
        self.assertEqual(t.rank(), 2)
        self.assertTrue(t == testing.get_interval_matrix())
        self.check_round_trip(t, self.intervals)

    def test_interval_from_object_array(self):
        t = tensorwrapper.Tensor(np.array(self.intervals, dtype=object))
        self.assertTrue(t == testing.get_interval_matrix())

    def test_interval_scalar(self):
        i = sigma.IDouble(1.0, 2.0)
        t = tensorwrapper.Tensor(i)
        self.assertEqual(t.rank(), 0)
        self.check_round_trip(t, i)

    def test_interval_from_cpp(self):
        self.check_round_trip(testing.get_interval_matrix(), self.intervals)

    def test_triplex_from_list(self):
        t = tensorwrapper.Tensor(self.triplexes)
        self.assertEqual(t.rank(), 2)
        self.assertTrue(t == testing.get_triplex_matrix())
        self.check_round_trip(t, self.triplexes)

    def test_triplex_from_object_array(self):
        t = tensorwrapper.Tensor(np.array(self.triplexes, dtype=object))
        self.assertTrue(t == testing.get_triplex_matrix())

    def test_triplex_scalar(self):
        tp = sigma.TPDouble(0.9, 1.0, 1.1)
        t = tensorwrapper.Tensor(tp)
        self.assertEqual(t.rank(), 0)
        self.check_round_trip(t, tp)

    def test_triplex_from_cpp(self):
        t = testing.get_triplex_matrix()
        self.check_round_trip(t, self.triplexes)
        nominals = [x.nominal() for x in np.array(t).flat]
        self.assertEqual(nominals, [1.0, 2.0, 3.0, 4.0])

    def test_float_types(self):
        intervals = [sigma.IFloat(1.0, 2.0), sigma.IFloat(3.0, 4.0)]
        self.check_round_trip(tensorwrapper.Tensor(intervals), intervals)
        triplexes = [sigma.TPFloat(0.5, 1.0, 1.5), sigma.TPFloat(2.0)]
        self.check_round_trip(tensorwrapper.Tensor(triplexes), triplexes)

    def test_mixed_types_raise(self):
        mixed = [sigma.IDouble(1.0, 2.0), sigma.TPDouble(1.0)]
        with self.assertRaises(TypeError):
            tensorwrapper.Tensor(mixed)

    def test_unsupported_objects_raise(self):
        with self.assertRaises(RuntimeError):
            tensorwrapper.Tensor(["not", "a", "number"])

    def setUp(self):
        self.intervals = [
            [sigma.IDouble(0.5, 1.5), sigma.IDouble(1.5, 2.5)],
            [sigma.IDouble(2.5, 3.5), sigma.IDouble(3.5, 4.5)],
        ]
        self.triplexes = [
            [sigma.TPDouble(0.5, 1.0, 1.5), sigma.TPDouble(1.5, 2.0, 2.5)],
            [sigma.TPDouble(2.5, 3.0, 3.5), sigma.TPDouble(3.5, 4.0, 4.5)],
        ]
