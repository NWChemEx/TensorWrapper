.. Copyright 2023 NWChemEx-Project
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
.. http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

########################
Motivating TensorWrapper
########################

************
Why Tensors?
************

.. |N| replace:: :math:`N`

TensorWrapper defines a tensor as a multi-dimensional array of values. The
values are usually scalars, but in general can be other tensors. In practice
this means that more-or-less any indexed quantity is a tensor (according to
TensorWrapper). Given the prevalence of indexed-quantities in physics
equations, tensors can be seen as a natural domain-specific language (DSL)
for physics, since most physics theories can be expressed succinctly in terms
of tensors. As a perfect example consider the energy, :math:`E`, of an
|N|-dimensional harmonic oscillator:

.. math::

   E = \frac{1}{2}\sum_{i=1}^N k_i \left(r_i - r^{(0)}_i\right)^2

where :math:`k_i` and :math:`\left(r_i-r^{(0)}_i\right)` respectively are the
force constant and displacement in the :math:`i`-th dimension. This is a perfect
example because the equation is simple to express in terms of tensors, but
few people would actually code this up as a tensor equation, which brings us
to...

*****************************
Are tensors a sufficient DSL?
*****************************

In practice, most people would compute :math:`E` something like:

.. code-block:: c++

   double e = 0.0;
   for(auto i = 0; i < N; ++i){
       auto dr = r[i] - r0[i];
       e+= k[i] * dr * dr;
   }
   return 0.5 * e;

That is they'd use a for-loop. Why? Consider pseudo code for the tensor-based
version:

.. code-block:: c++

   Tensor dr, dr2, e;
   dr("i")  = r("i") - r0("i");
   dr2("i") = dr("i") * dr("i");
   e = 0.5 * k("i") * dr2("i");

Ignoring compiler optimizations, and taking the code at face value the
operation counts are the same for each algorithm (|N|
subtractions, |N| squares, and an |N|-element dot-product). The primary
difference is that the tensor version stores two additional |N|-element vectors
(the displacements and the square of the displacements). While not strictly
inferrable from the code presented here, assuming ``r``,
``r0``, and ``k`` are something akin to a ``double *``, it is also quite
likely that the compiler will be able to better optimize the loop-based
version.

So if the for-loop produces better overall code, why would we want the DSL
version? The short answer is the for-loop based version is not optimal in
all situations. By using the DSL, we can express our intent while punting the
optimizations to the backend. This is because, using modern object-oriented
programming techniques, there is a disconnect between the apparent API and how
things are actually implemented. Point being, just because the tensor DSL code
looks like it has extra memory usage doesn't mean it really does (simple
reference counting techniques could identify ``dr`` and ``dr2`` as
intermediates). Furthermore it's entirely possible that, as written, the
implementation underlying the tensor DSL is parallelized and/or GPU
accelerated. The loop-based version is, however, not parallelized, and would
need to be rewritten for threading and or process parallelization. This can
potentially be a lot of work if multiple loop-based implementations need to be
rewritten.

******************
Why TensorWrapper?
******************

In short, there's no tensor library out there (that we can find) that really
strives to achieve the full-featured DSL we want. While a number of tensor
libraries have a DSL, the libraries assume you're willing to drop out of the
DSL for more complicated optimizations. In practice, these are precisely the
optimizations we are interested in.
