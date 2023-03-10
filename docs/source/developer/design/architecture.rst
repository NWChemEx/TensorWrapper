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

#####################
Proposed Tensor Stack
#####################

TensorWrapper was motivated by the needs of electronic structure theory (EST).
EST involves a number of complicated equations involving high-rank tensors. To
the extent that EST is "just" preparing initial tensor state followed by
evaluating the resulting tensor equations, this places a tremendous burden on
the tensor stack, which in turn must be able to solve the equations subject to
the considerations raised in Section :ref:`tw_considerations`. To this end we
envision the full tensor stack as having several layers, which are shown in
Fig. :numref:`fig_tensor_stack`.

.. _fig_tensor_stack:

.. figure:: assets/tensor_stack.png
   :align: center

   Proposed tensor stack. TensorWrapper is envisioned as living in the box
   labeled "Tensor DSL".

At the top of Fig. :numref:`fig_tensor_stack` is a box labeled "Compile Time/
Code Generator" which includes mechanisms for writing tensor
expressions. In EST, many tensor expressions are auto-generated given the
second-quantized forms of the equations. Other expressions are simple enough
that they are written out by hand. Other physics disciplines may have other
means of generating tensor equations, regardless of what those means are they
fall in this top-box and the output is the tensor DSL, which is the entry point
into the bottom-box labeled "Runtime".
