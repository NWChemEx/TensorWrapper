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

#################
Creating a Tensor
#################

The entry point into TensorWrapper is creating a tensor. We break tensor 
creation into two pieces: 

#. Specifying the details of the tensor.
#. Translating the details into a literal tensor object.

The relevant pieces are summarized in Fig. :numref:`fig_tensor_creation`.


.. _fig_tensor_creation:

.. figure:: assets/creation.png
   :align: center

   Pieces needed to create a tensor.

Fig. :numref:`fig_tensor_creation` 