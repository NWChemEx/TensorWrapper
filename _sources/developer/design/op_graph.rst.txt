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

.. _tw_designing_the_opgraph:

#####################
Designing the OpGraph
#####################

**********************
OpGraph Considerations
**********************

Multiple source nodes
   The graph may have multiple inputs. Each input is assumed to be a
   ``TensorWrapper`` object.

Multiple sink nodes
   The graph may have multiple results. We assume the results have different
   types based on whether or not they are terminating the graph. The graph
   terminates on a ``TensorWrapper`` object.

References to actual data
   The ``OpGraph`` must either have handles to the actual data, or somehow be
   associated with the actual data in order for backends to translate the AST
   into results.

*******************
OpGraph Motivations
*******************

.. figure:: assets/graph_example.png
   :align: center

   Examples of some tensor graphs arising in electronic structure theory.
