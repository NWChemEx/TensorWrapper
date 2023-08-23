.. _einstein_summation_convention:

#############################
Einstein Summation Convention
#############################

The purpose of this page is to give a primer on the Einstein summation
conventions and to point out why they are useful and when they can't be used.

**********
Motivation
**********

.. |n| replace:: :math:`n`
.. |u| replace:: :math:`\mathbf{u}`
.. |v| replace:: :math:`\mathbf{v}`
.. |M| replace:: :math:`\mathbf{M}`
.. |A| replace:: :math:`\mathbf{A}`
.. |B| replace:: :math:`\mathbf{B}`
.. |C| replace:: :math:`\mathbf{C}`

Many tensor operations involve summing over pairs of repeated indices. For
example, the inner-product between two vectors, |u| and |v|, looks like:

.. math::

   c = \sum_{i} u_i v_i.

Other prominent examples include the product of a vector |v| with a matrix |M|,
:math:`\mathbf{vM}`:

.. math::

   \left[\mathbf{vM}\right]_j = \sum_{i} v_i M_{ij}

and the matrix-matrix product, |C|, between two matrices |A| and |B|:

.. math::

   C_{ij} = \sum_{k} A_{ik}B_{kj}.

Given the prevalence of such tensor operations people wanting to save some
writing/typing often forgo the explicit summation symbols and agree on a
summation convention which says that pairs of repeated indices appearing in a
term are summed over. This summation convention was brought to physics by Albert
Einstein in his work on general relativity and is thus commonly known as the
Einstein summation convention. Using the Einstein summation convention the
above three equations would look like:

.. math::

   c =& u_i v_i\\
   \left[\mathbf{vM}\right]_j =& \sum_{i} v_i M_{ij}\\
   C_{ij} =& \sum_{k} A_{ik}B_{kj}

The traditional Einstein summation convention doesn't allow for element-wise
products like (note there is no implicit summation in the next equation):

.. math::

   C_{ij} = A_{ij}B_{ij}.

Making the observation that indices which are summed over only appear on one
side of the equation. We can define a "generalized Einstein summation
convention" which says that if an index only appears on side of an equation it
is implicitly summed over. Note that this also relaxes the "pair"
restriction so summing over a row of a matrix to form a vector, *i.e.*,

.. math::

   v_j = \sum_{i} A_{ij}

could be written using the generalized Einstein summation convention as:

.. math::

   v_j = A_{ij}.

For the purposes of TensorWrapper, generalized Einstein summation convention
allows us to write many tensor operations in a user-friendly manner. For example
the above equations

.. code-block:: c++

   // Vector-vector inner-product
   TensorWrapper c, u, v;
   c("") = u("i") * v("i");

   // Vector-matrix product
   TensorWrapper M, vM;
   vM("j") = v("i") * M("i,j");

   // Matrix-matrix product
   TensorWrapper C, A, B;
   C("i,j") = A("i,k") * B("k,j");

   // Element-wise matrix product
   C("i,j") = A("i,j") * B("i,j");

   // Summing over a row of a matrix
   v("j") = A("i,j");

***********
Limitations
***********

The conventional Einstein summation convention is limited to pairs of repeated
indices because

.. math::

   A_{ij}B_{ij}C_{ij}
