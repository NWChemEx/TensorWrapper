#pragma once
#include "tensorwrapper/detail_/hashing.hpp"
#include "tensorwrapper/tensor/tensor_wrapper.hpp"
#include <Eigen/Dense>
#include <deque>

/** TODO: This should probably be moved to TensorWrapper at a point when things
 *  are more stable there. Also, this makes Eigen an explicit dependency of
 *  wherever it lives. Currently, we get Eigen implicitly because TiledArray
 *  needs it, but that's not an assumption we should be making.
 */

namespace tensorwrapper::diis {

/** @brief Performs Direct Inversion of the Iterative Subspace extrapolation.
 *
 *  This class performs DIIS extrapolation and stores the necessary previous
 *  values for the extrapolation.
 *
 */
class DIIS {
public:
    /// Type used for the number of value and error matrices stored
    using size_type = std::size_t;

    /// Type of the value and error matrices
    using tensor_type = tensorwrapper::tensor::ScalarTensorWrapper;

private:
    /// Type of the container that stores the value and error matrices
    using deque_type = std::deque<tensor_type>;

    /// Type of the coefficient and b vectors
    using vector_type = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    /// Type of the B and A matrices
    using matrix_type =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

public:
    /** @brief Initializes the object with the given vector size.
     *
     *  @param[in] max_vec The max number of previous values stored.
     *                     Default is 5.
     */
    DIIS(size_type max_vec = 5) :
      m_max_vec_(max_vec), m_B_(matrix_type::Zero(max_vec, max_vec)) {}

    /** @brief Performs DIIS extrapolation with the new value and error
     * matrices.
     *
     *  Appends @p X and @p E to the list of previous values and returns a new
     *  value extrapolated from the previous ones.
     *
     *  @param[in] X The current estimate of the desired value.
     *  @param[in] E The error associated with the current estimate.
     *
     *  @return The new extrapolated value as a tensor.
     */
    tensor_type extrapolate(const tensor_type& X, const tensor_type& E);

    /** @brief Determines if two DIIS instances are the same.
     *
     *  The two instances are equal if they have the same max vector size,
     *  the same previous values and errors, and the same intermediate B matrix.
     *
     *  @param[in] rhs The DIIS to compare against.
     *
     *  @return True if the DIIS instances has the same states. False otherwise.
     */
    bool operator==(const DIIS& rhs) const noexcept;

    /** @brief Adds a hash of the current DIIS to the provided hashing object.
     *
     *  @param[in,out] h The object being used for hashing. After this function
     *                   the internal state will be updated with a hash of this
     *                   object.
     */
    void hash(tensorwrapper::detail_::Hasher& h) const;

private:
    size_type m_max_vec_;
    deque_type m_x_values_;
    deque_type m_errors_;
    matrix_type m_B_;
};

inline bool operator!=(const DIIS& lhs, const DIIS& rhs) {
    return !(lhs == rhs);
}

} // end namespace tensorwrapper::diis
