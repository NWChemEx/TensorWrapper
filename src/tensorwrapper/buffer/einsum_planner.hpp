#pragma once
#include <tensorwrapper/dsl/dummy_indices.hpp>

namespace tensorwrapper::buffer {

/** @brief Works out the details pertaining to an arbitrary binary einsum op.
 *
 *  For a general einsum operation the indices in a label fall into one of four
 *  categories:
 *
 *  - trace indices: appear in only one of the input tensors, but not the output
 *  - dummy indices: those that appear in both input tensors, but not the output
 *  - free indices: appear in result and ONE of the input tensors
 *  - batch indices: appear in all three tensors
 *
 *  N.b., though the set of indices in say lhs_batch and rhs_batch must be the
 *  same, the order can be different. This applies to dummy indices too.
 */
class EinsumPlanner {
public:
    using string_type = std::string;

    using label_type = dsl::DummyIndices<string_type>;

    EinsumPlanner(std::string result, std::string lhs, std::string rhs) :
      EinsumPlanner(label_type(result), label_type(lhs), label_type(rhs)) {}

    EinsumPlanner(label_type result, label_type lhs, label_type rhs) :
      m_result_(std::move(result)),
      m_lhs_(std::move(lhs)),
      m_rhs_(std::move(rhs)) {}

    // Labels that ONLY appear in LHS
    label_type lhs_trace() const {
        return m_lhs_.difference(m_rhs_).difference(m_result_);
    }

    /// Labels that ONLY appear in RHS
    label_type rhs_trace() const {
        return m_rhs_.difference(m_lhs_).difference(m_result_);
    }

    /// Labels that appear in both LHS and RHS, but NOT in result
    label_type lhs_dummy() const {
        return m_lhs_.intersection(m_rhs_).difference(m_result_);
    }

    /// Labels that appear in both LHS and RHS, but NOT in result
    label_type rhs_dummy() const {
        return m_rhs_.intersection(m_lhs_).difference(m_result_);
    }

    /// Labels that appear in result and LHS, but NOT in RHS
    label_type lhs_free() const {
        return m_lhs_.intersection(m_result_).difference(m_rhs_);
    }

    /// Labels that appear in result and RHS, but NOT in LHS
    label_type rhs_free() const {
        return m_rhs_.intersection(m_result_).difference(m_lhs_);
    }

    /// Labels that appear in all three tensors
    label_type lhs_batch() const {
        return m_lhs_.intersection(m_result_).intersection(m_rhs_);
    }

    /// Labels that appear in all three tensors
    label_type rhs_batch() const {
        return m_rhs_.intersection(m_result_).intersection(m_lhs_);
    }

private:
    label_type m_result_;
    label_type m_lhs_;
    label_type m_rhs_;
};

} // namespace tensorwrapper::buffer