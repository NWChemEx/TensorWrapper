#pragma once
#include <tensorwrapper/dsl/dummy_indices.hpp>

namespace tensorwrapper::buffer {

/** @brief Class for working out details pertaining to a tensor contraction.
 *
 *
 */
class ContractionPlanner {
public:
    using string_type = std::string;
    using label_type  = dsl::DummyIndices<string_type>;

    ContractionPlanner(string_type result, string_type lhs, string_type rhs) :
      ContractionPlanner(label_type(result), label_type(lhs), label_type(rhs)) {
    }

    ContractionPlanner(label_type result, label_type lhs, label_type rhs) :
      m_result_(std::move(result)),
      m_lhs_(std::move(lhs)),
      m_rhs_(std::move(rhs)) {}

    /// Labels in LHS that are NOT summed over
    label_type lhs_free() const { return m_lhs_.intersection(m_result_); }

    /// Labels in RHS that are NOT summed over
    label_type rhs_free() const { return m_rhs_.intersection(m_result_); }

    /// Labels in LHS that ARE summed over
    label_type lhs_dummy() const { return m_lhs_.difference(m_result_); }

    /// Labels in RHS that ARE summed over
    label_type rhs_dummy() const { return m_rhs_.difference(m_result_) }

private:
    label_type m_result_;
    label_type m_lhs_;
    label_type m_rhs_;
};

} // namespace tensorwrapper::buffer