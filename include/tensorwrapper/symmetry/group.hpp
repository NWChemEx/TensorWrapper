#pragma once
#include <tensorwrapper/symmetry/relation.hpp>
#include <vector>

namespace tensorwrapper::symmetry {

/** @brief Container of the symmetry elements for a tensor.
 *
 *  Many tensors have elements which are related by symmetry. For example, a
 *  symmetric matrix is a matrix where the @f$(i,j)@f$-th element is the same
 *  as the @f$(j,i)@f$-th element. As the rank of the tensor increases, more
 *  symmetry relations are possible. The Group class models the set of symmetry
 *  operations which hold true for a given tensor.
 */
class Group {
public:
    /// The base type of each object in *this
    using value_type = Relation;

private:
    using value_pointer = Relation::base_type;

    using relation_container_type = std::vector<value_pointer>;

    relation_container_type m_relations_;
};

} // namespace tensorwrapper::symmetry
