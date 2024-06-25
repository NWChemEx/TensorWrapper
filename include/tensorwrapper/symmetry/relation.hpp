#pragma once
#include <memory>

namespace tensorwrapper::symmetry {

/** @brief Common API for classes describing a symmetry relation.
 *
 *  The Group class interacts with the elements of the group through a common
 *  API. This class defines that API.
 */
class Relation {
public:
    /// Common base class for all symmetry relations
    using base_type = Relation;

    /// Type of a pointer to a symmetry relation's base class
    using base_pointer = std::unique_ptr<Relation>;

    /// Type used to index tensor modes
    using mode_index_type = unsigned short;

    // -------------------------------------------------------------------------
    // -- Ctors, assignment, and dtor
    // -------------------------------------------------------------------------

    /// Defaulted no-throw dtor
    virtual ~Relation() noexcept = default;

    /** @brief Polymorphic copy constructor.
     *
     *  Derived classes implement this method by overriding clone_
     */
    base_pointer clone() const { return clone_(); }

protected:
    virtual base_pointer clone_() const = 0;
};

} // namespace tensorwrapper::symmetry
