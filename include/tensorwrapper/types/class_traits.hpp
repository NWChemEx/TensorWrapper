#pragma once

namespace tensorwrapper::types {

/** @brief Defines the member types for the @p ClassType  class.
 *
 *  This class will serve as the single-source of truth for defining the member
 *  types for the @p ClassType class. The primary template is not defined and
 *  developers are expected to specialize the template for each @p ClassType
 *  in the TensorWrapper library.
 *
 *  @tparam ClassType The, possibly cv-qualified, type of the class which *this
 *                    defines the types for.
 */
template<typename ClassType>
struct ClassTraits;

} // namespace tensorwrapper::types