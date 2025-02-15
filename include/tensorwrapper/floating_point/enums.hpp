#pragma once

namespace tensorwrapper::fp {

/// Enums of supported floating point types
enum class FloatingPoint { Float32, Float64, UFloat32, UFloat64 };

/** @brief Converts @p FloatType to the corresponding enum.
 *
 *  @tparam FloatType Explicit template parameter used to specify the floating
 *                    point type to map to an enum.
 *
 *  Manipulating floating point types at runtime is easier with enums than with
 *  RTTI. This method will convert the type to the corresponding enum. Note that
 *  if UQ is not enabled, than uncertain_float/uncertain_double are float/double
 *  respectively and will map to Float32 and Float64 NOT UFloat32 and UFloat64.
 *
 *  @return The enum corresponding to @p FloatType.
 *
 *  @throw std::runtime_error if @p FloatType is NOT a supported floating point
 *                            type. Strong throw guarantee.
 */
template<typename FloatType>
auto convert_to_enum() {
    if constexpr(std::is_same_v<FloatType, float>) {
        return FloatingPoint::Float32;
    } else if constexpr(std::is_same_v<FloatType, double>) {
        return FloatingPoint::Float64;
    } else if constexpr(std::is_same_v<FloatType, type::uncertain_float>) {
        return FloatingPoint::UFloat32;
    } else if constexpr(std::is_same_v<FloatType, type::uncertain_double>) {
        return FloatingPoint::UFloat64;
    } else {
        throw std::runtime_error("Unregistered floating-point type");
    }
}

} // namespace tensorwrapper::fp