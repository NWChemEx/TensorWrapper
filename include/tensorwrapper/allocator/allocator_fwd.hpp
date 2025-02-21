#pragma once

namespace tensorwrapper::allocator {

class AllocatorBase;

template<typename FloatType, unsigned short Rank>
class Eigen;

class Local;

class Replicated;

template<typename FloatType>
class Contiguous;

} // namespace tensorwrapper::allocator