#pragma once
#include <parallelzone/hasher.hpp>

// Functions and types needed for hashing
namespace tensorwrapper::detail_ {

// From ParallelZone
using parallelzone::hash_objects;
using parallelzone::Hasher;
using parallelzone::HashType;
using parallelzone::HashValue;
using parallelzone::make_hash;

} // namespace tensorwrapper::detail_