#pragma once
#include <parallelzone/hasher.hpp>

// Functions and types needed for hashing
namespace tensorwrapper::detail_ {

// From ParallelZone
using pz::hash_objects;
using pz::Hasher;
using pz::HashType;
using pz::HashValue;
using pz::make_hash;

} // namespace tensorwrapper::detail_