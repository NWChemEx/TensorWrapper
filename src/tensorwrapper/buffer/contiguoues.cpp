/*
 * Copyright 2025 NWChemEx-Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <tensorwrapper/buffer/contiguous.hpp>

namespace tensorwrapper::buffer {

#define DEFINE_CONTIG_BUFFER(TYPE) template class Contiguous<TYPE>

TW_APPLY_FLOATING_POINT_TYPES(DEFINE_CONTIG_BUFFER);

#undef DEFINE_CONTIG_BUFFER

} // namespace tensorwrapper::buffer