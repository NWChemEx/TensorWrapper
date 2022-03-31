#pragma once
#include <tensorwrapper/tensor/detail_/backends/tiled_array.hpp>

namespace testing {

inline TA::TiledRange single_element_tiles(const std::vector<std::size_t>& ex) {
    std::vector<TA::TiledRange1> trs;
    for(auto e : ex) {
        std::vector<std::size_t> r(e+1); std::iota(r.begin(),r.end(),0);
	trs.emplace_back(r.begin(),r.end());
    }
    return TA::TiledRange(trs.begin(),trs.end());
}
inline TA::TiledRange one_big_tile(const std::vector<std::size_t>& ex) {
    std::vector<TA::TiledRange1> trs;
    for(auto e : ex) {
        std::vector<std::size_t> r = {0, e};
	trs.emplace_back(r.begin(),r.end());
    }
    return TA::TiledRange(trs.begin(),trs.end());
}

}
