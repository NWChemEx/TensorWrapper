#include "../buffer/make_pimpl.hpp"
#include "../test_tensor.hpp"
#include "tensorwrapper/tensor/conversion/conversion.hpp"

using namespace tensorwrapper::tensor;

using scalar_type = field::Scalar;
using tensor_type = field::Tensor;

template<typename FieldType>
using buffer_type = buffer::Buffer<FieldType>;

template<typename FieldType>
using pimpl_type = buffer::detail_::TABufferPIMPL<FieldType>;

using ta_scalar_type = TA::DistArray<TA::Tensor<double>, TA::SparsePolicy>;

using ta_tot_type =
  TA::DistArray<TA::Tensor<TA::Tensor<double>>, TA::SparsePolicy>;

using scalar_convert_type = conversion::Conversion<ta_scalar_type>;

using tot_convert_type = conversion::Conversion<ta_tot_type>;

TEST_CASE("Conversion") {
    auto&& [pvec, pmat, pt3d] = testing::make_pimpl<scalar_type>();
    buffer_type<scalar_type> vec(pvec->clone());
    buffer_type<scalar_type> mat(pmat->clone());
    buffer_type<scalar_type> t3d(pt3d->clone());

    auto&& [pvov, pvom, pmov] = testing::make_pimpl<tensor_type>();
    buffer_type<tensor_type> vov(pvov->clone());
    buffer_type<tensor_type> vom(pvom->clone());
    buffer_type<tensor_type> mov(pmov->clone());

    scalar_convert_type convert_scalar;
    tot_convert_type convert_tot;

    {
        auto test = convert_scalar.convert(vec);
        std::cout << test << std::endl;
    }
    {
        auto test = convert_scalar.convert(mat);
        std::cout << test << std::endl;
    }
    {
        auto test = convert_scalar.convert(t3d);
        std::cout << test << std::endl;
    }
    {
        auto test = convert_tot.convert(vov);
        std::cout << test << std::endl;
    }
    {
        auto test = convert_tot.convert(vom);
        std::cout << test << std::endl;
    }
    {
        auto test = convert_tot.convert(mov);
        std::cout << test << std::endl;
    }
}