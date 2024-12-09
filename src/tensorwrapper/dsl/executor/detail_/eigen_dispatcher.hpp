/*
 * Copyright 2024 NWChemEx-Project
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

#pragma once
#include "../../../allocator/detail_/eigen_buffer_unwrapper.hpp"
#include <tensorwrapper/tensor/tensor_class.hpp>
namespace tensorwrapper::dsl::executor::detail_ {

/** @brief Calls a functor after converting to buffer::Eigen objects.
 *
 *  @tparam Functor The type of the functor to call on the converted buffers.
 *                  See below for the API the functor must adhere to.
 *
 *  This class is used to execute DSL instructions by converting buffers to
 *  buffer::Eigen objects and then forwarding the buffer::Eigen objects
 *  to the provided functor.
 *
 *  Functor API
 *  -----------
 *
 *  The functor must define a member `run` which is capable of taking 1, 2,
 *  or 3 buffer::Eigen objects. Each buffer::Eigen object is templated on the
 *  floating-point type and the rank. The full set of buffer::Eigen objects
 *  which *this considers is given by EigenBufferUnwrapper::variant_type. The
 *  functor must provide `run` methods for ALL possible tuples of 1, 2, or 3
 *  buffer::Eigen objects. This is most easily done by defining your `run`
 *  method as follows:
 *
 *  @code
 *  template<typename...Args>
 *  auto run(Args&&...args){
 *      // Check if args is a tuple you want to handle
 *      if constexpr(if_handling){
 *      } else { throw std::runtime_error("Unsupported tuple of buffers"); }
 *  }
 *  @endcode
 *
 *  The above `run` method will allow the code to compile with arbitrary buffers
 *  without requiring the functor developer to implement the functor for all
 *  tuples. At the same time, if the functor encounters a tuple the developer is
 *  not expecting an error is raised.
 *
 *  Implementation Note
 *  -------------------
 *
 *  It should be possible to condense the dispatch
 *  overloads using parameter packs; however, in my attempts to do it it got
 *  messy quickly. Given that I don't foresee needing to dispatch based on more
 *  than three buffers at a time, I decided to just manually write the three
 *  overloads.
 *
 */
template<typename Functor>
class EigenDispatcher {
private:
    /// Type that knowns how to downcast to an buffer::Eigen object.
    using unwrapper = allocator::detail_::EigenBufferUnwrapper;

    /// Is @p T the same type as @p Functor up to references and cv-qualifiers?
    template<typename T>
    static constexpr auto is_functor_v =
      std::is_same_v<std::decay_t<T>, Functor>;

    /// Enables a function if is_functor_v<T> is true
    template<typename T>
    using enable_if_functor_t = std::enable_if_t<is_functor_v<T>>;

public:
    /** @brief Creates a new EigenDispatcher which will call @p functor.
     *
     *  @tparam FunctorType The type of the input functor. Assumed to be
     *                      implicitly convertible to @p Functor.
     *  @tparam <Anonymous> A template type parameter used to disable the ctor
     *                      when FunctorType is not implicitly convertible to
     *                      @p Functor.
     *
     *  This ctor initializes *this with @p functor, an object of type
     *  @p Functor. @p functor will be called with the converted buffer::Eigen
     *  objects when `dispatch` is called.
     */
    template<typename FunctorType, typename = enable_if_functor_t<FunctorType>>
    explicit EigenDispatcher(FunctorType&& functor) :
      m_f_(std::forward<FunctorType>(functor)) {}

    /// One argument overload. See three argument overload for description.
    template<typename Type0>
    decltype(auto) dispatch(Type0&& arg0) {
        auto variant_0 = unwrapper::downcast(std::forward<Type0>(arg0));

        return std::visit([this](auto&& buffer0) { return m_f_.run(buffer0); },
                          variant_0);
    }

    /// Two argument overload. See three argument overload for description.
    template<typename Type0, typename Type1>
    decltype(auto) dispatch(Type0&& arg0, Type1&& arg1) {
        auto variant_0 = unwrapper::downcast(std::forward<Type0>(arg0));
        auto variant_1 = unwrapper::downcast(std::forward<Type1>(arg1));

        return std::visit(
          [&variant_1, this](auto&& buffer0) {
              return std::visit(
                [&buffer0, this](auto&& buffer1) {
                    return m_f_.run(buffer0, buffer1);
                },
                variant_1);
          },
          variant_0);
    }

    /** @brief Dispatches to wrapped functor based on provided arguments.
     *
     *  @tparam Type0 The qualified type of @p arg0. Assumed to be a convertible
     *                to an instantiation of buffer::Eigen.
     *  @tparam Type1 The qualified type of @p arg1. Assumed to be a convertible
     *                to an instantiation of buffer::Eigen.
     *  @tparam Type2 The qualified type of @p arg2. Assumed to be a convertible
     *                to an instantiation of buffer::Eigen.
     *
     *  Each dispatch overload works the same way, they simply differ in the
     *  number of buffer objects they convert. The dispatch methods:
     *
     *  1. Convert the provided buffer objects to std::variants capable of
     *     holding every instantiation of buffer::Eigen we support.
     *  2. Rely on nested calls to std::visit to work out all of the possible
     *     dispatch scenarios.
     *  3. Call the functor held by *this via the dispatch scenario
     *     corresponding to the states of @p arg0, @p arg1, and @p arg2.
     *
     *  @param[in] arg0 The first buffer to dispatch on.
     *  @param[in] arg1 The second buffer to dispatch on.
     *  @param[in] arg2 The third buffer to dispatch on.
     *
     *  @return Whatever the functor held by *this returns when provided the
     *          converted buffer::Eigen objects.
     *
     *  @throw ??? Throws if the functor throws. Same throw guarantee.
     */
    template<typename Type0, typename Type1, typename Type2>
    decltype(auto) dispatch(Type0&& arg0, Type1&& arg1, Type2&& arg2) {
        auto variant_0 = unwrapper::downcast(std::forward<Type0>(arg0));
        auto variant_1 = unwrapper::downcast(std::forward<Type1>(arg1));
        auto variant_2 = unwrapper::downcast(std::forward<Type2>(arg2));

        return std::visit(
          [&variant_1, &variant_2, this](auto&& buffer0) {
              return std::visit(
                [&buffer0, &variant_2, this](auto&& buffer1) {
                    return std::visit(
                      [&buffer0, &buffer1, this](auto&& buffer2) {
                          return m_f_.run(buffer0, buffer1, buffer2);
                      },
                      variant_2);
                },
                variant_1);
          },
          variant_0);
    }

private:
    /// The functor to apply to the buffers.
    Functor m_f_;
};

/// Works out template type of the class based on ctor argument's type.
template<typename FunctorType, typename = void>
EigenDispatcher(FunctorType&&) -> EigenDispatcher<std::decay_t<FunctorType>>;

} // namespace tensorwrapper::dsl::executor::detail_