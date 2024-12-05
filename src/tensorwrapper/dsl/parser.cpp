#include "executor/executor.hpp"
#include <tensorwrapper/dsl/parser.hpp>

namespace tensorwrapper::dsl {

// Specialize this class to set the class based on the objects being combined
template<typename ObjectType>
struct ExecutorType;

// Sets the default executor for tensors
template<>
struct ExecutorType<Tensor> {
    using executor_type = executor::Eigen;
};

// Typedef to shorten retrieving the type of default executor for @p ObjectType
template<typename ObjectType>
using default_executor_type = typename ExecutorType<ObjectType>::executor_type;

#define TPARAMS template<typename ObjectType, typename LabelType>
#define PARSER Parser<ObjectType, LabelType>
#define LABELED_TYPE typename PARSER::labeled_type

TPARAMS LABELED_TYPE PARSER::assign(labeled_type lhs, labeled_type rhs) {
    return default_executor_type<ObjectType>::assign(std::move(lhs),
                                                     std::move(rhs));
}

TPARAMS LABELED_TYPE PARSER::add(labeled_type result, labeled_type lhs,
                                 labeled_type rhs) {
    return default_executor_type<ObjectType>::add(
      std::move(result), std::move(lhs), std::move(rhs));
}

#undef PARSER
#undef TPARAMS

template class Parser<Tensor, std::string>;

} // namespace tensorwrapper::dsl