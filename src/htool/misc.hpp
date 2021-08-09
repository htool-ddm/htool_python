#ifndef HTOOL_MISC_CPP
#define HTOOL_MISC_CPP

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

#endif