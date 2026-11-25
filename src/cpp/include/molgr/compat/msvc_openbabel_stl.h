#pragma once

// Open Babel 3.1 still derives from std::binary_function. MSVC 19.50+
// removed that C++17 compatibility type, including the legacy opt-in switch.
#if defined(_MSC_VER) && _MSC_VER >= 1950
#  if defined(_HAS_AUTO_PTR_ETC)
#    undef _HAS_AUTO_PTR_ETC
#  endif
#  define _HAS_AUTO_PTR_ETC 0

#  include <functional>

namespace std
{
  /// Supply the legacy argument and result aliases used by Open Babel binary predicates on MSVC.
  /// @tparam Arg1 First predicate argument.
  /// @tparam Arg2 Second predicate argument.
  /// @tparam Result Predicate return type.
  template <typename Arg1, typename Arg2, typename Result>
  struct binary_function
  {
    using first_argument_type = Arg1;
    using second_argument_type = Arg2;
    using result_type = Result;
  };
}
#endif
