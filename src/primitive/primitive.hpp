// Created by Lua (TeamPuzel) on May 25th 2025.
// Copyright (c) 2025 All rights reserved.
//
// Primitive aliases asserted to a specific memory representation.
// C++ is still defined loosely enough that on exotic platforms these would be invalid,
// but for all modern platforms I care about this should be enough to guarantee consistency.
//
// This avoids common bugs like assuming signedness or size of primitives, which does differ between platforms.
// The header also asserts that certain poorly defined operations evaluate as expected.
#pragma once
#include <concepts>

using char16 = char16_t;
using char32 = char32_t;

using f32 = float;
using f64 = double;

using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;

using i8 = signed char;
using i16 = signed short;
using i32 = signed int;

#ifdef _MSC_VER
using u64 = unsigned long long;
using i64 = signed long long;
#else
using u64 = unsigned long;
using i64 = signed long;
#endif

using usize = u64;
using isize = i64;

static_assert(sizeof(f32) == 4);
static_assert(sizeof(f64) == 8);

static_assert(sizeof(u8) == 1);
static_assert(sizeof(u16) == 2);
static_assert(sizeof(u32) == 4);
static_assert(sizeof(u64) == 8);

static_assert(sizeof(i8) == 1);
static_assert(sizeof(i16) == 2);
static_assert(sizeof(i32) == 4);
static_assert(sizeof(i64) == 8);

// We can assume we are running a 64 bit system as everyone dropped 32 bit support by now and I don't care.
static_assert(sizeof(usize) == sizeof(void*));
static_assert(sizeof(isize) == sizeof(void*));

// This is required by C++20, however it technically isn't in prior versions.
// Since this is C++ after all, best assert that we're not using a weird compiler, just in case.
static_assert(-4 >> 1 == -2, ">> doesn't do sign extension");

// C++ does not define the representation of signed primitives either.
// Assert that this is a modern and sensible compiler/architecture.
// This is a requirement. Code can rely on this representation from this point on.
static_assert(static_cast<i8>(-1) == ~i8(0), "Compiler is not using two's complement");
static_assert(static_cast<i32>(-1) == ~i32(0), "Compiler is not using two's complement");
static_assert(static_cast<i64>(-1) == ~i64(0), "Compiler is not using two's complement");

namespace math {
    template <std::signed_integral T> constexpr auto arithmetic_mod(T lhs, T rhs) noexcept -> T {
        return (lhs % rhs + rhs) % rhs;
    }
}

template <typename Fn> class ScopeExit final {
    Fn fn;

  public:
    constexpr ScopeExit(Fn fn) noexcept : fn(std::move(fn)) {}

    constexpr ScopeExit(ScopeExit const&) noexcept = delete;
    constexpr ScopeExit(ScopeExit&&) noexcept = delete;
    constexpr ScopeExit& operator=(ScopeExit const&) noexcept = delete;
    constexpr ScopeExit& operator=(ScopeExit&&) noexcept = delete;

    constexpr ~ScopeExit() {
        fn();
    }
};
