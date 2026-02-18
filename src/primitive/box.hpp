// Created by Lua (TeamPuzel) on August 3rd 2025.
// Copyright (c) 2025 All rights reserved.
//
// An owned allocation.
#pragma once
#include <utility>

template <typename T> class Box final {
    T* inner;

    explicit Box(T* inner) noexcept : inner(inner) {}

  public:
    Box() noexcept : inner(nullptr) {}

    template <typename U> friend class Box;

    template <typename... Args> static auto make(Args&&...args) -> Box {
        return Box(new T(std::forward<Args>(args)...));
    }

    static auto dangling() noexcept -> Box {
        return Box(0x1);
    }

    Box(Box const&) = delete;
    auto operator=(Box const&) -> Box& = delete;

    template <typename U, std::enable_if_t<std::is_convertible_v<U*, T*>, int> = 0>
    Box(Box<U>&& existing) noexcept : inner(existing.inner) {
        existing.inner = nullptr;
    }

    template <typename U, std::enable_if_t<std::is_convertible_v<U*, T*>, int> = 0>
    auto operator=(Box<U>&& existing) noexcept -> Box& {
        if ((void*)this != (void*)&existing) {
            delete this->inner;
            this->inner = existing.inner;
            existing.inner = nullptr;
        }
        return *this;
    }

    ~Box() noexcept {
        delete this->inner; // Deleting a nullptr is sound.
    }

    // // Handle variance.
    // template <typename U, std::enable_if_t<std::is_convertible_v<T*, U*>, int> = 0> operator box<U>() && noexcept {
    //     auto base_ptr = this->inner;
    //     this->inner = nullptr;
    //     return box(base_ptr);
    // }

    auto operator->() const noexcept [[clang::lifetimebound]] -> T* {
        return this->inner;
    }

    auto operator*() const noexcept [[clang::lifetimebound]] -> T& {
        return *this->inner;
    }

    /// Performs box identity.
    template <typename U> friend auto operator==(Box<T> const& lhs, Box<U> const& rhs) noexcept -> bool {
        return (void*) lhs.raw() == (void*) rhs.raw();
    }

    template <typename U> friend auto operator!=(Box<T> const& lhs, Box<U> const& rhs) noexcept -> bool {
        return (void*) lhs.raw() != (void*) rhs.raw();
    }

    auto raw() const noexcept [[clang::lifetimebound]] -> T* {
        return this->inner;
    }

    constexpr operator bool() const {
        return inner != nullptr;
    }

    /// Dynamic cast mapping on boxed types instead of raw pointers.
    /// The box will be null and the original not destroyed if the dynamic cast is invalid.
    template <typename To> auto cast() -> Box<To> {
        auto ptr = dynamic_cast<To*>(inner);

        if (ptr) {
            inner = nullptr;
            return Box<To>(ptr);
        } else {
            return Box<To>();
        }
    }
};
