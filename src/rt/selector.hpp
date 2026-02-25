// Created by Lua (TeamPuzel) on May 29th 2025.
// Copyright (c) 2025 All rights reserved.
//
// This header defines efficient selectors (interned strings / symbols) as seen in Smalltalk and some other languages.
// Constructing two selectors from the same string literal/value produces two selectors of equivalent identity,
// meaning that the conceptual string equality comparison between selectors cheaply compares their pointer.
#pragma once
#include <primitive>
#include <unordered_set>
#include <string>
#include <mutex>

namespace rt {
    /// A primitive interned string where equality comparison is as cheap as pointer comparison.
    ///
    /// This type is thread safe but constructing instances may lock.
    /// It also works across shared libraries but that requires a small runtime and setup.
    ///
    /// The runtime needs to be initialized before any selectors are constructed.
    class [[clang::trivial_abi]] selector final {
        char const* value;

        static std::unordered_set<std::string> pool;
        static std::mutex sync;

        static std::unordered_set<std::string>* pool_ref;
        static std::mutex* sync_ref;

      public:
        static void unsafe_set_pool(std::unordered_set<std::string>* ref) noexcept {
            pool_ref = ref;
        }

        static void unsafe_set_sync(std::mutex* ref) noexcept {
            sync_ref = ref;
        }

        static void unsafe_set_self() noexcept {
            pool_ref = &pool;
            sync_ref = &sync;
        }

        explicit(false) selector(char const* str) {
            sync.lock();
            auto [iter, _] = pool.emplace(str);
            this->value = iter->c_str();
            sync.unlock();
        }

        explicit(false) selector(std::string_view str) {
            sync.lock();
            auto [iter, _] = pool.emplace(str);
            this->value = iter->c_str();
            sync.unlock();
        }

        explicit(false) selector(std::string const& str) {
            sync.lock();
            auto [iter, _] = pool.emplace(str);
            this->value = iter->c_str();
            sync.unlock();
        }

        constexpr selector() {}

        constexpr auto operator == (this selector self, selector other) noexcept -> bool {
            return self.value == other.value;
        }

        constexpr auto operator != (this selector self, selector other) noexcept -> bool {
            return self.value != other.value;
        }

        constexpr auto cstr() const noexcept -> char const* {
            return value;
        }

        constexpr auto str() const noexcept -> std::string {
            return value;
        }

        constexpr auto view() const noexcept -> std::string_view {
            return value;
        }

        constexpr auto empty() const noexcept -> bool {
            return value == nullptr;
        }
    };

    template <const usize N> struct selstring final {
        char value[N];

        consteval selstring(const char (&str)[N]) {
            for (size_t i = 0; i < N; ++i) value[i] = str[i];
        }
    };

    /// A template cache trick for quick reuse of selectors without manually storing them.
    /// It effectively bridges the runtime and compile time and does not require locking beyond first use.
    template <const selstring STR> auto sel() noexcept -> selector {
        static selector cache = STR.value;
        return cache;
    }
}
