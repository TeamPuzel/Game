// Created by Lua (TeamPuzel) on August 13th 2025.
// Copyright (c) 2025 All rights reserved.
#pragma once
#include <primitive>
#include <string_view>
#include <vector>
#include <span>
#include <ranges>

/// Encapsulates all global side effects.
///
/// Loosely inspired by Zig's idea. Has the pleasant quality in how, similarly to an IO monad, it
/// indicates function purity, since you know functions that don't take Io& can't perform any, and
/// functions which are pure and don't take Io& can't call other Io& functions.
///
/// Unlike an IO monad it's easy to integrate in C++, it's just a context.
///
/// Io itself is an abstract class and requires a concrete platform implementation to be used.
class Io {
  public:
    class Error {
      protected:
        constexpr Error() {}

      public:
        Error(Error const&) = delete;
        Error(Error&&) = delete;
        auto operator=(Error const&) -> Error& = delete;
        auto operator=(Error&&) -> Error& = delete;
        virtual ~Error() noexcept {}
    };

  protected:
    Io() noexcept {}

    virtual auto perform_read_file(char const* path) -> std::vector<u8> = 0;
    virtual void perform_write_file(char const* path, std::span<u8> data) = 0;
    virtual auto perform_open_library(char const* path) -> void* = 0;
    virtual void perform_close_library(void* library) = 0;
    virtual auto perform_load_symbol(void* library, char const* name) -> void* = 0;

  public:
    Io(Io const&) = delete;
    Io(Io&&) = delete;
    auto operator=(Io const&) -> Io& = delete;
    auto operator=(Io&&) -> Io& = delete;

    virtual ~Io() noexcept {}

    class DynamicLibrary final {
        Io& io;
        void* obj;

        DynamicLibrary(Io& io, void* obj) : io(io), obj(obj) {}

        friend class Io;

      public:
        DynamicLibrary(DynamicLibrary const&) = delete;
        auto operator=(DynamicLibrary const&) -> DynamicLibrary& = delete;

        DynamicLibrary(DynamicLibrary&& other) noexcept : io(other.io), obj(other.obj) {
            other.obj = nullptr;
        }

        auto operator=(DynamicLibrary&& other) noexcept -> DynamicLibrary& {
            if (this != &other) {
                if (obj) io.perform_close_library(obj);
                obj = other.obj;
                other.obj = nullptr;
            }
            return *this;
        }

        ~DynamicLibrary() noexcept {
            if (obj) {
                io.perform_close_library(obj);
                obj = nullptr;
            }
        }

        auto symbol(std::string_view name) const -> void* {
            return io.perform_load_symbol(obj, name.data());
        }
    };

    auto open_library(std::string_view path) [[clang::lifetimebound]] -> DynamicLibrary {
        return DynamicLibrary(*this, perform_open_library(path.data()));
    }

    auto read_file(std::string_view path) -> std::vector<u8> {
        return perform_read_file(path.data());
    }

    auto write_file(std::string_view path, std::span<u8> data) {
        return perform_write_file(path.data(), data);
    }

  private:
    static thread_local std::vector<Io*> threadlocal_io_stack;

  public:
    static auto unsafe_push_threadlocal_io(Io* io) {
        threadlocal_io_stack.push_back(io);
    }

    static auto unsafe_pop_threadlocal_io() {
        threadlocal_io_stack.pop_back();
    }

    static auto unsafe_get_threadlocal_io() -> Io& {
        if (threadlocal_io_stack.empty()) [[unlikely]] {
            throw std::runtime_error("no thread local io present");
        }
        return *threadlocal_io_stack.back();
    }
};

namespace io {
    namespace endian {
        enum class Endian {
            Big,
            Little
        };

        template <std::integral T> constexpr auto from_le_bytes(std::array<u8, sizeof(T)> bytes) noexcept -> T {
            T value = 0;
            for (usize i = 0; i < sizeof(T); i += 1) value |= T(bytes[i]) << (i * 8);
            return value;
        }

        template <std::integral T> constexpr auto from_be_bytes(std::array<u8, sizeof(T)> bytes) noexcept -> T {
            T value = 0;
            for (usize i = 0; i < sizeof(T); i += 1) value |= T(bytes[i]) << ((sizeof(T) - 1 - i) * 8);
            return value;
        }

        template <std::integral T, typename... Bytes> constexpr auto from_le_bytes(Bytes... bytes) noexcept -> T {
            return from_le_bytes({ bytes... });
        }

        template <std::integral T, typename... Bytes> constexpr auto from_be_bytes(Bytes... bytes) noexcept -> T {
            return from_be_bytes({ bytes... });
        }

        template <std::integral T> constexpr auto to_le_bytes(T value) noexcept -> std::array<u8, sizeof(T)> {
            std::array<u8, sizeof(T)> bytes;
            for (usize i = 0; i < sizeof(T); i += 1) bytes[i] = u8((value >> (i * 8)) & 0xFF);
            return bytes;
        }

        template <std::integral T> constexpr auto to_be_bytes(T value) noexcept -> std::array<u8, sizeof(T)> {
            std::array<u8, sizeof(T)> bytes;
            for (usize i = 0; i < sizeof(T); i += 1) bytes[i] = u8((value >> ((sizeof(T) - 1 - i) * 8)) & 0xFF);
            return bytes;
        }
    }

    /// A sound implementation of a binary data reader for serialization purposes.
    ///
    /// It reads data out by individual bytes avoiding alignment issues and arranges
    /// these bytes according to the desired endianness.
    template <std::ranges::input_range I, const endian::Endian INPUT_ENDIAN = endian::Endian::Big>
    requires std::same_as<std::ranges::range_value_t<I>, u8>
    class BinaryReader final {
        using InputIterator = decltype(std::ranges::begin(std::declval<I>()));

        InputIterator input;
        InputIterator end;

      public:
        static constexpr bool NOEXCEPT_READABLE = noexcept(*std::declval<InputIterator>()++);

        constexpr BinaryReader(I input [[clang::lifetimebound]])
            : input(std::ranges::begin(input)), end(std::ranges::end(input)) {}

        // template <typename R> auto read() noexcept(noexcept(R::read(*this))) -> R {
        //     return R::read(*this);
        // }

        template <std::integral T> constexpr auto read() noexcept(NOEXCEPT_READABLE) -> T {
            std::array<::u8, sizeof(T)> bytes;
            for (usize i = 0; i < sizeof(T); i += 1) bytes[i] = *input++;

            return INPUT_ENDIAN == endian::Endian::Big
                ? endian::from_be_bytes<T>(bytes)
                : endian::from_le_bytes<T>(bytes);
        }

        constexpr auto u8() noexcept(NOEXCEPT_READABLE) -> ::u8 { return read<::u8>(); }
        constexpr auto u16() noexcept(NOEXCEPT_READABLE) -> ::u16 { return read<::u16>(); }
        constexpr auto u32() noexcept(NOEXCEPT_READABLE) -> ::u32 { return read<::u32>(); }
        constexpr auto u64() noexcept(NOEXCEPT_READABLE) -> ::u64 { return read<::u64>(); }

        constexpr auto i8() noexcept(NOEXCEPT_READABLE) -> ::i8 { return read<::i8>(); }
        constexpr auto i16() noexcept(NOEXCEPT_READABLE) -> ::i16 { return read<::i16>(); }
        constexpr auto i32() noexcept(NOEXCEPT_READABLE) -> ::i32 { return read<::i32>(); }
        constexpr auto i64() noexcept(NOEXCEPT_READABLE) -> ::i64 { return read<::i64>(); }

        constexpr auto f32() noexcept(NOEXCEPT_READABLE) -> ::f32 { return std::bit_cast<::f32>(read<::u32>()); }
        constexpr auto f64() noexcept(NOEXCEPT_READABLE) -> ::f64 { return std::bit_cast<::f64>(read<::u64>()); }

        template <std::floating_point T> constexpr auto read() noexcept(NOEXCEPT_READABLE) -> T {
            if constexpr (std::same_as<T, ::f32>) {
                return f32();
            } else {
                return f64();
            }
        }

        constexpr auto boolean() noexcept(NOEXCEPT_READABLE) -> bool {
            return *input++ ? true : false;
        }

        /// Assume the current position to be a C string in a fixed buffer.
        /// Returns the C string and skips over the buffer.
        // auto cstr(usize bufsize) -> char const* {
        //     // This cast is sound: u8 -> char (on arm architectures char is even already unsigned by default)
        //     // The alignment is a match.
        //     auto ret = reinterpret_cast<char const*>(&this->data.at(this->cursor));
        //     this->cursor += bufsize;
        //     return ret;
        // }

        /// Skips over the given number of bytes.
        constexpr void skip(usize count) noexcept(NOEXCEPT_READABLE) {
            for (usize i = 0; i < count; i += 1) ++input;
        }
    };

    /// A sound implementation of a binary data writer for serialization purposes.
    ///
    /// It writes data out by individual bytes avoiding alignment issues and arranges
    /// these bytes according to the desired endianness.
    template <std::output_iterator<u8> O, const endian::Endian OUTPUT_ENDIAN = endian::Endian::Big>
    class BinaryWriter final {
        O output;
        O end;

      public:
        static constexpr bool NOEXCEPT_WRITEABLE = noexcept(*std::declval<O>()++);

        constexpr BinaryWriter(O output [[clang::lifetimebound]]) noexcept : output(output), end(output) {}

        template <std::integral T> constexpr void write(T value) noexcept(NOEXCEPT_WRITEABLE) {
            auto bytes = OUTPUT_ENDIAN == endian::Endian::Big
                ? endian::to_be_bytes(value)
                : endian::to_le_bytes(value);

            for (const ::u8 byte : bytes) {
                *output++ = byte;
            }
        }

        constexpr void u8(::u8 value) noexcept(NOEXCEPT_WRITEABLE) { write(value); }
        constexpr void u16(::u16 value) noexcept(NOEXCEPT_WRITEABLE) { write(value); }
        constexpr void u32(::u32 value) noexcept(NOEXCEPT_WRITEABLE) { write(value); }
        constexpr void u64(::u64 value) noexcept(NOEXCEPT_WRITEABLE) { write(value); }

        constexpr void i8(::i8 value) noexcept(NOEXCEPT_WRITEABLE) { write(value); }
        constexpr void i16(::i16 value) noexcept(NOEXCEPT_WRITEABLE) { write(value); }
        constexpr void i32(::i32 value) noexcept(NOEXCEPT_WRITEABLE) { write(value); }
        constexpr void i64(::i64 value) noexcept(NOEXCEPT_WRITEABLE) { write(value); }

        constexpr void f32(::f32 value) noexcept(NOEXCEPT_WRITEABLE) { return write(std::bit_cast<::u32>(value)); }
        constexpr void f64(::f64 value) noexcept(NOEXCEPT_WRITEABLE) { return write(std::bit_cast<::u64>(value)); }

        void boolean(bool value) noexcept(NOEXCEPT_WRITEABLE) {
            *output++ = value ? 1 : 0;
        }
    };
}
