#pragma once
#include <concepts>
#include <primitive>

namespace math {
    constexpr auto pi = 3.141592653589793238462643383279502;

    template <typename T> constexpr auto sq(T const& value) noexcept(noexcept(value * value)) -> T {
        return value * value;
    }

    template <std::floating_point T> class Angle final {
        T _radians { 0 };

      public:
        [[nodiscard]] [[gnu::pure]] constexpr auto radians() const -> T { return _radians; }
        [[nodiscard]] [[gnu::pure]] constexpr auto degrees() const -> T { return _radians * 180 / pi; }

        constexpr explicit Angle(T radians) noexcept : _radians(radians) {}
        constexpr Angle() noexcept = default;

        constexpr auto operator + (const Angle other) const noexcept -> Angle { return Angle(_radians + other._radians); }
        constexpr auto operator - (const Angle other) const noexcept -> Angle { return Angle(_radians - other._radians); }
        constexpr auto operator * (const Angle other) const noexcept -> Angle { return Angle(_radians * other._radians); }
        constexpr auto operator / (const Angle other) const noexcept -> Angle { return Angle(_radians / other._radians); }

        constexpr auto operator += (const Angle other) noexcept -> void { _radians = _radians + other._radians; }
        constexpr auto operator -= (const Angle other) noexcept -> void { _radians = _radians - other._radians; }
        constexpr auto operator *= (const Angle other) noexcept -> void { _radians = _radians * other._radians; }
        constexpr auto operator /= (const Angle other) noexcept -> void { _radians = _radians / other._radians; }

        constexpr auto operator < (const Angle other) const noexcept -> bool { return _radians < other._radians; }
        constexpr auto operator <= (const Angle other) const noexcept -> bool { return _radians <= other._radians; }
        constexpr auto operator > (const Angle other) const noexcept -> bool { return _radians > other._radians; }
        constexpr auto operator >= (const Angle other) const noexcept -> bool { return _radians >= other._radians; }

        constexpr auto operator == (const Angle other) const noexcept -> bool { return _radians == other._radians; }
        constexpr auto operator != (const Angle other) const noexcept -> bool { return _radians != other._radians; }

        constexpr auto operator - () const noexcept -> Angle { return Angle(-_radians); }
        constexpr auto operator + () const noexcept -> Angle { return Angle(+_radians); }

        /// TODO: Make the following functions constexpr in a way that doesn't break MSVC.
        [[nodiscard]] auto sin() const -> T { return std::sin(_radians); }
        [[nodiscard]] auto cos() const -> T { return std::cos(_radians); }
        [[nodiscard]] auto tan() const -> T { return std::tan(_radians); }

        constexpr explicit(false) operator Angle<f32>() { return Angle<f32>(this->_radians); }
    };

    template <std::floating_point T> [[nodiscard]] constexpr auto deg(T angle) -> Angle<T> { return Angle<T>(angle * T(pi) / 180.0f); }
    template <std::floating_point T> [[nodiscard]] constexpr auto rad(T angle) -> Angle<T> { return Angle<T>(angle); }
    template <std::integral T> [[nodiscard]] constexpr auto deg(T angle) -> Angle<f64> { return Angle<f64>(angle * pi / 180.0); }
    template <std::integral T> [[nodiscard]] constexpr auto rad(T angle) -> Angle<f64> { return Angle<f64>(angle); }
}
