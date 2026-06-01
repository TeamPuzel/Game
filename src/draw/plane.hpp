// Created by Lua (TeamPuzel) on May 29th 2025.
// Copyright (c) 2025 All rights reserved.
//
// Yet another port of my Swift/Rust/Kotlin graphics library. The approach is viable in C++ but extremely verbose.
// It's kind of like std::ranges but for graphics I guess, though less esoteric to extend.
#pragma once
#include <concepts>
#include <primitive>
#include <utility>
#include <algorithm>
#include <vector>
#include <thread>
#include "color.hpp"

namespace draw {
    template <typename Self> concept Plane = requires(Self const& self, i32 x, i32 y) {
        { self.get(x, y) } -> std::same_as<Color>;
    };

    template <typename Self> concept MutablePlane = Plane<Self> and requires(Self& self, i32 x, i32 y, Color color) {
        { self.set(x, y, color) } -> std::same_as<void>;
    };

    template <typename Self> concept SizedPlane = Plane<Self> and requires(Self const& self) {
        { self.width() } -> std::same_as<i32>;
        { self.height() } -> std::same_as<i32>;
    };

    template <typename Self, typename From> concept PrimitivePlane = SizedPlane<From> and requires(From const& other) {
        { Self::flatten(other) } -> std::same_as<Self>;
    };

    template <typename Self> concept SizedMutablePlane = SizedPlane<Self> and MutablePlane<Self>;

    enum class Origin {
        Center,
        Left,
        Right,
        Top,
        Bottom,
        TopLeft,
        TopRight,
        BottomLeft,
        BottomRight,
    };

    struct Offset final { i32 x, y; };

    auto offset_for(Origin origin, SizedPlane auto const& plane) -> Offset {
        switch (origin) {
            case Origin::Center:      return { .x = plane.width() / 2, .y = plane.height() / 2 };
            case Origin::Left:        return { .x = 0,                 .y = plane.height() / 2 };
            case Origin::Right:       return { .x = plane.width() - 1, .y = plane.height() / 2 };
            case Origin::Top:         return { .x = plane.width() / 2, .y = 0                  };
            case Origin::Bottom:      return { .x = plane.width() / 2, .y = plane.height() - 1 };
            case Origin::TopLeft:     return { .x = 0,                 .y = 0                  };
            case Origin::TopRight:    return { .x = plane.width() - 1, .y = 0                  };
            case Origin::BottomLeft:  return { .x = 0,                 .y = plane.height() - 1 };
            case Origin::BottomRight: return { .x = plane.width() - 1, .y = plane.height() - 1 };
        }
    }
}

/// Performs forwarding adapter composition. Based on the design of std::ranges.
template <draw::Plane Self, typename Adapt> [[clang::always_inline]]
constexpr auto operator|(Self&& self, Adapt&& adapt) noexcept(noexcept(std::forward<Adapt>(adapt)(std::forward<Self>(self))))
    -> decltype(std::forward<Adapt>(adapt)(std::forward<Self>(self)))
{
    return std::forward<Adapt>(adapt)(std::forward<Self>(self));
}

/// When sized, there is a universal, least optimized fallback equality definition available between all sized drawables.
///
/// Comparing infinite drawables is often possible through various tricks based on the math behind
/// its infinite size, except for some truly infinite planes like an InfiniteImage (type not implemented in the C++ version).
template <draw::SizedPlane L, draw::SizedPlane R> constexpr auto operator==(L const& lhs, R const& rhs) -> bool {
    if (lhs.width() != rhs.width() or lhs.height() != rhs.height()) return false;
    for (i32 x = 0; x < lhs.width(); x += 1) {
        for (i32 y = 0; y < lhs.height(); y += 1) {
            if (lhs.get(x, y) != rhs.get(x, y)) return false;
        }
    }
    return true;
}

/// Older C++ versions do not derive this from equality itself yet.
template <draw::SizedPlane L, draw::SizedPlane R> constexpr auto operator!=(L const& lhs, R const& rhs) -> bool {
    return !(lhs == rhs);
}

// Flattening ----------------------------------------------------------------------------------------------------------
namespace draw {
    namespace adapt {
        template <Plane T> struct Flatten final {
            template <SizedPlane U> constexpr T operator()(U const& self) const noexcept(noexcept(T::flatten(self)))
            requires
                PrimitivePlane<T, U>
            {
                return T::flatten(self);
            }
        };
    }

    template <typename T> constexpr adapt::Flatten<T> flatten() noexcept {
        return adapt::Flatten<T> {};
    }
}

// Trait composition ---------------------------------------------------------------------------------------------------
// These adapters simply map to the usual pixel interface but can be composed in chained expressions.
// You could of course chain with the dot operator but that requires parentheses which negates the
// benefit of having nice, readable composition.

namespace draw {
    namespace adapt {
        struct Width final {
            template <SizedPlane T> constexpr auto operator()(T const& self) const noexcept(noexcept(self.width())) -> i32 {
                return self.width();
            }
        };

        struct Height final {
            template <SizedPlane T> constexpr auto operator()(T const& self) const noexcept(noexcept(self.height())) -> i32 {
                return self.height();
            }
        };

        struct Get final {
            i32 x, y;

            template <Plane T> constexpr auto operator()(T const& self) const noexcept(noexcept(self.get(x, y))) -> Color {
                return self.get(x, y);
            }
        };

        struct Set final {
            i32 x, y;
            Color color;

            template <MutablePlane T> constexpr auto operator()(T& self) const noexcept(self.set(x, y, color)) -> T& {
                return self.set(x, y, color);
            }
        };

        template <typename T> struct Eq final {
            T const& rhs;

            template <typename U> constexpr auto operator()(U const& lhs) const -> bool {
                return lhs == rhs;
            }
        };

        template <typename T> struct NotEq final {
            T const& rhs;

            template <typename U> constexpr auto operator()(U const& lhs) const -> bool {
                return lhs != rhs;
            }
        };
    }

    constexpr adapt::Width width() noexcept {
        return adapt::Width {};
    }

    constexpr adapt::Height height() noexcept {
        return adapt::Height {};
    }

    constexpr adapt::Get get(i32 x, i32 y) noexcept {
        return adapt::Get { x, y };
    }

    constexpr adapt::Set get(i32 x, i32 y, Color color) noexcept {
        return adapt::Set { x, y, color };
    }

    template <typename T> constexpr adapt::Eq<T> eq(T const& rhs) noexcept {
        return adapt::Eq<T> { rhs };
    }

    template <typename T> constexpr adapt::NotEq<T> ne(T const& rhs) noexcept {
        return adapt::NotEq<T> { rhs };
    }
}

// Mutation ------------------------------------------------------------------------------------------------------------
namespace draw {
    namespace adapt {
        struct Clear final {
            Color color;

            template <typename T> constexpr T& operator()(T& self) const requires SizedPlane<T> and MutablePlane<T> {
                for (i32 x = 0; x < self.width(); x += 1) {
                    for (i32 y = 0; y < self.height(); y += 1) {
                        self.set(x, y, color);
                    }
                }
                return self;
            }
        };

        template <typename Blend> struct Pixel final {
            i32 x, y;
            Color color;
            Blend blend_mode;

            constexpr Pixel(i32 x, i32 y, Color color, Blend blend_mode)
                : x(x), y(y), color(color), blend_mode(blend_mode) {}

            template <typename T> constexpr T& operator()(T& self) const requires SizedPlane<T> and MutablePlane<T> {
                self.set(x, y, color.blend_over(self.get(x, y), blend_mode));
                return self;
            }
        };

        struct Line final {
            i32 sx, sy, dx, dy;
            Color color;

            template <MutablePlane T> constexpr T& operator()(T& self) const {
                i32 x0 = sx;
                i32 y0 = sy;
                i32 x1 = dx;
                i32 y1 = dy;

                i32 delta_x = std::abs(x1 - x0);
                i32 delta_y = std::abs(y1 - y0);
                i32 step_x = (x0 < x1) ? 1 : -1;
                i32 step_y = (y0 < y1) ? 1 : -1;
                i32 err = delta_x - delta_y;

                while (true) {
                    self.set(x0, y0, color);

                    if (x0 == x1 && y0 == y1) break;
                    i32 e2 = 2 * err;
                    if (e2 > -delta_y) {
                        err -= delta_y;
                        x0 += step_x;
                    }
                    if (e2 < delta_x) {
                        err += delta_x;
                        y0 += step_y;
                    }
                }

                return self;
            }
        };

        template <SizedPlane D, typename Blend> struct Draw final {
            D const& drawable;
            i32 x, y;
            Blend blend_mode;

            constexpr Draw(D const& drawable, i32 x, i32 y, Blend blend_mode)
                : drawable(drawable), x(x), y(y), blend_mode(blend_mode) {}

            template <typename T> constexpr T& operator()(T& self) const requires SizedPlane<T> and MutablePlane<T> {
                const auto width = drawable.width();
                const auto height = drawable.height();

                for (i32 x = 0; x < width; x += 1) {
                    for (i32 y = 0; y < height; y += 1) {
                        self.set(
                            x + this->x, y + this->y,
                            drawable.get(x, y).blend_over(self.get(x + this->x, y + this->y), blend_mode)
                        );
                    }
                }

                return self;
            }
        };

        template <SizedPlane D, typename Blend> struct OriginDraw final {
            D const& drawable;
            Origin from, at;
            Blend blend_mode;

            constexpr OriginDraw(D const& drawable, Origin from, Origin at, Blend blend_mode)
                : drawable(drawable), from(from), at(at), blend_mode(blend_mode) {}

            template <typename T> constexpr T& operator()(T& self) const requires SizedPlane<T> and MutablePlane<T> {
                auto [x, y] = offset_for(at, self);
                auto offset = offset_for(from, drawable);
                return Draw(drawable, x - offset.x, y - offset.y, blend_mode)(self);
            }
        };

        template <SizedPlane D, typename Blend> struct ThreadedDraw final {
            D const& drawable;
            i32 x, y;
            Blend blend_mode;

            constexpr ThreadedDraw(D const& drawable, i32 x, i32 y, Blend blend_mode)
                : drawable(drawable), x(x), y(y), blend_mode(blend_mode) {}

            template <typename T> constexpr T& operator()(T& self) const requires SizedPlane<T> and MutablePlane<T> {
                const auto width = drawable.width();
                const auto height = drawable.height();

                const u32 thread_count = std::thread::hardware_concurrency();
                const i32 rows_per_thread = (height + thread_count - 1) / thread_count;

                std::vector<std::jthread> threads;
                threads.reserve(thread_count);

                for (u32 t = 0; t < thread_count; t += 1) {
                    const i32 y_start = t * rows_per_thread;
                    const i32 y_end = std::min(height, y_start + rows_per_thread);

                    threads.emplace_back([&, y_start, y_end] {
                        for (i32 y = y_start; y < y_end; y += 1) {
                            for (i32 x = 0; x < width; x += 1) {
                                self.set(
                                    x + this->x, y + this->y,
                                    drawable.get(x, y).blend_over(self.get(x + this->x, y + this->y), blend_mode)
                                );
                            }
                        }
                    });
                }

                return self;
            }
        };
    }

    constexpr adapt::Clear clear(Color color = color::CLEAR) {
        return adapt::Clear { color };
    }

    template <typename Blend> constexpr adapt::Pixel<Blend> pixel(i32 x, i32 y, Color color, Blend blend_mode) {
        return adapt::Pixel { x, y, color, blend_mode };
    }

    constexpr adapt::Pixel<decltype(&blend::binary)> pixel(i32 x, i32 y, Color color) {
        return adapt::Pixel { x, y, color, blend::binary };
    }

    template <SizedPlane D, typename Blend> constexpr adapt::Draw<D, Blend> draw(D const& drawable, i32 x, i32 y, Blend blend_mode) {
        return adapt::Draw { drawable, x, y, blend_mode };
    }

    template <typename D> constexpr adapt::Draw<D, decltype(&blend::binary)> draw(D const& drawable, i32 x, i32 y) {
        return adapt::Draw { drawable, x, y, blend::binary };
    }

    template <typename D> constexpr adapt::Draw<D, decltype(&blend::binary)> draw(D const& drawable) {
        return adapt::Draw { drawable, 0, 0, blend::binary };
    }

    template <typename D, typename Blend> constexpr adapt::OriginDraw<D, Blend> draw(D const& drawable, Origin from, Origin at, Blend blend_mode) {
        return adapt::OriginDraw { drawable, from, at, blend_mode };
    }

    template <typename D> constexpr adapt::OriginDraw<D, decltype(&blend::binary)> draw(D const& drawable, Origin from, Origin at) {
        return adapt::OriginDraw { drawable, from, at, blend::binary };
    }

    template <typename D, typename Blend> constexpr adapt::OriginDraw<D, Blend> draw(D const& drawable, Origin aligning, Blend blend_mode) {
        return adapt::OriginDraw { drawable, aligning, aligning, blend_mode };
    }

    template <typename D> constexpr adapt::OriginDraw<D, decltype(&blend::binary)> draw(D const& drawable, Origin aligning) {
        return adapt::OriginDraw { drawable, aligning, aligning, blend::binary };
    }

    template <SizedPlane D, typename Blend> constexpr adapt::ThreadedDraw<D, Blend> draw_threaded(D const& drawable, i32 x, i32 y, Blend blend_mode) {
        return adapt::ThreadedDraw { drawable, x, y, blend_mode };
    }

    template <typename D> constexpr adapt::ThreadedDraw<D, decltype(&blend::binary)> draw_threaded(D const& drawable, i32 x, i32 y) {
        return adapt::ThreadedDraw { drawable, x, y, blend::binary };
    }

    template <typename D> constexpr adapt::ThreadedDraw<D, decltype(&blend::binary)> draw_threaded(D const& drawable) {
        return adapt::ThreadedDraw { drawable, 0, 0, blend::binary };
    }

    constexpr adapt::Line line(i32 sx, i32 sy, i32 dx, i32 dy, Color color = color::WHITE) {
        return adapt::Line { sx, sy, dx, dy, color };
    }
}

// Transformation ------------------------------------------------------------------------------------------------------
namespace draw {
    /// A construct similar to the standard reference wrapper but exposing the plane interface.
    template <Plane T> struct Ref final {
        T& inner;

        constexpr explicit(false) Ref(T& inner) : inner(inner) {}

        constexpr auto width() const noexcept(noexcept(inner.width())) -> i32 requires SizedPlane<T> {
            return inner.width();
        }

        constexpr auto height() const noexcept(noexcept(inner.height())) -> i32 requires SizedPlane<T> {
            return inner.height();
        }

        constexpr auto get(i32 x, i32 y) const noexcept(noexcept(inner.get(x, y))) -> Color {
            return inner.get(x, y);
        }

        constexpr void set(i32 x, i32 y, Color color) noexcept(noexcept(inner.set(x, y, color))) requires MutablePlane<T> {
            inner.set(x, y, color);
        }
    };

    template <Plane T> class DiscardSlice final {
        T inner;
        i32 x, y, w, h;

      public:
        constexpr explicit DiscardSlice(T inner, i32 x, i32 y, i32 width, i32 height) noexcept
            : inner(inner), x(x), y(y), w(width), h(height) {}

        constexpr auto get(i32 x, i32 y) const noexcept(noexcept(inner.get(this->x + x, this->y + y))) -> Color {
            if (x >= 0 and x < w and y >= 0 and y < h) {
                return inner.get(this->x + x, this->y + y);
            } else {
                return color::CLEAR;
            }
        }

        constexpr void set(i32 x, i32 y, Color color) noexcept(noexcept(inner.set(this->x + x, this->y + y, color))) {
            if (x >= 0 and x < w and y >= 0 and y < h) {
                inner.set(this->x + x, this->y + y, color);
            }
        }

        constexpr auto width() const noexcept -> i32 {
            return w;
        }

        constexpr auto height() const noexcept -> i32 {
            return h;
        }

        constexpr auto resize_left(i32 offset) const noexcept -> DiscardSlice {
            return DiscardSlice { inner, x - offset, y, std::max(0, w + offset), h };
        }

        constexpr auto resize_right(i32 offset) const noexcept -> DiscardSlice {
            return DiscardSlice { inner, x, y, std::max(0, w + offset), h };
        }

        constexpr auto resize_top(i32 offset) const noexcept -> DiscardSlice {
            return DiscardSlice { inner, x, y - offset, w, std::max(0, h + offset) };
        }

        constexpr auto resize_bottom(i32 offset) const noexcept -> DiscardSlice {
            return DiscardSlice { inner, x, y, w, std::max(0, h + offset) };
        }

        constexpr auto resize_horizontal(i32 offset) const noexcept -> DiscardSlice {
            return DiscardSlice { inner, x - offset, y, std::max(0, w + offset * 2), h };
        }

        constexpr auto resize_vertical(i32 offset) const noexcept -> DiscardSlice {
            return DiscardSlice { inner, x, y - offset, w, std::max(0, h + offset * 2) };
        }

        constexpr auto shift(i32 off_x, i32 off_y) const noexcept -> DiscardSlice {
            return DiscardSlice { inner, x + off_x, y + off_y, w, h };
        }
    };

    template <Plane T> class Slice final {
        T inner;
        i32 x, y, w, h;

      public:
        constexpr explicit Slice(T inner, i32 x, i32 y, i32 width, i32 height) noexcept
            : inner(inner), x(x), y(y), w(width), h(height) {}

        constexpr auto get(i32 x, i32 y) const noexcept(noexcept(inner.get(this->x + x, this->y + y))) -> Color {
            return inner.get(this->x + x, this->y + y);
        }

        constexpr void set(i32 x, i32 y, Color color) noexcept(noexcept(inner.set(this->x + x, this->y + y, color))) {
            inner.set(this->x + x, this->y + y, color);
        }

        constexpr auto width() const noexcept -> i32 {
            return w;
        }

        constexpr auto height() const noexcept -> i32 {
            return h;
        }

        constexpr auto resize_left(i32 offset) const noexcept -> Slice {
            return Slice { inner, x - offset, y, std::max(0, w + offset), h };
        }

        constexpr auto resize_right(i32 offset) const noexcept -> Slice {
            return Slice { inner, x, y, std::max(0, w + offset), h };
        }

        constexpr auto resize_top(i32 offset) const noexcept -> Slice {
            return Slice { inner, x, y - offset, w, std::max(0, h + offset) };
        }

        constexpr auto resize_bottom(i32 offset) const noexcept -> Slice {
            return Slice { inner, x, y, w, std::max(0, h + offset) };
        }

        constexpr auto resize_horizontal(i32 offset) const noexcept -> Slice {
            return Slice { inner, x - offset, y, std::max(0, w + offset * 2), h };
        }

        constexpr auto resize_vertical(i32 offset) const noexcept -> Slice {
            return Slice { inner, x, y - offset, w, std::max(0, h + offset * 2) };
        }

        constexpr auto resize(i32 offset) const noexcept -> Slice {
            return resize_horizontal(offset).resize_vertical(offset);
        }

        constexpr auto shift(i32 off_x, i32 off_y) const noexcept -> Slice {
            return Slice { inner, x + off_x, y + off_y, w, h };
        }

        /// Convert to a slice which discards out of bounds access instead of forwarding it.
        constexpr auto discard() const noexcept -> DiscardSlice<T> {
            return DiscardSlice { inner, x, y, w, h };
        }
    };

    template <Plane T> struct Grid final {
        T inner;
        i32 tile_width, tile_height;

      public:
        constexpr explicit Grid(T inner, i32 tile_width, i32 tile_height) noexcept
            : inner(std::move(inner)), tile_width(tile_width), tile_height(tile_height) {}

        constexpr auto tile(i32 x, i32 y) const noexcept -> Slice<T> {
            return Slice(inner, x * tile_width, y * tile_height, tile_width, tile_height);
        }

        constexpr auto tile_ref(i32 x, i32 y) const noexcept -> Slice<Ref<const T>> {
            return Slice(Ref(inner), x * tile_width, y * tile_height, tile_width, tile_height);
        }

        constexpr auto tile_ref(i32 x, i32 y) noexcept -> Slice<Ref<T>> {
            return Slice(Ref(inner), x * tile_width, y * tile_height, tile_width, tile_height);
        }

        constexpr auto ref() noexcept -> Grid<Ref<T>> {
            return Grid<Ref<T>>(Ref(inner), tile_width, tile_height);
        }

        constexpr auto ref() const noexcept -> Grid<Ref<const T>> {
            return Grid<Ref<const T>>(Ref(inner), tile_width, tile_height);
        }
    };

    namespace adapt {
        struct Slice final {
            i32 x, y, width, height;

            template <Plane T> constexpr auto operator()(T inner) const noexcept -> draw::Slice<T> {
                return draw::Slice<T>(inner, x, y, width, height);
            }
        };

        struct Grid final {
            i32 tile_width, tile_height;

            template <Plane T> constexpr auto operator()(T inner) const noexcept -> draw::Grid<T> {
                return draw::Grid<T>(std::move(inner), tile_width, tile_height);
            }
        };

        struct Shift final {
            i32 x, y;

            template <SizedPlane T> constexpr auto operator()(T inner) const noexcept -> draw::Slice<T> {
                return draw::Slice<T>(inner, x, y, inner.width(), inner.height());
            }
        };

        struct AsSlice final {
            template <SizedPlane T> constexpr auto operator()(T inner) const noexcept -> draw::Slice<T> {
                return draw::Slice<T>(inner, 0, 0, inner.width(), inner.height());
            }
        };

        struct AsRef final {
            template <Plane T> constexpr auto operator()(T& inner) const noexcept -> Ref<T> {
                return inner;
            }

            template <Plane T> constexpr auto operator()(T const& inner) const noexcept -> Ref<const T> {
                return inner;
            }
        };

        struct Move final {
            template <Plane T> constexpr auto operator()(T&& inner) const noexcept -> T&& {
                return std::move(inner);
            }
        };

        struct AsConst final {
            template <Plane T> constexpr auto operator()(T& inner) const noexcept -> T const& {
                return inner;
            }
        };

        struct Discard final {
            template <SizedPlane T> constexpr auto operator()(T&& inner) const noexcept {
                i32 w = inner.width(), h = inner.height();
                return DiscardSlice(std::forward<T>(inner), 0, 0, w, h);
            }
        };

        struct Tile final {
            i32 x, y;

            template <Plane T> constexpr auto operator()(draw::Grid<T> const& self) const noexcept -> draw::Slice<T> {
                return self.tile(x, y);
            }
        };

        struct TileRef final {
            i32 x, y;

            template <Plane T> constexpr auto operator()(draw::Grid<T> const& self) const noexcept -> draw::Slice<Ref<T>> {
                return self.tile_ref(x, y);
            }
        };

        struct ResizeLeft final {
            i32 offset;

            template <SizedPlane T> constexpr auto operator()(T inner) const noexcept -> draw::Slice<T> {
                return draw::Slice<T>(inner, 0, 0, inner.width(), inner.height()).resize_left(offset);
            }
        };

        struct ResizeRight final {
            i32 offset;

            template <SizedPlane T> constexpr auto operator()(T inner) const noexcept -> draw::Slice<T> {
                return draw::Slice<T>(inner, 0, 0, inner.width(), inner.height()).resize_right(offset);
            }
        };

        struct ResizeTop final {
            i32 offset;

            template <SizedPlane T> constexpr auto operator()(T inner) const noexcept -> draw::Slice<T> {
                return draw::Slice<T>(inner, 0, 0, inner.width(), inner.height()).resize_top(offset);
            }
        };

        struct ResizeBottom final {
            i32 offset;

            template <SizedPlane T> constexpr auto operator()(T inner) const noexcept -> draw::Slice<T> {
                return draw::Slice<T>(inner, 0, 0, inner.width(), inner.height()).resize_bottom(offset);
            }
        };

        struct ResizeHorizontal final {
            i32 offset;

            template <SizedPlane T> constexpr auto operator()(T inner) const noexcept -> draw::Slice<T> {
                return draw::Slice<T>(inner, 0, 0, inner.width(), inner.height()).resize_horizontal(offset);
            }
        };

        struct ResizeVertical final {
            i32 offset;

            template <SizedPlane T> constexpr auto operator()(T inner) const noexcept -> draw::Slice<T> {
                return draw::Slice<T>(inner, 0, 0, inner.width(), inner.height()).resize_vertical(offset);
            }
        };

        struct Resize final {
            i32 horizontal, vertical;

            template <SizedPlane T> constexpr auto operator()(T inner) const noexcept -> draw::Slice<T> {
                return draw::Slice<T>(inner, 0, 0, inner.width(), inner.height())
                    .resize_horizontal(horizontal)
                    .resize_vertical(vertical);
            }
        };
    }

    /// Slices an area from a drawable.
    constexpr adapt::Slice slice(i32 x, i32 y, i32 width, i32 height) noexcept {
        return adapt::Slice { x, y, width, height };
    }

    /// Creates a grid which slices out tiles of provided size.
    constexpr adapt::Grid grid(i32 tile_width, i32 tile_height) noexcept {
        return adapt::Grid { tile_width, tile_height };
    }

    /// Similar to the slice shift but wraps the type in a slice first.
    constexpr adapt::Shift shift(i32 x, i32 y) noexcept {
        return adapt::Shift { x, y };
    }

    /// Wraps a sized drawable in a slice of matching proportions.
    /// Convenient for accessing slice operations from a neutral area.
    constexpr adapt::AsSlice as_slice() noexcept {
        return adapt::AsSlice {};
    }

    /// Explicitly wraps a drawable in a reference wrapper.
    /// This is usually done through the implicit conversion but can still be useful in a chained expression.
    constexpr adapt::AsRef as_ref() noexcept {
        return adapt::AsRef {};
    }

    constexpr adapt::Move move() noexcept {
        return adapt::Move {};
    }

    constexpr adapt::AsConst as_const() noexcept {
        return adapt::AsConst {};
    }

    constexpr adapt::Discard discard() noexcept {
        return adapt::Discard {};
    }

    constexpr adapt::Tile tile(i32 x, i32 y) noexcept {
        return adapt::Tile { x, y };
    }

    constexpr adapt::TileRef tile_ref(i32 x, i32 y) noexcept {
        return adapt::TileRef { x, y };
    }

    /// Resizes a drawable's left bound by wrapping it in a slice.
    constexpr adapt::ResizeLeft resize_left(i32 offset) noexcept {
        return adapt::ResizeLeft { offset };
    }

    /// Resizes a drawable's right bound by wrapping it in a slice.
    constexpr adapt::ResizeRight resize_right(i32 offset) noexcept {
        return adapt::ResizeRight { offset };
    }

    /// Resizes a drawable's top bound by wrapping it in a slice.
    constexpr adapt::ResizeTop resize_top(i32 offset) noexcept {
        return adapt::ResizeTop { offset };
    }

    /// Resizes a drawable's bottom bound by wrapping it in a slice.
    constexpr adapt::ResizeBottom resize_bottom(i32 offset) noexcept {
        return adapt::ResizeBottom { offset };
    }

    /// Resizes a drawable horizontally from both sides by wrapping it in a slice.
    constexpr adapt::ResizeHorizontal resize_horizontal(i32 offset) noexcept {
        return adapt::ResizeHorizontal { offset };
    }

    /// Resizes a drawable vertically from both sides by wrapping it in a slice.
    constexpr adapt::ResizeVertical resize_vertical(i32 offset) noexcept {
        return adapt::ResizeVertical { offset };
    }

    /// Resizes a drawable symmetrically on both axes by wrapping it in a slice.
    constexpr adapt::Resize resize(i32 offset) noexcept {
        return adapt::Resize { offset, offset };
    }

    /// Resizes a drawable asymmetrically on both axes by wrapping it in a slice.
    constexpr adapt::Resize resize(i32 horizontal, i32 vertical) noexcept {
        return adapt::Resize { horizontal, vertical };
    }
}

// Mapping -------------------------------------------------------------------------------------------------------------
namespace draw {
    template <Plane T, typename F> struct Map final {
        T inner;
        F fn;

        constexpr auto width() const noexcept(noexcept(inner.width())) -> i32 {
            return inner.width();
        }

        constexpr auto height() const noexcept(noexcept(inner.height())) -> i32 {
            return inner.height();
        }

        constexpr auto get(i32 x, i32 y) const noexcept(noexcept(fn(inner.get(x, y), x, y))) -> Color {
            return fn(inner.get(x, y), x, y);
        }

        constexpr void set(i32 x, i32 y, Color color) noexcept(noexcept(inner.set(x, y, fn(color, x, y)))) {
            inner.set(x, y, fn(color, x, y));
        }
    };

    template <Plane T, typename F> struct MapPos final {
        T inner;
        F fn;

        constexpr auto width() const noexcept(noexcept(inner.width())) -> i32 {
            return inner.width();
        }

        constexpr auto height() const noexcept(noexcept(inner.height())) -> i32 {
            return inner.height();
        }

        constexpr auto get(i32 x, i32 y) const noexcept(noexcept(fn(x, y)) and noexcept(inner.get(0, 0))) -> Color {
            auto [px, py] = fn(x, y);
            return inner.get(px, py);
        }

        constexpr void set(i32 x, i32 y, Color color) noexcept(noexcept(fn(x, y)) and noexcept(inner.set(0, 0, color))) {
            auto [px, py] = fn(x, y);
            inner.set(px, py, color);
        }
    };

    namespace adapt {
        template <typename F> struct Map final {
            F fn;

            template <Plane T> constexpr auto operator()(T inner) const noexcept -> draw::Map<T, F> {
                return draw::Map<T, F> { inner, fn };
            }
        };

        template <typename F> struct MapPos final {
            F fn;

            template <Plane T> constexpr auto operator()(T inner) const noexcept -> draw::MapPos<T, F> {
                return draw::MapPos<T, F> { inner, fn };
            }
        };
    }

    /// Performs a custom mapping with a provided functor. Can often be used instead of making entirely new drawables.
    /// Receives the color and position, so the signature is `(Color c, i32 x, i32 y) -> Color`
    template <typename F> constexpr adapt::Map<F> map(F fn) noexcept
        requires requires(Color c, i32 x, i32 y) { fn(c, x, y); }
    {
        return adapt::Map<F> { fn };
    }

    /// Performs a custom mapping with a provided functor. Can often be used instead of making entirely new drawables.
    /// Receives the color and position, so the signature is `(Color c, i32 x, i32 y) -> Color`
    template <typename F> constexpr auto map(F fn) noexcept
        requires requires(Color c) { fn(c); }
    {
        return adapt::Map {
            [fn] (Color c, i32 x, i32 y) { return fn(c); }
        };
    }

    /// Performs a mapping of just the coordinate space.
    /// Receives the position, so the signature is `(i32 x, i32 y) -> (i32 x, i32 y)`
    template <typename F> constexpr adapt::MapPos<F> map_pos(F fn) noexcept {
        return adapt::MapPos<F> { fn };
    }
}

// Effects -------------------------------------------------------------------------------------------------------------
namespace draw {
    namespace adapt {
        template <const bool OFFSET> struct Dither final {
            Color dither_color;

            struct DitherFn final {
                Color dither_color;

                constexpr auto operator()(Color color, i32 x, i32 y) const noexcept -> Color {
                    if constexpr (not OFFSET) {
                         return (x + y) % 2 == 0 ? color : dither_color;
                    } else {
                        return (x + y + 1) % 2 == 0 ? color : dither_color;
                    }
                }
            };

            template <Plane T> constexpr auto operator()(T inner) const noexcept -> draw::Map<T, DitherFn> {
                return draw::Map<T, DitherFn> { inner, DitherFn { dither_color } };
            }
        };
    }

    constexpr adapt::Dither<false> dither(Color color = draw::color::CLEAR) noexcept {
        return adapt::Dither<false> { color };
    }

    constexpr adapt::Dither<true> dither_off(Color color = draw::color::CLEAR) noexcept {
        return adapt::Dither<true> { color };
    }
}

// Shapes --------------------------------------------------------------------------------------------------------------
namespace draw {
    struct Rectangle final {
        i32 w { 0 }, h { 0 };
        Color color { color::WHITE };

        constexpr auto width() const noexcept -> i32 {
            return w;
        }

        constexpr auto height() const noexcept -> i32 {
            return h;
        }

        constexpr auto get(i32 x, i32 y) const noexcept -> Color {
            return x == 0 or y == 0 or x == w - 1 or y == h - 1 ? color : color::CLEAR;
        }
    };

    static_assert(SizedPlane<Rectangle>);

    struct FilledRectangle final {
        i32 w { 0 }, h { 0 };
        Color color { color::WHITE };

        constexpr auto width() const noexcept -> i32 {
            return w;
        }

        constexpr auto height() const noexcept -> i32 {
            return h;
        }

        constexpr auto get(i32 x, i32 y) const noexcept -> Color {
            return color;
        }
    };

    static_assert(SizedPlane<FilledRectangle>);
}

// Abstract ------------------------------------------------------------------------------------------------------------
namespace draw {
    template <typename F> struct Generator final {
        F fn;

        constexpr Generator(F&& fn) : fn(fn) {}

        constexpr auto get(i32 x, i32 y) const noexcept(noexcept(fn(x, y))) -> Color {
            return fn(x, y);
        }
    };

    template <SizedPlane T> struct Repeat final {
        T inner;

      public:
        constexpr auto width() const noexcept(noexcept(inner.width())) -> i32 {
            return inner.width();
        }

        constexpr auto height() const noexcept(noexcept(inner.height())) -> i32 {
            return inner.height();
        }

        constexpr auto get(i32 x, i32 y) const
            noexcept(noexcept(inner.get(math::arithmetic_mod(x, width()), math::arithmetic_mod(y, height())))) -> Color
        {
            return inner.get(
                math::arithmetic_mod(x, width()),
                math::arithmetic_mod(y, height())
            );
        }

        constexpr void set(i32 x, i32 y, Color color)
            noexcept(noexcept(inner.set(math::arithmetic_mod(x, width()), math::arithmetic_mod(y, height()), color)))
        {
            return inner.set(
                math::arithmetic_mod(x, width()),
                math::arithmetic_mod(y, height()),
                color
            );
        }
    };

    namespace adapt {
        struct Repeat final {
            template <SizedPlane T> constexpr auto operator()(T inner) const noexcept -> draw::Repeat<T> {
                return draw::Repeat<T> { inner };
            }
        };
    }

    /// Tiles the drawable infinitely outside of its sized area.
    constexpr adapt::Repeat repeat() noexcept {
        return adapt::Repeat {};
    }
}

// Conditionals --------------------------------------------------------------------------------------------------------
namespace draw {
    template <Plane Left, Plane Right> class EitherPlane final {
        enum class Case { L, R } tag;
        bool is_alive { true };

        union {
            Left left;
            Right right;
        };

      public:
        constexpr EitherPlane(Left value) noexcept : tag(Case::L), left(value) {}
        constexpr EitherPlane(Right value) noexcept : tag(Case::R), right(value) {}

        ~EitherPlane() noexcept {
            if (is_alive) {
                switch (tag) {
                    case Case::L: left.~Left(); break;
                    case Case::R: right.~Right(); break;
                }
                is_alive = false;
            }
        }

        constexpr auto width() const noexcept(noexcept(left.width()) and noexcept(right.width())) -> i32
        requires
            SizedPlane<Left> and SizedPlane<Right>
        {
            switch (tag) {
                case Case::L: return left.width();
                case Case::R: return right.width();
            }
        }

        constexpr auto height() const noexcept(noexcept(left.height()) and noexcept(right.height())) -> i32
        requires
            SizedPlane<Left> and SizedPlane<Right>
        {
            switch (tag) {
                case Case::L: return left.height();
                case Case::R: return right.height();
            }
        }

        constexpr auto get(i32 x, i32 y) const noexcept(noexcept(left.get(x, y)) and noexcept(right.get(x, y))) -> Color {
            switch (tag) {
                case Case::L: return left.get(x, y);
                case Case::R: return right.get(x, y);
            }
        }
    };

    namespace adapt {
        template <typename F> struct ApplyIf final {
            bool cond;
            F fn;

            constexpr ApplyIf(bool cond, F const& fn) noexcept : cond(cond), fn(fn) {}

            template <Plane T> constexpr auto operator()(T const& inner) const noexcept -> EitherPlane<T, decltype(fn(inner))> {
                if (not cond) {
                    return EitherPlane<T, decltype(fn(inner))>(inner);
                } else {
                    return EitherPlane<T, decltype(fn(inner))>(fn(inner));
                }
            }
        };
    }

    /// Applies another adapter conditionally.
    template <typename F> constexpr adapt::ApplyIf<F> apply_if(bool cond, F fn) noexcept {
        return adapt::ApplyIf<F> { cond, fn };
    }
}

// Layout --------------------------------------------------------------------------------------------------------------
namespace draw {
    enum class MirrorAxis : u8 { X, Y };

    template <const MirrorAxis AXIS, SizedPlane T> struct MirroredPlane final {
        T inner;

        constexpr explicit MirroredPlane(T inner) noexcept : inner(inner) {}

        constexpr auto width() const noexcept(noexcept(inner.width())) -> i32 {
            return inner.width();
        }

        constexpr auto height() const noexcept(noexcept(inner.height())) -> i32 {
            return inner.height();
        }

        constexpr auto get(i32 x, i32 y) const noexcept(noexcept(inner.get(width(), height()))) -> Color {
            if constexpr (AXIS == MirrorAxis::X) {
                return inner.get(width() - 1 - x, y);
            } else {
                return inner.get(x, height() - 1 - y);
            }
        }

        constexpr void set(i32 x, i32 y, Color color) noexcept(noexcept(inner.set(width(), height(), color))) {
            if constexpr (AXIS == MirrorAxis::X) {
                inner.set(width() - 1 - x, y, color);
            } else {
                inner.set(x, height() - 1 - y, color);
            }
        }
    };

    template <SizedPlane T> struct RotatedPlane final {
        T inner;
        i32 rotation_step;

        constexpr RotatedPlane(T inner, i32 rotation_step) noexcept : inner(inner), rotation_step(rotation_step) {}

      private:
        constexpr auto normalized_step() const noexcept -> i32 {
            return math::arithmetic_mod(rotation_step, 4);
        }

      public:
        constexpr auto width() const noexcept(noexcept(inner.width()) and noexcept(inner.height())) -> i32 {
            switch (normalized_step()) {
                case 0:
                case 2:
                    return inner.width();
                case 1:
                case 3:
                    return inner.height();
            }
            std::unreachable();
        }

        constexpr auto height() const noexcept(noexcept(inner.width()) and noexcept(inner.height())) -> i32 {
            switch (normalized_step()) {
                case 0:
                case 2:
                    return inner.height();
                case 1:
                case 3:
                    return inner.width();
            }
            std::unreachable();
        }

        constexpr auto get(i32 x, i32 y) const noexcept(noexcept(inner.get(inner.width(), inner.height()))) -> Color {
            switch (normalized_step()) {
                case 0: return inner.get(x, y);
                case 1: return inner.get(inner.width() - 1 - y, x);
                case 2: return inner.get(inner.width() - 1 - x, inner.height() - 1 - y);
                case 3: return inner.get(y, inner.height() - 1 - x);
            }
            std::unreachable();
        }

        constexpr void set(i32 x, i32 y, Color color) noexcept(noexcept(inner.set(inner.width(), inner.height(), color))) {
            switch (normalized_step()) {
                case 0: inner.set(x, y, color);                                          break;
                case 1: inner.set(inner.width() - 1 - y, x, color);                      break;
                case 2: inner.set(inner.width() - 1 - x, inner.height() - 1 - y, color); break;
                case 3: inner.set(y, inner.height() - 1 - x, color);                     break;
            }
            std::unreachable();
        }
    };

    template <Plane T> struct RotatedGlobalPlane final {
        T inner;
        i32 rotation_step;

        constexpr explicit RotatedGlobalPlane(T inner, i32 rotation_step) noexcept
            : inner(inner), rotation_step(rotation_step) {}

      private:
        constexpr auto normalized_step() const noexcept -> i32 {
            return math::arithmetic_mod(rotation_step, 4);
        }

      public:
        constexpr auto width() const noexcept(noexcept(inner.width())) -> i32 {
            return inner.width();
        }

        constexpr auto height() const noexcept(noexcept(inner.height())) -> i32 {
            return inner.height();
        }

        constexpr auto get(i32 x, i32 y) const noexcept(noexcept(inner.get(x, y))) -> Color {
            switch (normalized_step()) {
                case 0: return inner.get(x, y);
                case 1: return inner.get(y, -x);
                case 2: return inner.get(-x, -y);
                case 3: return inner.get(-y, x);
            }
            std::unreachable();
        }

        constexpr void set(i32 x, i32 y, Color color) noexcept(noexcept(inner.set(x, y, color))) {
            switch (normalized_step()) {
                case 0: inner.set(+x, +y, color); break;
                case 1: inner.set(+y, -x, color); break;
                case 2: inner.set(-x, -y, color); break;
                case 3: inner.set(-y, +x, color); break;
            }
        }
    };

    template <Plane T> struct ScaledPlane final {
        T inner;
        i32 scale;

        constexpr explicit ScaledPlane(T inner, i32 scale) noexcept
            : inner(inner), scale(scale) {}

        constexpr auto width() const noexcept(noexcept(inner.width())) -> i32 {
            return inner.width() * scale;
        }

        constexpr auto height() const noexcept(noexcept(inner.height())) -> i32 {
            return inner.height() * scale;
        }

        constexpr auto get(i32 x, i32 y) const noexcept(noexcept(inner.get(x, y))) -> Color {
            return inner.get(x / scale, y / scale);
        }

        constexpr void set(i32 x, i32 y, Color color) noexcept(noexcept(inner.set(x, y, color))) {
            inner.set(x / scale, y / scale, color);
        }
    };

    template <Plane T> struct MozaicPlane final {
        T inner;
        i32 scale;

        constexpr explicit MozaicPlane(T inner, i32 scale) noexcept
            : inner(inner), scale(scale) {}

        constexpr auto width() const noexcept(noexcept(inner.width())) -> i32 {
            return inner.width() / scale;
        }

        constexpr auto height() const noexcept(noexcept(inner.height())) -> i32 {
            return inner.height() / scale;
        }

        constexpr auto get(i32 x, i32 y) const noexcept(noexcept(inner.get(x, y))) -> Color {
            return inner.get(x * scale, y * scale);
        }

        constexpr void set(i32 x, i32 y, Color color) noexcept(noexcept(inner.set(x, y, color))) {
            inner.set(x * scale, y * scale, color);
        }
    };

    namespace adapt {
        template <const MirrorAxis AXIS> struct Mirror final {
            template <SizedPlane T> constexpr auto operator()(T inner) const noexcept -> MirroredPlane<AXIS, T> {
                return MirroredPlane<AXIS, T>(inner);
            }
        };

        struct Rotate final {
            i32 rotation_step;

            template <SizedPlane T> constexpr auto operator()(T inner) const noexcept -> RotatedPlane<T> {
                return RotatedPlane<T>(inner, rotation_step);
            }
        };

        struct RotateGlobal final {
            i32 rotation_step;

            template <SizedPlane T> constexpr auto operator()(T inner) const noexcept -> RotatedGlobalPlane<T> {
                return RotatedGlobalPlane<T>(inner, rotation_step);
            }
        };

        struct Scale final {
            i32 scale;

            template <Plane T> constexpr auto operator()(T inner) const noexcept -> ScaledPlane<T> {
                return ScaledPlane(inner, scale);
            }
        };

        struct Mozaic final {
            i32 scale;

            template <Plane T> constexpr auto operator()(T inner) const noexcept -> MozaicPlane<T> {
                return MozaicPlane(inner, scale);
            }
        };
    }

    /// Mirrors the drawable in terms of its sized area on the x axis.
    constexpr adapt::Mirror<MirrorAxis::X> mirror_x() noexcept {
        return adapt::Mirror<MirrorAxis::X> {};
    }

    /// Mirrors the drawable in terms of its sized area on the y axis.
    constexpr adapt::Mirror<MirrorAxis::Y> mirror_y() noexcept {
        return adapt::Mirror<MirrorAxis::Y> {};
    }

    /// Rotates the drawable in terms of its sized area clockwise, can accept negative values to rotate counter-clockwise.
    constexpr adapt::Rotate rotate(i32 step) noexcept {
        return adapt::Rotate { step };
    }

    /// Rotates the drawable in terms of the global origin, can accept negative values to rotate counter-clockwise.
    constexpr adapt::RotateGlobal rotate_global(i32 step) noexcept {
        return adapt::RotateGlobal { step };
    }

    constexpr adapt::Scale scale(i32 scale) noexcept {
        return adapt::Scale { scale };
    }

    /// An inversly scaled plane. Viewing a plane through a mozaic makes it seem smaller by the scale value,
    /// and the sampling becomes sparse and truncated. Writing sets entire chunks of pixels.
    ///
    /// A mozaic can be used as an interestic graphical effect or a quick way to draw at a lower resolution into
    /// what would otherwise be a higher resolution plane.
    constexpr adapt::Mozaic mozaic(i32 scale) noexcept {
        return adapt::Mozaic { scale };
    }
}
