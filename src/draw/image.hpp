// Created by Lua (TeamPuzel) on May 29th 2025.
// Copyright (c) 2025 All rights reserved.
//
// This defines the simplest possible primitive.
// It's not the most primitive however, that title belongs to the InfiniteImage.
#pragma once
#include <primitive>
#include <vector>
#include "plane.hpp"

namespace draw {
    /// The simplest sized primitive, it makes for a general purpose read/write drawable.
    ///
    /// The state surrounding the described sized area is always clear.
    ///
    /// TODO: Replace std::vector with something sane, 24 bytes is a joke.
    class Image final {
        std::vector<Color> data;
        i32 w, h;

      public:
        /// The default, empty image of nil proportions.
        Image() : w(0), h(0) {}

        Image(Image const&) = delete;
        auto operator=(Image const&) -> Image& = delete;
        Image(Image&&) = default;
        auto operator=(Image&&) -> Image& = default;

        /// Initializes the image with the provided function of signature:
        /// (x: i32, y: i32) -> Color
        template <typename F> Image(i32 width, i32 height, F init) : w(width), h(height) {
            data.reserve(width * height);
            data.resize(width * height);
            for (i32 x = 0; x < width; x += 1) {
                for (i32 y = 0; y < height; y += 1) {
                    data.at(x + y * width) = init(x, y);
                }
            }
        }

        Image(i32 width, i32 height) : Image(width, height, [] (i32 x, i32 y) { return color::CLEAR; }) {}

        auto clone() const -> Image {
            return Image(w, h, [this] (i32 x, i32 y) -> Color {
                return this->get(x, y);
            });
        }

        void resize(i32 width, i32 height) {
            *this = Image(width, height, [this] (i32 x, i32 y) -> Color {
                return this->get(x, y);
            });
        }

        auto width() const noexcept -> i32 {
            return w;
        }

        auto height() const noexcept -> i32 {
            return h;
        }

        auto get(i32 x, i32 y) const noexcept -> Color {
            if (x >= 0 and x < w and y >= 0 and y < h) {
                return data[x + y * w];
            } else {
                return color::CLEAR;
            }
        }

        void set(i32 x, i32 y, Color color) noexcept {
            if (x >= 0 and x < w and y >= 0 and y < h) {
                data[x + y * w] = color;
            }
        }

        auto raw() const noexcept -> Color const* {
            return data.data();
        }

        auto raw() -> Color* {
            return data.data();
        }

        template <SizedPlane U> static auto flatten(U const& other) -> Image {
            return Image(other.width(), other.height(), [&] (i32 x, i32 y) -> Color {
                return other.get(x, y);
            });
        }
    };

    // Assert that our type properly satisfies the desired interface.
    static_assert(SizedPlane<Image> and MutablePlane<Image>);

    class TgaImage final {
        std::vector<u8> data;

        struct BGRA final {
            u8 b, g, r, a;

            constexpr operator Color() const noexcept {
                return Color::rgba(r, g, b, a);
            }
        };

        TgaImage(std::vector<u8> data) : data(std::move(data)) {}

      public:
        auto width() const -> i32 {
            return *reinterpret_cast<u16 const*>(data.data() + 12);
        }

        auto height() const -> i32 {
            return *reinterpret_cast<u16 const*>(data.data() + 14);
        }

        auto get(i32 x, i32 y) const -> Color {
            return *reinterpret_cast<BGRA const*>(data.data() + 18 + (x + y * width()) * sizeof(BGRA));
        }

        static auto from(std::vector<u8> data) -> TgaImage {
            return TgaImage(std::move(data));
        }
    };

    // Assert that our type properly satisfies the desired interface.
    static_assert(SizedPlane<TgaImage>);
}
