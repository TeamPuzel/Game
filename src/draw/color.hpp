// Created by Lua (TeamPuzel) on May 29th 2025.
// Copyright (c) 2025 All rights reserved.
//
// This header defines color functionality such as blending.
// The sonic game does not need blending so only the binary blending mode is implemented.
#pragma once
#include <primitive>

namespace draw {
    struct Color final {
        u8 r { 0 }, g { 0 }, b { 0 }, a { 0 };

        [[clang::always_inline]] [[gnu::const]]
        static constexpr Color rgba(u8 r, u8 g, u8 b, u8 a = 255) noexcept {
            return Color { r, g, b, a };
        }

        [[clang::always_inline]] [[gnu::const]]
        static constexpr Color gray(u8 c, u8 a = 255) noexcept {
            return Color { c, c, c, a };
        }

        template <typename Blend>
        [[clang::always_inline]]
        constexpr auto blend_over(this Color self, Color other, Blend const& blend) noexcept -> Color {
            return blend(self, other);
        }

        template <typename Blend>
        [[clang::always_inline]]
        constexpr auto blend_under(this Color self, Color other, Blend const& blend) noexcept -> Color {
            return blend(other, self);
        }

        [[clang::always_inline]] [[gnu::const]]
        constexpr auto operator==(this Color self, Color other) noexcept -> bool {
            return self.r == other.r and self.g == other.g and self.b == other.b and self.a == other.a;
        }

        [[clang::always_inline]] [[gnu::const]]
        constexpr auto operator!=(this Color self, Color other) noexcept -> bool {
            return !(self == other);
        }

        [[clang::always_inline]] [[gnu::const]]
        constexpr auto with_r(this Color self, u8 r) noexcept -> Color {
            return { r, self.g, self.b, self.a };
        }

        [[clang::always_inline]] [[gnu::const]]
        constexpr auto with_g(this Color self, u8 g) noexcept -> Color {
            return { self.r, g, self.b, self.a };
        }

        [[clang::always_inline]] [[gnu::const]]
        constexpr auto with_b(this Color self, u8 b) noexcept -> Color {
            return { self.r, self.g, b, self.a };
        }

        [[clang::always_inline]] [[gnu::const]]
        constexpr auto with_a(this Color self, u8 a) noexcept -> Color {
            return { self.r, self.g, self.b, a };
        }
    };

    namespace blend {
        [[clang::always_inline]] [[gnu::const]]
        constexpr auto overwrite(Color top, Color bottom) noexcept -> Color {
            return top;
        }

        /// The default style of blending, if any transparency is present the color is discarded
        /// completely and no blending is actually performed.
        ///
        /// This is a great default because it remains associative unlike more advanced alpha blending.
        [[clang::always_inline]] [[gnu::const]]
        constexpr auto binary(Color top, Color bottom) noexcept -> Color {
            return top.a == 255 ? top : bottom;
        }

        /// Alpha blending, not intended for use by the game since the original hardware didn't support
        /// transparency, however it could be useful for transparent debug overlays.
        [[clang::always_inline]] [[gnu::const]]
        constexpr auto alpha(Color top, Color bottom) noexcept -> Color {
            const i32 tr = top.r, tg = top.g, tb = top.b, ta = top.a;
            const i32 br = bottom.r, bg = bottom.g, bb = bottom.b, ba = bottom.a;
            const i32 inv_a = 255 - ta;

            return Color::rgba(
                (tr * ta + br * inv_a) / 255,
                (tg * ta + bg * inv_a) / 255,
                (tb * ta + bb * inv_a) / 255,
                (ta + (ba * inv_a) / 255)
            );
        }
    }

    namespace color {
        constexpr Color CLEAR = Color::rgba(0, 0, 0, 0);
        constexpr Color WHITE = Color::rgba(255, 255, 255);
        constexpr Color BLACK = Color::rgba(0, 0, 0);
    }

    namespace color::pico {
        constexpr Color BLACK       = Color::rgba(0,   0,   0  );
        constexpr Color DARK_BLUE   = Color::rgba(29,  43,  83 );
        constexpr Color DARK_PURPLE = Color::rgba(126, 37,  83 );
        constexpr Color DARK_GREEN  = Color::rgba(0,   135, 81 );
        constexpr Color BROWN       = Color::rgba(171, 82,  53 );
        constexpr Color DARK_GRAY   = Color::rgba(95,  87,  79 );
        constexpr Color LIGHT_GRAY  = Color::rgba(194, 195, 199);
        constexpr Color WHITE       = Color::rgba(255, 241, 232);
        constexpr Color RED         = Color::rgba(255, 0,   77 );
        constexpr Color ORANGE      = Color::rgba(255, 163, 0  );
        constexpr Color YELLOW      = Color::rgba(255, 236, 39 );
        constexpr Color GREEN       = Color::rgba(0,   228, 54 );
        constexpr Color LIGHT_BLUE  = Color::rgba(41,  173, 255);
        constexpr Color LAVENDER    = Color::rgba(131, 118, 156);
        constexpr Color PINK        = Color::rgba(255, 119, 168);
        constexpr Color PEACH       = Color::rgba(255, 204, 170);

        constexpr Color BLUE         = Color::rgba(48,  93,  166);
        constexpr Color TEAL         = Color::rgba(73,  162, 160);
        constexpr Color VIOLET       = Color::rgba(111, 80,  147);
        constexpr Color DARK_TEAL    = Color::rgba(32,  82,  88 );
        constexpr Color DARK_BROWN   = Color::rgba(108, 51,  44 );
        constexpr Color UMBER        = Color::rgba(69,  46,  56 );
        constexpr Color GRAY         = Color::rgba(158, 137, 123);
        constexpr Color LIGHT_PINK   = Color::rgba(243, 176, 196);
        constexpr Color CRIMSON      = Color::rgba(179, 37,  77 );
        constexpr Color DARK_ORANGE  = Color::rgba(219, 114, 44 );
        constexpr Color LIME         = Color::rgba(165, 234, 95 );
        constexpr Color DARK_LIME    = Color::rgba(79,  175, 92 );
        constexpr Color SKY          = Color::rgba(133, 220, 243);
        constexpr Color LIGHT_VIOLET = Color::rgba(183, 155, 218);
        constexpr Color MAGENTA      = Color::rgba(208, 48,  167);
        constexpr Color DARK_PEACH   = Color::rgba(239, 139, 116);
    }
}

namespace color::nes {

}
