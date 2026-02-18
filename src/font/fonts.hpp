// Created by Lua (TeamPuzel) on August 10th 2025.
// Copyright (c) 2025 All rights reserved.
#pragma once
#include <draw>
#include <io>

namespace font {
    using draw::Font;
    using draw::TgaImage;
    using draw::Image;
    using draw::Ref;

    inline auto sonic(Io& io = Io::unsafe_get_threadlocal_io()) -> Font<Ref<const Image>, char> const& {
        using Inner = Ref<const Image>;
        using Symbol = draw::Symbol<Inner>;

        static auto sonicfont = TgaImage::from(io.read_file("res/sonicfont.tga"))
            | draw::flatten<Image>();

        static Font<Ref<const Image>, char> font = {
            sonicfont, // source
            10, // height
            0, // baseline
            2, // spacing
            1, // leading
            [] (Ref<const Image> const& src, char c) -> Symbol {
                const auto grid = src | draw::grid(9, 10);

                switch (c) {
                    case ' ': return Symbol::Space { 5 };

                    case 'A': case 'a': return grid.tile(1,  0).resize_right(-3);
                    case 'B': case 'b': return grid.tile(2,  0).resize_right(-3);
                    case 'C': case 'c': return grid.tile(3,  0).resize_right(-3);
                    case 'D': case 'd': return grid.tile(4,  0).resize_right(-3);
                    case 'E': case 'e': return grid.tile(5,  0).resize_right(-3);
                    case 'F': case 'f': return grid.tile(6,  0).resize_right(-3);
                    case 'G': case 'g': return grid.tile(7,  0).resize_right(-3);
                    case 'H': case 'h': return grid.tile(8,  0).resize_right(-3);
                    case 'I': case 'i': return grid.tile(9,  0).resize_right(-7);
                    case 'J': case 'j': return grid.tile(10, 0).resize_right(-3);
                    case 'K': case 'k': return grid.tile(11, 0).resize_right(-2);
                    case 'L': case 'l': return grid.tile(12, 0).resize_right(-4);
                    case 'M': case 'm': return grid.tile(13, 0);
                    case 'N': case 'n': return grid.tile(14, 0).resize_right(-1);
                    case 'O': case 'o': return grid.tile(15, 0).resize_right(-3);
                    case 'P': case 'p': return grid.tile(16, 0).resize_right(-3);
                    case 'Q': case 'q': return grid.tile(17, 0).resize_right(-3);
                    case 'R': case 'r': return grid.tile(18, 0).resize_right(-3);
                    case 'S': case 's': return grid.tile(19, 0).resize_right(-3);
                    case 'T': case 't': return grid.tile(20, 0).resize_right(-3);
                    case 'U': case 'u': return grid.tile(21, 0).resize_right(-3);
                    case 'V': case 'v': return grid.tile(22, 0).resize_right(-3);
                    case 'W': case 'w': return grid.tile(23, 0);
                    case 'X': case 'x': return grid.tile(24, 0).resize_right(-2);
                    case 'Y': case 'y': return grid.tile(25, 0).resize_right(-3);
                    case 'Z': case 'z': return grid.tile(26, 0).resize_right(-2);

                    case '0': return grid.tile(27, 0).resize_right(-3);
                    case '1': return grid.tile(28, 0).resize_right(-6);
                    case '2': return grid.tile(29, 0).resize_right(-3);
                    case '3': return grid.tile(30, 0).resize_right(-3);
                    case '4': return grid.tile(31, 0).resize_right(-3);
                    case '5': return grid.tile(32, 0).resize_right(-3);
                    case '6': return grid.tile(33, 0).resize_right(-3);
                    case '7': return grid.tile(34, 0).resize_right(-2);
                    case '8': return grid.tile(35, 0).resize_right(-3);
                    case '9': return grid.tile(36, 0).resize_right(-3);

                    case ':': return grid.tile(37, 0).resize_right(-7);
                    case ';': return grid.tile(38, 0).resize_right(-7);

                    case '.': return grid.tile(39, 0).resize_right(-7);
                    case ',': return grid.tile(40, 0).resize_right(-7);

                    default: return grid.tile(0, 0).resize_right(-3);
                }
            },
        };

        return font;
    }

    inline auto pico(Io& io = Io::unsafe_get_threadlocal_io()) -> Font<Ref<const Image>, char> const& {
        using Inner = Ref<const Image>;
        using Symbol = draw::Symbol<Inner>;

        static auto minefont = TgaImage::from(io.read_file("res/picofont.tga"))
            | draw::flatten<Image>();

        static Font<Ref<const Image>, char> font = {
            minefont, // source
            5, // height
            0, // baseline
            1, // spacing
            1, // leading
            [] (Ref<const Image> const& src, char c) -> Symbol {
                const auto grid = src | draw::grid(3, 5);

                switch (c) {
                    case ' ': return Symbol::Space { 3 };

                    case '0': return grid.tile(0, 0);
                    case '1': return grid.tile(1, 0);
                    case '2': return grid.tile(2, 0);
                    case '3': return grid.tile(3, 0);
                    case '4': return grid.tile(4, 0);
                    case '5': return grid.tile(5, 0);
                    case '6': return grid.tile(6, 0);
                    case '7': return grid.tile(7, 0);
                    case '8': return grid.tile(8, 0);
                    case '9': return grid.tile(9, 0);

                    case 'A': case 'a': return grid.tile(10, 0);
                    case 'B': case 'b': return grid.tile(11, 0);
                    case 'C': case 'c': return grid.tile(12, 0);
                    case 'D': case 'd': return grid.tile(13, 0);
                    case 'E': case 'e': return grid.tile(14, 0);
                    case 'F': case 'f': return grid.tile(15, 0);
                    case 'G': case 'g': return grid.tile(16, 0);
                    case 'H': case 'h': return grid.tile(17, 0);
                    case 'I': case 'i': return grid.tile(18, 0);
                    case 'J': case 'j': return grid.tile(19, 0);
                    case 'K': case 'k': return grid.tile(20, 0);
                    case 'L': case 'l': return grid.tile(21, 0);
                    case 'M': case 'm': return grid.tile(22, 0);
                    case 'N': case 'n': return grid.tile(23, 0);
                    case 'O': case 'o': return grid.tile(24, 0);
                    case 'P': case 'p': return grid.tile(25, 0);
                    case 'Q': case 'q': return grid.tile(26, 0);
                    case 'R': case 'r': return grid.tile(27, 0);
                    case 'S': case 's': return grid.tile(28, 0);
                    case 'T': case 't': return grid.tile(29, 0);
                    case 'U': case 'u': return grid.tile(30, 0);
                    case 'V': case 'v': return grid.tile(31, 0);
                    case 'W': case 'w': return grid.tile(32, 0);
                    case 'X': case 'x': return grid.tile(33, 0);
                    case 'Y': case 'y': return grid.tile(34, 0);
                    case 'Z': case 'z': return grid.tile(35, 0);

                    case '.':  return grid.tile(36, 0);
                    case ',':  return grid.tile(37, 0);
                    case '!':  return grid.tile(38, 0);
                    case '?':  return grid.tile(39, 0);
                    case '"':  return grid.tile(40, 0);
                    case '\'': return grid.tile(41, 0);
                    case '`':  return grid.tile(42, 0);
                    case '@':  return grid.tile(43, 0);
                    case '#':  return grid.tile(44, 0);
                    case '$':  return grid.tile(45, 0);
                    case '%':  return grid.tile(46, 0);
                    case '&':  return grid.tile(47, 0);
                    case '(':  return grid.tile(48, 0);
                    case ')':  return grid.tile(49, 0);
                    case '[':  return grid.tile(50, 0);
                    case ']':  return grid.tile(51, 0);
                    case '{':  return grid.tile(52, 0);
                    case '}':  return grid.tile(53, 0);
                    case '|':  return grid.tile(54, 0);
                    case '/':  return grid.tile(55, 0);
                    case '\\': return grid.tile(56, 0);
                    case '+':  return grid.tile(57, 0);
                    case '-':  return grid.tile(58, 0);
                    case '*':  return grid.tile(59, 0);
                    case ':':  return grid.tile(60, 0);
                    case ';':  return grid.tile(61, 0);
                    case '=':  return grid.tile(62, 0);
                    case '<':  return grid.tile(63, 0);
                    case '>':  return grid.tile(64, 0);
                    case '_':  return grid.tile(65, 0);
                    case '~':  return grid.tile(66, 0);

                    default: return Symbol::Space { 3 };
                }
            },
        };

        return font;
    }

    inline auto mine(Io& io = Io::unsafe_get_threadlocal_io()) -> Font<Ref<const Image>, char> const& {
        using Inner = Ref<const Image>;
        using Symbol = draw::Symbol<Inner>;

        static auto minefont = TgaImage::from(io.read_file("res/minefont.tga"))
            | draw::flatten<Image>();

        static Font<Ref<const Image>, char> font = {
            minefont, // source
            8, // height
            1, // baseline
            1, // spacing
            1, // leading
            [] (Ref<const Image> const& src, char c) -> Symbol {
                const auto grid = src | draw::grid(5, 8);

                switch (c) {
                    case ' ': return Symbol::Space { 3 };

                    case 'A': return grid.tile(1,   0);
                    case 'B': return grid.tile(2,   0);
                    case 'C': return grid.tile(3,   0);
                    case 'D': return grid.tile(4,   0);
                    case 'E': return grid.tile(5,   0);
                    case 'F': return grid.tile(6,   0);
                    case 'G': return grid.tile(7,   0);
                    case 'H': return grid.tile(8,   0);
                    case 'I': return grid.tile(9,   0).resize_horizontal(-1);
                    case 'J': return grid.tile(10,  0);
                    case 'K': return grid.tile(11,  0);
                    case 'L': return grid.tile(12,  0);
                    case 'M': return grid.tile(13,  0);
                    case 'N': return grid.tile(14,  0);
                    case 'O': return grid.tile(15,  0);
                    case 'P': return grid.tile(16,  0);
                    case 'Q': return grid.tile(17,  0);
                    case 'R': return grid.tile(18,  0);
                    case 'S': return grid.tile(19,  0);
                    case 'T': return grid.tile(20,  0);
                    case 'U': return grid.tile(21,  0);
                    case 'V': return grid.tile(22,  0);
                    case 'W': return grid.tile(23,  0);
                    case 'X': return grid.tile(24,  0);
                    case 'Y': return grid.tile(25,  0);
                    case 'Z': return grid.tile(26,  0);

                    case 'a': return grid.tile(27,  0);
                    case 'b': return grid.tile(28,  0);
                    case 'c': return grid.tile(29,  0);
                    case 'd': return grid.tile(30,  0);
                    case 'e': return grid.tile(31,  0);
                    case 'f': return grid.tile(32,  0).resize_left(-1);
                    case 'g': return grid.tile(33,  0);
                    case 'h': return grid.tile(34,  0);
                    case 'i': return grid.tile(35,  0).resize_horizontal(-2);
                    case 'j': return grid.tile(36,  0);
                    case 'k': return grid.tile(37,  0).resize_left(-1);
                    case 'l': return grid.tile(38,  0).resize_left(-1).resize_right(-2);
                    case 'm': return grid.tile(39,  0);
                    case 'n': return grid.tile(40,  0);
                    case 'o': return grid.tile(41,  0);
                    case 'p': return grid.tile(42,  0);
                    case 'q': return grid.tile(43,  0);
                    case 'r': return grid.tile(44,  0);
                    case 's': return grid.tile(45,  0);
                    case 't': return grid.tile(46,  0).resize_horizontal(-1);
                    case 'u': return grid.tile(47,  0);
                    case 'v': return grid.tile(48,  0);
                    case 'w': return grid.tile(49,  0);
                    case 'x': return grid.tile(50,  0);
                    case 'y': return grid.tile(51,  0);
                    case 'z': return grid.tile(52,  0);

                    case '0': return grid.tile(53,  0);
                    case '1': return grid.tile(54,  0);
                    case '2': return grid.tile(55,  0);
                    case '3': return grid.tile(56,  0);
                    case '4': return grid.tile(57,  0);
                    case '5': return grid.tile(58,  0);
                    case '6': return grid.tile(59,  0);
                    case '7': return grid.tile(60,  0);
                    case '8': return grid.tile(61,  0);
                    case '9': return grid.tile(62,  0);

                    case '.':  return grid.tile(63,  0).resize_horizontal(-2);
                    case ',':  return grid.tile(64,  0).resize_horizontal(-2);
                    case ':':  return grid.tile(65,  0).resize_horizontal(-2);
                    case ';':  return grid.tile(66,  0).resize_horizontal(-2);
                    case '\'': return grid.tile(67,  0).resize_horizontal(-2);
                    case '"':  return grid.tile(68,  0).resize_horizontal(-1);
                    case '!':  return grid.tile(69,  0).resize_horizontal(-2);
                    case '?':  return grid.tile(70,  0);

                    case '#': return grid.tile(71,  0);
                    case '%': return grid.tile(72,  0);
                    case '&': return grid.tile(73,  0);
                    case '$': return grid.tile(74,  0);
                    case '(': return grid.tile(75,  0).resize_horizontal(-1);
                    case ')': return grid.tile(76,  0).resize_horizontal(-1);

                    case '*': return grid.tile(77,  0).resize_horizontal(-1);
                    case '-': return grid.tile(78,  0).resize_horizontal(-1);
                    case '+': return grid.tile(79,  0).resize_horizontal(-1);
                    // case '×': return grid.tile(80,  0).resize_horizontal(-1);
                    // case '÷': return grid.tile(81,  0).resize_horizontal(-1);

                    case '<': return grid.tile(82,  0).resize_left(-1);
                    case '>': return grid.tile(83,  0).resize_right(-1);
                    case '=': return grid.tile(84,  0).resize_horizontal(-1);

                    case '_': return grid.tile(85,  0);
                    case '[': return grid.tile(86,  0).resize_horizontal(-1);
                    case ']': return grid.tile(87,  0).resize_horizontal(-1);

                    case '/':  return grid.tile(88,  0);
                    case '\\': return grid.tile(89,  0);

                    case '^': return grid.tile(90,  0);
                    // case '±': return grid.tile(91,  0).resize_horizontal(-1);

                    case '@': return grid.tile(92,  0);
                    case '|': return grid.tile(93,  0).resize_horizontal(-2);
                    case '{': return grid.tile(94,  0).resize_horizontal(-1);
                    case '}': return grid.tile(95,  0).resize_horizontal(-1);

                    case '~': return grid.tile(96,  0).resize_right(1);

                    // case '§': return grid.tile(98,  0);

                    // case '©': return grid.tile(99,  0).resize_right(2);
                    // case '®': return grid.tile(101, 0).resize_left(2);
                    // case '™': return grid.tile(102, 0).resize_right(5).resize_left(-1); // TODO(!): Improve with universal resize

                    // case '–': return grid.tile(104, 0);
                    // case '¡': return grid.tile(105, 0).resize_horizontal(-2);
                    // case '¿': return grid.tile(106, 0);
                    // case '£': return grid.tile(107, 0);
                    // case '¥': return grid.tile(108, 0);
                    // case '¢': return grid.tile(109, 0);
                    // case '…': return grid.tile(110, 0);

                    // case '·': return grid.tile(111, 0).resize_horizontal(-2);
                    // case '—': return grid.tile(112, 0).resize_horizontal(2);

                    // case '°': return grid.tile(114, 0).resize_right(-1);

                    default: return grid.tile(0, 0);
                }
            },
        };

        return font;
    }

    inline auto mine_u16(Io& io = Io::unsafe_get_threadlocal_io()) -> Font<Ref<const Image>, char16> const& {
        using Inner = Ref<const Image>;
        using Symbol = draw::Symbol<Inner>;

        static auto minefont = TgaImage::from(io.read_file("res/minefont.tga"))
            | draw::flatten<Image>();

        static Font<Ref<const Image>, char16> font = {
            minefont, // source
            8, // height
            1, // baseline
            1, // spacing
            1, // leading
            [] (Ref<const Image> const& src, char16 c) -> Symbol {
                const auto grid = src | draw::grid(5, 8);

                switch (c) {
                    case u' ': return Symbol::Space { 3 };

                    case u'A': return grid.tile(1,   0);
                    case u'B': return grid.tile(2,   0);
                    case u'C': return grid.tile(3,   0);
                    case u'D': return grid.tile(4,   0);
                    case u'E': return grid.tile(5,   0);
                    case u'F': return grid.tile(6,   0);
                    case u'G': return grid.tile(7,   0);
                    case u'H': return grid.tile(8,   0);
                    case u'I': return grid.tile(9,   0).resize_horizontal(-1);
                    case u'J': return grid.tile(10,  0);
                    case u'K': return grid.tile(11,  0);
                    case u'L': return grid.tile(12,  0);
                    case u'M': return grid.tile(13,  0);
                    case u'N': return grid.tile(14,  0);
                    case u'O': return grid.tile(15,  0);
                    case u'P': return grid.tile(16,  0);
                    case u'Q': return grid.tile(17,  0);
                    case u'R': return grid.tile(18,  0);
                    case u'S': return grid.tile(19,  0);
                    case u'T': return grid.tile(20,  0);
                    case u'U': return grid.tile(21,  0);
                    case u'V': return grid.tile(22,  0);
                    case u'W': return grid.tile(23,  0);
                    case u'X': return grid.tile(24,  0);
                    case u'Y': return grid.tile(25,  0);
                    case u'Z': return grid.tile(26,  0);

                    case u'a': return grid.tile(27,  0);
                    case u'b': return grid.tile(28,  0);
                    case u'c': return grid.tile(29,  0);
                    case u'd': return grid.tile(30,  0);
                    case u'e': return grid.tile(31,  0);
                    case u'f': return grid.tile(32,  0).resize_left(-1);
                    case u'g': return grid.tile(33,  0);
                    case u'h': return grid.tile(34,  0);
                    case u'i': return grid.tile(35,  0).resize_horizontal(-2);
                    case u'j': return grid.tile(36,  0);
                    case u'k': return grid.tile(37,  0).resize_left(-1);
                    case u'l': return grid.tile(38,  0).resize_left(-1).resize_right(-2);
                    case u'm': return grid.tile(39,  0);
                    case u'n': return grid.tile(40,  0);
                    case u'o': return grid.tile(41,  0);
                    case u'p': return grid.tile(42,  0);
                    case u'q': return grid.tile(43,  0);
                    case u'r': return grid.tile(44,  0);
                    case u's': return grid.tile(45,  0);
                    case u't': return grid.tile(46,  0).resize_horizontal(-1);
                    case u'u': return grid.tile(47,  0);
                    case u'v': return grid.tile(48,  0);
                    case u'w': return grid.tile(49,  0);
                    case u'x': return grid.tile(50,  0);
                    case u'y': return grid.tile(51,  0);
                    case u'z': return grid.tile(52,  0);

                    case u'0': return grid.tile(53,  0);
                    case u'1': return grid.tile(54,  0);
                    case u'2': return grid.tile(55,  0);
                    case u'3': return grid.tile(56,  0);
                    case u'4': return grid.tile(57,  0);
                    case u'5': return grid.tile(58,  0);
                    case u'6': return grid.tile(59,  0);
                    case u'7': return grid.tile(60,  0);
                    case u'8': return grid.tile(61,  0);
                    case u'9': return grid.tile(62,  0);

                    case u'.':  return grid.tile(63,  0).resize_horizontal(-2);
                    case u',':  return grid.tile(64,  0).resize_horizontal(-2);
                    case u':':  return grid.tile(65,  0).resize_horizontal(-2);
                    case u';':  return grid.tile(66,  0).resize_horizontal(-2);
                    case u'\'': return grid.tile(67,  0).resize_horizontal(-2);
                    case u'"':  return grid.tile(68,  0).resize_horizontal(-1);
                    case u'!':  return grid.tile(69,  0).resize_horizontal(-2);
                    case u'?':  return grid.tile(70,  0);

                    case u'#': return grid.tile(71,  0);
                    case u'%': return grid.tile(72,  0);
                    case u'&': return grid.tile(73,  0);
                    case u'$': return grid.tile(74,  0);
                    case u'(': return grid.tile(75,  0).resize_horizontal(-1);
                    case u')': return grid.tile(76,  0).resize_horizontal(-1);

                    case u'*': return grid.tile(77,  0).resize_horizontal(-1);
                    case u'-': return grid.tile(78,  0).resize_horizontal(-1);
                    case u'+': return grid.tile(79,  0).resize_horizontal(-1);
                    case u'×': return grid.tile(80,  0).resize_horizontal(-1);
                    case u'÷': return grid.tile(81,  0).resize_horizontal(-1);

                    case u'<': return grid.tile(82,  0).resize_left(-1);
                    case u'>': return grid.tile(83,  0).resize_right(-1);
                    case u'=': return grid.tile(84,  0).resize_horizontal(-1);

                    case u'_': return grid.tile(85,  0);
                    case u'[': return grid.tile(86,  0).resize_horizontal(-1);
                    case u']': return grid.tile(87,  0).resize_horizontal(-1);

                    case u'/':  return grid.tile(88,  0);
                    case u'\\': return grid.tile(89,  0);

                    case u'^': return grid.tile(90,  0);
                    case u'±': return grid.tile(91,  0).resize_horizontal(-1);

                    case u'@': return grid.tile(92,  0);
                    case u'|': return grid.tile(93,  0).resize_horizontal(-2);
                    case u'{': return grid.tile(94,  0).resize_horizontal(-1);
                    case u'}': return grid.tile(95,  0).resize_horizontal(-1);

                    case u'~': return grid.tile(96,  0).resize_right(1);

                    case u'§': return grid.tile(98,  0);

                    case u'©': return grid.tile(99,  0).resize_right(2);
                    case u'®': return grid.tile(101, 0).resize_left(2);
                    case u'™': return grid.tile(102, 0).resize_right(5).resize_left(-1); // TODO(!): Improve with universal resize

                    case u'–': return grid.tile(104, 0);
                    case u'¡': return grid.tile(105, 0).resize_horizontal(-2);
                    case u'¿': return grid.tile(106, 0);
                    case u'£': return grid.tile(107, 0);
                    case u'¥': return grid.tile(108, 0);
                    case u'¢': return grid.tile(109, 0);
                    case u'…': return grid.tile(110, 0);

                    case u'·': return grid.tile(111, 0).resize_horizontal(-2);
                    case u'—': return grid.tile(112, 0).resize_horizontal(2);

                    case u'°': return grid.tile(114, 0).resize_right(-1);

                    default: return grid.tile(0, 0);
                }
            },
        };

        return font;
    }

    inline auto pod(Io& io = Io::unsafe_get_threadlocal_io()) -> Font<Ref<const Image>, char> const& {
        using Inner = Ref<const Image>;
        using Symbol = draw::Symbol<Inner>;

        static auto podfont = TgaImage::from(io.read_file("res/podfont.tga"))
            | draw::flatten<Image>();

        static Font<Ref<const Image>, char> font = {
            podfont, // source
            12, // height
            3,  // baseline
            2,  // spacing
            1,  // leading
            [] (Ref<const Image> const& src, char c) -> Symbol {
                const auto grid = src | draw::grid(6, 12);

                switch (c) {
                    case ' ': return Symbol::Space { 3 };

                    case 'A': return grid.tile(1,   0);
                    case 'B': return grid.tile(2,   0);
                    case 'C': return grid.tile(3,   0);
                    case 'D': return grid.tile(4,   0);
                    case 'E': return grid.tile(5,   0).resize_right(-1);
                    case 'F': return grid.tile(6,   0).resize_right(-1);
                    case 'G': return grid.tile(7,   0);
                    case 'H': return grid.tile(8,   0);
                    case 'I': return grid.tile(9,   0).resize_horizontal(-2);
                    case 'J': return grid.tile(10,  0);
                    case 'K': return grid.tile(11,  0);
                    case 'L': return grid.tile(12,  0).resize_right(-1);
                    case 'M': return grid.tile(13,  0).resize_right(4);
                    case 'N': return grid.tile(15,  0).resize_right(1);
                    case 'O': return grid.tile(17,  0);
                    case 'P': return grid.tile(18,  0);
                    case 'Q': return grid.tile(19,  0);
                    case 'R': return grid.tile(20,  0);
                    case 'S': return grid.tile(21,  0).resize_right(-1);
                    case 'T': return grid.tile(22,  0);
                    case 'U': return grid.tile(23,  0);
                    case 'V': return grid.tile(24,  0);
                    case 'W': return grid.tile(25,  0).resize_right(4);
                    case 'X': return grid.tile(27,  0);
                    case 'Y': return grid.tile(28,  0);
                    case 'Z': return grid.tile(29,  0);

                    case 'a': return grid.tile(30,  0);
                    case 'b': return grid.tile(31,  0);
                    case 'c': return grid.tile(32,  0).resize_right(-1);
                    case 'd': return grid.tile(33,  0);
                    case 'e': return grid.tile(34,  0);
                    case 'f': return grid.tile(35,  0).resize_left(-1);
                    case 'g': return grid.tile(36,  0);
                    case 'h': return grid.tile(37,  0);
                    case 'i': return grid.tile(38,  0).resize_horizontal(-2);
                    case 'j': return grid.tile(39,  0).resize_right(-1);
                    case 'k': return grid.tile(40,  0);
                    case 'l': return grid.tile(41,  0).resize_right(-3);
                    case 'm': return grid.tile(42,  0).resize_right(4);
                    case 'n': return grid.tile(44,  0);
                    case 'o': return grid.tile(45,  0);
                    case 'p': return grid.tile(46,  0);
                    case 'q': return grid.tile(47,  0);
                    case 'r': return grid.tile(48,  0).resize_right(-1);
                    case 's': return grid.tile(49,  0).resize_right(-1);
                    case 't': return grid.tile(50,  0).resize_horizontal(-1);
                    case 'u': return grid.tile(51,  0);
                    case 'v': return grid.tile(52,  0);
                    case 'w': return grid.tile(53,  0).resize_right(4);
                    case 'x': return grid.tile(55,  0);
                    case 'y': return grid.tile(56,  0);
                    case 'z': return grid.tile(57,  0);

                    case '0': return grid.tile(58,  0);
                    case '1': return grid.tile(59,  0).resize_horizontal(-2);
                    case '2': return grid.tile(60,  0);
                    case '3': return grid.tile(61,  0);
                    case '4': return grid.tile(62,  0).resize_right(1);
                    case '5': return grid.tile(64,  0);
                    case '6': return grid.tile(65,  0);
                    case '7': return grid.tile(66,  0).resize_right(1);
                    case '8': return grid.tile(68,  0);
                    case '9': return grid.tile(69,  0);

                    case '+':  return grid.tile(70,  0).resize_right(-3);
                    case '-':  return grid.tile(71,  0).resize_right(-3);
                    case '*':  return grid.tile(72,  0).resize_right(-3);
                    case '/':  return grid.tile(73,  0).resize_right(-1);
                    case '\\': return grid.tile(74,  0).resize_right(-1);
                    case '|':  return grid.tile(75,  0).resize_right(-5);
                    case '=':  return grid.tile(76,  0).resize_right(-3);
                    case '<':  return grid.tile(77,  0).resize_right(-2);
                    case '>':  return grid.tile(78,  0).resize_right(-2);

                    case '%': return grid.tile(79,  0).resize_right(3);

                    case '"':  return grid.tile(84,  0).resize_right(-3);
                    case '\'': return grid.tile(85,  0).resize_right(-5);

                    case '#': return grid.tile(86,  0).resize_right(3);
                    case '@': return grid.tile(88,  0).resize_right(3);
                    case '&': return grid.tile(90,  0).resize_right(2);

                    case '_': return grid.tile(92,  0);
                    case '(': return grid.tile(93,  0).resize_right(-3);
                    case ')': return grid.tile(94,  0).resize_right(-3);
                    case ',': return grid.tile(95,  0).resize_right(-4);
                    case '.': return grid.tile(96,  0).resize_right(-4);
                    case ';': return grid.tile(97,  0).resize_right(-4);
                    case ':': return grid.tile(98,  0).resize_right(-4);
                    case '?': return grid.tile(99,  0);
                    case '!': return grid.tile(100, 0).resize_horizontal(-2);
                    case '{': return grid.tile(101, 0).resize_right(-2);
                    case '}': return grid.tile(102, 0).resize_right(-2);
                    case '[': return grid.tile(103, 0).resize_right(-3);
                    case ']': return grid.tile(104, 0).resize_right(-3);
                    case '`': return grid.tile(105, 0).resize_right(-3);
                    case '^': return grid.tile(106, 0).resize_right(-1);
                    case '~': return grid.tile(107, 0);

                    default: return grid.tile(0, 0);
                }
            },
        };

        return font;
    }
}
