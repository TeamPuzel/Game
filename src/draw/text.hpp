// Created by Lua (TeamPuzel) on May 29th 2025.
// Copyright (c) 2025 All rights reserved.
//
// This defines the simplest possible primitive.
// It's not the most primitive however, that title belongs to the InfiniteImage.
#pragma once
#include <primitive>
#include <string_view>
#include <optional>
#include <ranges>
#include "color.hpp"
#include "plane.hpp"
#include "image.hpp"

namespace draw {
    template <Plane T> struct Symbol final {
        enum class Type {
            Glyph,
            Space,
        } type;

        using Glyph = Slice<T>;
        struct Space final { i32 width; };

        union {
            Glyph glyph;
            Space space;
        };

        Symbol(Glyph glyph) : type(Type::Glyph), glyph(glyph) {}
        Symbol(Space space) : type(Type::Space), space(space) {}

        auto width() const -> i32 {
            switch (type) {
                case Type::Glyph: return glyph.width();
                case Type::Space: return space.width;
            }
        }
    };

    template <Plane T, typename Chr> struct Font final {
        T source;
        i32 height;
        i32 baseline;
        i32 spacing;
        i32 leading;
        auto (*map) (T const&, Chr) -> Symbol<T>;

        auto symbol(Chr c) const -> Symbol<T> {
            return this->map(source, c);
        }
    };

    /// A drawable representing text.
    ///
    /// Notably this type performs caching to be remotely efficient while maintaining the composable
    /// drawable interface. This type is not thread-safe and has to be guarded if shared in any way.
    ///
    /// The view type used is generic because C++17 is sad and as usual the version prior to anything nice.
    /// Not that it's much better now anyway.
    template <Plane T, typename Str = std::string_view> struct Text final {
        using StringView = Str;
        using Char = typename Str::value_type;

        StringView content;
        Color color;
        Font<T, Char> font;

      private:
        i32 width_cache;
        mutable std::optional<Image> cache;

        auto redraw() const -> Image {
            using SymbolType = typename Symbol<T>::Type;

            auto ret = Image(width(), height());

            i32 cursor = 0;

            for (Char c : content) {
                auto sym = font.symbol(c);
                switch (sym.type) {
                    case SymbolType::Glyph:
                        ret | draw(
                            sym.glyph | draw::map([this] (Color c, i32 x, i32 y) -> Color {
                                return c == color::WHITE ? color : c;
                            }),
                            cursor, 0,
                            blend::overwrite
                        );

                        cursor += sym.width() + font.spacing;
                        break;
                    case SymbolType::Space:
                        cursor += sym.width();
                        break;
                }
            }

            return ret;
        }

      public:
        Text(StringView content, Font<T, Char> font, Color color = color::WHITE)
            : content(content), color(color), font(font)
        {
            if (content.empty()) {
                this->width_cache = 0;
            } else {
                i32 acc = -font.spacing;
                for (Char c : content) acc += font.symbol(c).width() + font.spacing;
                this->width_cache = acc;
            }
        }

        auto width() const -> i32 {
            return width_cache;
        }

        auto height() const -> i32 {
            return font.height;
        }

        auto get(i32 x, i32 y) const -> Color {
            if (not cache) cache = redraw();
            return cache->get(x, y);
        }
    };

    template <typename T> Text(char const*, Font<T, char>, Color = color::WHITE) -> Text<T, std::string_view>;

    static_assert(SizedPlane<Text<Image>>);

    // TODO: Alignment
    template <Plane T, typename Str = std::string_view> struct MultilineText final {
        using StringView = Str;
        using Char = typename Str::value_type;

        StringView content;
        Color color;
        Font<T, Char> font;

      private:
        i32 width_cache;
        i32 height_cache;
        mutable std::optional<Image> cache;

        decltype(auto) lines() const {
            return content
                | std::views::split('\n')
                | std::views::transform([](auto&& r) {
                    if (r.begin() == r.end()) return {}; // Iterating an exhausted iterator makes STL assertions sad.
                    return StringView(&*r.begin(), usize(std::ranges::distance(r)));
                });
        }

        auto line_width(StringView line) const -> i32 {
            if (line.empty()) return 0;
            i32 w = -font.spacing;
            for (Char c : line)
                w += font.symbol(c).width() + font.spacing;
            return w;
        }

        auto redraw() const -> Image {
            using SymbolType = typename Symbol<T>::Type;
            Image ret(width_cache, height_cache);

            i32 cursor_y = 0;
            for (auto line : lines()) {
                i32 cursor_x = 0;
                for (Char c : line) {
                    auto sym = font.symbol(c);
                    switch (sym.type) {
                        case SymbolType::Glyph:
                            ret | draw(
                                sym.glyph | draw::map([this] (Color px, i32, i32) -> Color {
                                    return px == color::WHITE ? color : px;
                                }),
                                cursor_x, cursor_y,
                                blend::overwrite
                            );
                            cursor_x += sym.width() + font.spacing;
                            break;
                        case SymbolType::Space:
                            cursor_x += sym.space.width;
                            break;
                    }
                }
                cursor_y += font.height + font.leading;
            }

            return ret;
        }

      public:
        MultilineText(StringView content, Font<T, Char> font, Color color = color::WHITE)
            : content(content), color(color), font(font)
        {
            i32 max_width = 0;
            i32 line_count = 0;
            for (auto line : lines()) {
                max_width = std::max(max_width, line_width(line));
                line_count += 1;
            }

            width_cache = max_width;
            height_cache = line_count == 0
                ? 0
                : line_count * font.height + (line_count - 1) * font.leading;
        }

        auto width() const -> i32 {
            return width_cache;
        }

        auto height() const -> i32 {
            return height_cache;
        }

        auto get(i32 x, i32 y) const -> Color {
            if (not cache) cache = redraw();
            return cache->get(x, y);
        }
    };

    template <typename T> MultilineText(char const*, Font<T, char>, Color = color::WHITE) -> MultilineText<T, std::string_view>;

    static_assert(SizedPlane<MultilineText<Image>>);

    template <Plane T, typename Str = std::string_view> struct WrappingText final {
        using StringView = Str;
        using Char = typename Str::value_type;

        StringView content;
        Color color;
        Font<T, Char> font;
        i32 width_limit;

      private:
        i32 width_cache;
        i32 height_cache;
        mutable std::optional<Image> cache;

        auto lines() const -> std::vector<StringView> {
            std::vector<StringView> result;

            auto it = content.begin();
            auto end = content.end();

            auto emit_line = [&](auto begin, auto end) {
                if (begin != end)
                    result.emplace_back(&*begin, usize(end - begin));
                else
                    result.emplace_back(StringView{});
            };

            auto line_begin = it;
            auto word_begin = it;

            i32 current_width = 0;

            while (it != end) {
                if (*it == '\n') {
                    emit_line(line_begin, it);
                    ++it;
                    line_begin = it;
                    word_begin = it;
                    current_width = 0;
                    continue;
                }

                // detect word boundary
                if (*it == ' ') {
                    ++it;
                    continue;
                }

                // start of a word
                word_begin = it;
                while (it != end && *it != ' ' && *it != '\n')
                    ++it;

                StringView word(&*word_begin, usize(it - word_begin));
                i32 word_width = line_width(word);
                i32 space_width = (line_begin != word_begin)
                    ? font.spacing + font.symbol(Char(' ')).width()
                    : 0;

                if (current_width + space_width + word_width > width_limit && current_width > 0) {
                    emit_line(line_begin, word_begin - 1); // trim trailing space
                    line_begin = word_begin;
                    current_width = word_width;
                } else {
                    current_width += space_width + word_width;
                }
            }

            emit_line(line_begin, it);
            return result;
        }

        auto line_width(StringView line) const -> i32 {
            if (line.empty()) return 0;
            i32 w = -font.spacing;
            for (Char c : line)
                w += font.symbol(c).width() + font.spacing;
            return w;
        }

        auto redraw() const -> Image {
            using SymbolType = typename Symbol<T>::Type;
            Image ret(width_cache, height_cache);

            i32 cursor_y = 0;
            for (auto line : lines()) {
                i32 cursor_x = 0;
                for (Char c : line) {
                    auto sym = font.symbol(c);
                    switch (sym.type) {
                        case SymbolType::Glyph:
                            ret | draw(
                                sym.glyph | draw::map([this] (Color px, i32, i32) -> Color {
                                    return px == color::WHITE ? color : px;
                                }),
                                cursor_x, cursor_y,
                                blend::overwrite
                            );
                            cursor_x += sym.width() + font.spacing;
                            break;
                        case SymbolType::Space:
                            cursor_x += sym.space.width;
                            break;
                    }
                }
                cursor_y += font.height + font.leading;
            }

            return ret;
        }

      public:
        WrappingText(StringView content, Font<T, Char> font, i32 width_limit, Color color = color::WHITE)
            : content(content), color(color), font(font), width_limit(width_limit)
        {
            i32 max_width = 0;
            i32 line_count = 0;
            for (auto line : lines()) {
                max_width = std::max(max_width, line_width(line));
                line_count += 1;
            }

            width_cache = max_width;
            height_cache = line_count == 0
                ? 0
                : line_count * font.height + (line_count - 1) * font.leading;
        }

        WrappingText(StringView content, Font<T, Char> font, Color color = color::WHITE)
            : WrappingText(content, font, std::numeric_limits<i32>::max(), color) {}

        auto width() const -> i32 {
            return width_cache;
        }

        auto height() const -> i32 {
            return height_cache;
        }

        auto get(i32 x, i32 y) const -> Color {
            if (not cache) cache = redraw();
            return cache->get(x, y);
        }
    };

    template <typename T> WrappingText(char const*, Font<T, char>, i32, Color = color::WHITE) -> WrappingText<T, std::string_view>;
    template <typename T> WrappingText(char const*, Font<T, char>, Color = color::WHITE) -> WrappingText<T, std::string_view>;

    static_assert(SizedPlane<MultilineText<Image>>);
}
