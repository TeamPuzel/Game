#pragma once
#include <primitive>
#include "image.hpp"

namespace draw::util {
    template <typename Tuple, typename F, usize... Is> void tuple_for_each_impl(
        Tuple& t, F&& f, std::index_sequence<Is...>
    ) {
        (f(static_cast<int>(Is), std::get<Is>(t)), ...);
    }

    template <typename Tuple, typename F> void tuple_for_each(Tuple& t, F&& f) {
        constexpr usize N = std::tuple_size_v<std::remove_reference_t<Tuple>>;
        tuple_for_each_impl(t, std::forward<F>(f), std::make_index_sequence<N>{});
    }
}

namespace draw {
    enum class VAlignment : u8 { Center, Left, Right };
    enum class HAlignment : u8 { Center, Top, Bottom };

    /// A powerful layout primitive combining layout at compile time.
    template <SizedPlane... T> class VStack final {
      public:
        struct Position final { i32 x, y; };

      private:
        std::tuple<std::pair<T, Position>...> inner;

        i32 width_cache;
        i32 height_cache;
        mutable std::optional<Image> cache;

        auto redraw() const -> Image {
            auto ret = Image(width_cache, height_cache);
            std::apply([&](auto const&... e) {
                (ret | ... | draw::draw(e.first, e.second.x, e.second.y, blend::overwrite));
            }, inner);
            return ret;
        }

      public:
        VStack(VAlignment alignment, i32 spacing, T... planes)
            : inner(std::pair { std::move(planes), Position { 0, 0 } }...)
        {
            if constexpr (sizeof...(T) > 0) {
                width_cache = std::apply([](auto const&... e) { return std::max({ e.first.width()... }); }, inner);
            } else {
                width_cache = 0;
            }

            height_cache = 0;

            i32 cursor_y = 0;
            util::tuple_for_each(inner, [&](usize i, auto& element) {
                auto& [plane, position] = element;

                height_cache += plane.height();
                if (i != sizeof...(T) - 1) height_cache += spacing;

                i32 offset; switch (alignment) {
                    case VAlignment::Center: offset = (width_cache - plane.width()) / 2; break;
                    case VAlignment::Left:   offset = 0;                                 break;
                    case VAlignment::Right:  offset = width_cache - plane.width();       break;
                }

                position.x = offset;
                position.y = cursor_y;

                cursor_y += plane.height() + spacing;
            });
        }

        VStack(T... planes) : VStack(VAlignment::Center, 0, std::move(planes)...) {}
        VStack(VAlignment alignment, T... planes) : VStack(alignment, 0, std::move(planes)...) {}
        VStack(i32 spacing, T... planes) : VStack(VAlignment::Center, spacing, std::move(planes)...) {}

        auto width() const -> i32 {
            return width_cache;
        }

        auto height() const -> i32 {
            return height_cache;
        }

        auto get(i32 x, i32 y) const -> Color {
            if (not cache) cache.emplace(redraw());
            return cache->get(x, y);
        }
    };

    /// A powerful layout primitive combining layout at compile time.
    template <SizedPlane... T> class HStack final {
        public:
        struct Position final { i32 x, y; };

        private:
        std::tuple<std::pair<T, Position>...> inner;

        i32 width_cache;
        i32 height_cache;
        mutable std::optional<Image> cache;

        auto redraw() const -> Image {
            auto ret = Image(width_cache, height_cache);
            std::apply([&](auto const&... e) {
                (ret | ... | draw::draw(e.first, e.second.x, e.second.y, blend::overwrite));
            }, inner);
            return ret;
        }

        public:
        HStack(HAlignment alignment, i32 spacing, T... planes)
            : inner(std::pair { std::move(planes), Position { 0, 0 } }...)
        {
            if constexpr (sizeof...(T) > 0) {
                // The stack's height is bounded by the tallest element.
                height_cache = std::apply([](auto const&... e) {
                    return std::max({ e.first.height()... });
                }, inner);
            } else {
                height_cache = 0;
            }

            width_cache = 0;

            i32 cursor_x = 0;
            util::tuple_for_each(inner, [&](usize i, auto& element) {
                auto& [plane, position] = element;

                width_cache += plane.width();
                if (i != sizeof...(T) - 1) width_cache += spacing;

                i32 offset; switch (alignment) {
                    case HAlignment::Center: offset = (height_cache - plane.height()) / 2; break;
                    case HAlignment::Top:    offset = 0;                                   break;
                    case HAlignment::Bottom: offset = height_cache - plane.height();       break;
                }

                position.x = cursor_x;
                position.y = offset;

                cursor_x += plane.width() + spacing;
            });
        }

        HStack(T... planes) : HStack(HAlignment::Center, 0, std::move(planes)...) {}
        HStack(HAlignment alignment, T... planes) : HStack(alignment, 0, std::move(planes)...) {}
        HStack(i32 spacing, T... planes) : HStack(HAlignment::Center, spacing, std::move(planes)...) {}

        auto width() const -> i32 {
            return width_cache;
        }

        auto height() const -> i32 {
            return height_cache;
        }

        auto get(i32 x, i32 y) const -> Color {
            if (not cache) cache.emplace(redraw());
            return cache->get(x, y);
        }
    };

    /// A powerful layout primitive combining layout at compile time.
    template <SizedPlane... T> struct ZStack final {
        std::tuple<T...> inner;
    };

    /// A dynamic layout primitive that creates a vertical stack from an iterable range.
    template <typename Range, typename Map> class VForEach final {
      public:
        using Item = std::ranges::range_value_t<Range>;
        using PlaneType = std::invoke_result_t<Map, Item>;

        static_assert(SizedPlane<PlaneType>, "VForEach mapping must return a SizedPlane");

      private:
        std::vector<Item> items;
        Map map;
        VAlignment alignment;
        i32 spacing;

        i32 width_cache;
        i32 height_cache;
        mutable std::optional<Image> cache;

        auto redraw() const -> Image {
            auto ret = Image(width_cache, height_cache);
            i32 cursor_y = 0;

            for (usize i = 0; i < items.size(); ++i) {
                auto plane = map(items[i]);
                i32 offset;
                switch (alignment) {
                    case VAlignment::Center: offset = (width_cache - plane.width()) / 2; break;
                    case VAlignment::Left:   offset = 0;                                 break;
                    case VAlignment::Right:  offset = width_cache - plane.width();       break;
                }

                ret | draw::draw(plane, offset, cursor_y, blend::overwrite);
                cursor_y += plane.height() + spacing;
            }
            return ret;
        }

      public:
        template <typename R> VForEach(VAlignment alignment, i32 spacing, R&& range, Map map)
            : items(std::ranges::begin(range), std::ranges::end(range)),
                map(std::move(map)), alignment(alignment), spacing(spacing)
        {
            width_cache = 0;
            height_cache = 0;

            for (usize i = 0; i < items.size(); ++i) {
                auto plane = this->map(items[i]);
                width_cache = std::max(width_cache, plane.width());
                height_cache += plane.height();
                if (i != items.size() - 1) height_cache += spacing;
            }
        }

        template <typename R> VForEach(R&& range, Map map)
            : VForEach(VAlignment::Center, 0, std::forward<R>(range), std::move(map)) {}
        template <typename R> VForEach(VAlignment alignment, R&& range, Map map)
            : VForEach(alignment, 0, std::forward<R>(range), std::move(map)) {}
        template <typename R> VForEach(i32 spacing, R&& range, Map map)
            : VForEach(VAlignment::Center, spacing, std::forward<R>(range), std::move(map)) {}

        auto width() const -> i32 {
            return width_cache;
        }

        auto height() const -> i32 {
            return height_cache;
        }

        auto get(i32 x, i32 y) const -> Color {
            if (not cache) cache.emplace(redraw());
            return cache->get(x, y);
        }
    };

    template <typename R, typename M> VForEach(VAlignment, i32, R&&, M) -> VForEach<std::decay_t<R>, std::decay_t<M>>;
    template <typename R, typename M> VForEach(R&&, M) -> VForEach<std::decay_t<R>, std::decay_t<M>>;
    template <typename R, typename M> VForEach(VAlignment, R&&, M) -> VForEach<std::decay_t<R>, std::decay_t<M>>;
    template <typename R, typename M> VForEach(i32, R&&, M) -> VForEach<std::decay_t<R>, std::decay_t<M>>;

    /// A dynamic layout primitive that creates a horizontal stack from an iterable range.
    template <typename Range, typename Map> class HForEach final {
      public:
        using Item = std::ranges::range_value_t<Range>;
        using PlaneType = std::invoke_result_t<Map, Item>;

        static_assert(SizedPlane<PlaneType>, "HForEach mapping must return a SizedPlane");

      private:
        std::vector<Item> items;
        Map map;
        HAlignment alignment;
        i32 spacing;

        i32 width_cache;
        i32 height_cache;
        mutable std::optional<Image> cache;

        auto redraw() const -> Image {
            auto ret = Image(width_cache, height_cache);
            i32 cursor_x = 0;

            for (usize i = 0; i < items.size(); ++i) {
                auto plane = map(items[i]);
                i32 offset;
                switch (alignment) {
                    case HAlignment::Center: offset = (height_cache - plane.height()) / 2; break;
                    case HAlignment::Top:    offset = 0;                                   break;
                    case HAlignment::Bottom: offset = height_cache - plane.height();       break;
                }

                ret | draw::draw(plane, cursor_x, offset, blend::overwrite);
                cursor_x += plane.width() + spacing;
            }
            return ret;
        }

      public:
        template <typename R> HForEach(HAlignment alignment, i32 spacing, R&& range, Map map)
            : items(std::ranges::begin(range), std::ranges::end(range)),
                map(std::move(map)), alignment(alignment), spacing(spacing)
        {
            width_cache = 0;
            height_cache = 0;

            for (usize i = 0; i < items.size(); ++i) {
                auto plane = this->map(items[i]);
                height_cache = std::max(height_cache, plane.height());
                width_cache += plane.width();
                if (i != items.size() - 1) width_cache += spacing;
            }
        }

        template <typename R> HForEach(R&& range, Map map)
            : HForEach(HAlignment::Center, 0, std::forward<R>(range), std::move(map)) {}
        template <typename R> HForEach(HAlignment alignment, R&& range, Map map)
            : HForEach(alignment, 0, std::forward<R>(range), std::move(map)) {}
        template <typename R> HForEach(i32 spacing, R&& range, Map map)
            : HForEach(HAlignment::Center, spacing, std::forward<R>(range), std::move(map)) {}

        auto width() const -> i32 {
            return width_cache;
        }

        auto height() const -> i32 {
            return height_cache;
        }

        auto get(i32 x, i32 y) const -> Color {
            if (not cache) cache.emplace(redraw());
            return cache->get(x, y);
        }
    };

    template <typename R, typename M> HForEach(HAlignment, i32, R&&, M) -> HForEach<std::decay_t<R>, std::decay_t<M>>;
    template <typename R, typename M> HForEach(R&&, M) -> HForEach<std::decay_t<R>, std::decay_t<M>>;
    template <typename R, typename M> HForEach(HAlignment, R&&, M) -> HForEach<std::decay_t<R>, std::decay_t<M>>;
    template <typename R, typename M> HForEach(i32, R&&, M) -> HForEach<std::decay_t<R>, std::decay_t<M>>;

    struct HSpacer final {
        i32 size;

        HSpacer(i32 size) : size(size) {}

        constexpr auto width() const noexcept -> i32 {
            return size;
        }

        constexpr auto height() const noexcept -> i32 {
            return 0;
        }

        constexpr auto get(i32 x, i32 y) const noexcept -> Color {
            return color::CLEAR;
        }
    };

    struct VSpacer final {
        i32 size;

        VSpacer(i32 size) : size(size) {}

        constexpr auto width() const noexcept -> i32 {
            return 0;
        }

        constexpr auto height() const noexcept -> i32 {
            return size;
        }

        constexpr auto get(i32 x, i32 y) const noexcept -> Color {
            return color::CLEAR;
        }
    };
}
