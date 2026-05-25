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
        VStack(VAlignment alignment, i32 spacing, T... planes) : inner(std::pair { planes, Position { 0, 0 } }...) {
            width_cache = std::max({ planes.width()... });
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

        VStack(T... planes) : VStack(VAlignment::Center, 0, planes...) {}
        VStack(VAlignment alignment, T... planes) : VStack(alignment, 0, planes...) {}
        VStack(i32 spacing, T... planes) : VStack(VAlignment::Center, spacing, planes...) {}

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
    template <SizedPlane... T> struct HStack final {
        std::tuple<T...> inner;
    };

    /// A powerful layout primitive combining layout at compile time.
    template <SizedPlane... T> struct ZStack final {
        std::tuple<T...> inner;
    };
}
