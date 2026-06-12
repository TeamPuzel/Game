#pragma once
#include <primitive>
#include <io>
#include <concepts>

namespace rng {
    template <typename Self> concept RandomGenerator = requires(Self& rng) {
        { rng.next() } -> std::same_as<u64>;
    };

    template <typename Self> concept GlobalRandomGenerator = RandomGenerator<Self> and
        std::is_default_constructible_v<Self>;

    class ThreadLocalIoGenerator final {
      public:
        ThreadLocalIoGenerator() = default;

        auto next() -> u64 {
            return Io::unsafe_get_threadlocal_io().get_random();
        }
    };

    using DefaultGenerator = ThreadLocalIoGenerator;

    template <typename T, RandomGenerator R> struct Random final {};

    template <typename T, RandomGenerator R = DefaultGenerator> constexpr auto random(
        R&& rng = DefaultGenerator()
    ) -> T {
        return Random<T, R>::random(std::forward<R>(rng));
    }

    template <typename T, RandomGenerator R = DefaultGenerator> constexpr auto random_to(
        T from, T to, R&& rng = DefaultGenerator()
    ) -> T {
        return Random<T, R>::random_to(from, to, std::forward<R>(rng));
    }

    template <typename T, RandomGenerator R = DefaultGenerator> constexpr auto random_until(
        T from, T until, R&& rng = DefaultGenerator()
    ) {
        return Random<T, R>::random_until(from, until, std::forward<R>(rng));
    }

    template <std::integral T, RandomGenerator R> struct Random<T, R> final {
        static constexpr auto random(R&& rng) -> T {
            return rng.next();
        }

        static constexpr auto random_to(T from, T to, R&& rng) -> T {
            auto delta = (to - from) + 1;
            if (delta < 1) return from;
            return from + rng.next() % delta;
        }

        static constexpr auto random_until(T from, T until, R&& rng = DefaultGenerator()) -> T {
            auto delta = until - from;
            if (delta < 1) return from;
            return from + rng.next() % delta;
        }
    };

    template <RandomGenerator R> struct Random<fixed, R> final {
        static constexpr auto random_to(fixed from, fixed to, R&& rng = DefaultGenerator()) -> fixed {
            i32 r_from = fixed::into_raw(from);
            i32 r_to = fixed::into_raw(to);

            if (r_from >= r_to) return from;

            u64 delta = static_cast<u64>(r_to - r_from) + 1;

            i32 result_raw = r_from + static_cast<i32>(rng.next() % delta);
            return fixed::from_raw(result_raw);
        }

        static constexpr auto random_until(fixed from, fixed until, R&& rng = DefaultGenerator()) -> fixed {
            i32 r_from = fixed::into_raw(from);
            i32 r_until = fixed::into_raw(until);

            if (r_from >= r_until) return from;

            u64 delta = static_cast<u64>(r_until - r_from);

            i32 result_raw = r_from + static_cast<i32>(rng.next() % delta);
            return fixed::from_raw(result_raw);
        }
    };

    class Xoshiro256StarStar final {
        u64 state[4];

        static constexpr auto rotl(u64 x, u64 k) -> u64 { return (x << k) | (x >> (64 - k)); }

      public:
        constexpr explicit Xoshiro256StarStar(u64 state1, u64 state2, u64 state3, u64 state4)
            : state { state1, state2, state3, state4 }
        {
            if ((state1 | state2 | state3 | state4) == 0) throw std::logic_error("the seed must not be equal to zero");
        }

        constexpr explicit Xoshiro256StarStar(u64 seed) {
            auto splitmix64 = [&seed]() -> u64 {
                u64 z = (seed += 0x9e3779b97f4a7c15);
                z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
                z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
                return z ^ (z >> 31);
            };

            state[0] = splitmix64();
            state[1] = splitmix64();
            state[2] = splitmix64();
            state[3] = splitmix64();
        }

        constexpr auto next() -> u64 {
            u64 result = rotl(state[1] * 5, 7) * 9;

            u64 t = state[1] << 17;

            state[2] ^= state[0];
            state[3] ^= state[1];
            state[1] ^= state[2];
            state[0] ^= state[3];

            state[2] ^= t;

            state[3] = rotl(state[3], 45);

            return result;
        }
    };
}
