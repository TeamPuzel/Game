#pragma once
#include <primitive>
#include <utility>
#include <algorithm>
#include <vector>
#include <unordered_map>

namespace sound {
    template <typename Self> concept Sound = requires(Self const& self, usize index) {
        { self.get(index) } -> std::same_as<f32>;
    };

    template <typename Self> concept MutableSound = Sound<Self> and requires(Self& self, usize index, f32 value) {
        { self.set(index, value) };
    };

    template <typename Self> concept SizedSound = Sound<Self> and requires(Self const& self, usize index) {
        { self.count() } -> std::same_as<usize>;
    };

    template <typename Self, typename From> concept PrimitiveSound = SizedSound<From> and requires(From const& other) {
        { Self::flatten(other) } -> std::same_as<Self>;
    };

    template <typename Self> concept SizedMutableSound = SizedSound<Self> and MutableSound<Self>;
}

/// Performs forwarding adapter composition. Based on the design of std::ranges.
template <sound::Sound Self, typename Adapt> [[clang::always_inline]]
constexpr auto operator|(Self&& self, Adapt&& adapt) noexcept(noexcept(std::forward<Adapt>(adapt)(std::forward<Self>(self))))
    -> decltype(std::forward<Adapt>(adapt)(std::forward<Self>(self)))
{
    return std::forward<Adapt>(adapt)(std::forward<Self>(self));
}

template <sound::SizedSound L, sound::SizedSound R> constexpr auto operator==(L const& lhs, R const& rhs) -> bool {
    if (lhs.count() != rhs.count()) return false;
    for (usize i = 0; i < lhs.count(); i += 1) if (lhs.get(i) != rhs.get(i)) return false;
    return true;
}

/// Older C++ versions do not derive this from equality itself yet.
template <sound::SizedSound L, sound::SizedSound R> constexpr auto operator!=(L const& lhs, R const& rhs) -> bool {
    return !(lhs == rhs);
}

namespace sound {
    template <Sound T> class Slice final {
        T inner;
        usize offset, c;

      public:
        template <typename U> constexpr explicit Slice(U&& inner, usize offset, usize count) noexcept
            : inner(std::forward<U>(inner)), offset(offset), c(count) {}

        constexpr auto get(usize index) const noexcept(noexcept(inner.get(index))) -> f32 {
            return inner.get(index + offset);
        }

        constexpr void set(usize index, f32 value) noexcept(noexcept(inner.set(index, value))) {
            inner.set(index + offset, value);
        }

        constexpr auto count() const noexcept -> usize {
            return c;
        }

        constexpr auto resize_front(usize amount) const noexcept -> Slice {
            return Slice { inner, this->offset + amount, c > amount ? c - amount : 0 };
        }

        constexpr auto resize_back(usize amount) const noexcept -> Slice {
            return Slice { inner, this->offset, c > amount ? c - amount : 0 };
        }

        constexpr auto shift(usize offset) const noexcept -> Slice {
            return Slice { inner, this->offset + offset, c };
        }
    };

    template <typename T> Slice(T&&, usize, usize) -> Slice<T>;

    template <Sound T> struct Ref final {
        T& inner;

        constexpr explicit(false) Ref(T& inner) : inner(inner) {}

        constexpr auto count() const noexcept(noexcept(inner.count())) -> usize requires SizedSound<T> {
            return inner.count();
        }

        constexpr auto get(usize index) const noexcept(noexcept(inner.get(index))) -> f32 {
            return inner.get(index);
        }

        constexpr void set(usize index, f32 value) noexcept(noexcept(inner.set(index, value))) requires MutableSound<T> {
            inner.set(index, value);
        }
    };

    template <SizedSound T> struct Loop final {
        T inner;

        constexpr auto get(usize index)
            const noexcept(noexcept(inner.get(index)) and noexcept(inner.count())) -> f32
        {
            return inner.get(index % inner.count());
        }

        constexpr void set(usize index, f32 value)
            noexcept(noexcept(inner.set(index, value)) and noexcept(inner.count())) requires MutableSound<T>
        {
            inner.set(index % inner.count(), value);
        }
    };

    template <SizedSound T> struct LoopFrom final {
        T inner;
        usize start;

        constexpr auto get(usize index)
            const noexcept(noexcept(inner.get(index)) and noexcept(inner.count())) -> f32
        {
            const usize c = inner.count();

            // Pre-loop playback.
            if (index < c) return inner.get(index);
            // If start is out of bounds, output silence to prevent division by zero.
            if (c <= start) return .0f;

            // Loop playback.
            const usize loop_length = c - start;
            return inner.get(start + ((index - c) % loop_length));
        }

        constexpr void set(usize index, f32 value)
            noexcept(noexcept(inner.set(index, value)) and noexcept(inner.count())) requires MutableSound<T>
        {
            const usize c = inner.count();

            // Pre-loop playback.
            if (index < c) return inner.get(index);

            // Loop playback.
            const usize loop_length = c - start;
            return inner.set(start + ((index - c) % loop_length), value);
        }
    };

    template <Sound T> struct Volume final {
        T inner;
        f32 volume;

        constexpr auto get(usize index) const noexcept(noexcept(inner.get(index))) -> f32 {
            return inner.get(index) * volume;
        }

        constexpr void set(usize index, f32 value) noexcept(noexcept(inner.set(index, value))) requires MutableSound<T> {
            inner.set(index, value);
        }

        constexpr auto count() const noexcept(noexcept(inner.count())) -> usize requires SizedSound<T> {
            return inner.count();
        }
    };

    namespace adapt {
        struct Slice final {
            usize offset, count;

            template <Sound T> constexpr auto operator()(T&& inner) const noexcept -> sound::Slice<std::decay_t<T>> {
                return sound::Slice<std::decay_t<T>>(std::forward<T>(inner), offset, count);
            }
        };

        struct Shift final {
            usize offset;

            template <SizedSound T> constexpr auto operator()(T inner) const noexcept -> sound::Slice<T> {
                return sound::Slice<T>(inner, offset, inner.count());
            }
        };

        struct AsSlice final {
            template <Sound T> constexpr auto operator()(T inner) const noexcept -> sound::Slice<T> {
                return sound::Slice<T>(inner, 0, inner.count());
            }
        };

        struct AsRef final {
            template <Sound T> constexpr auto operator()(T& inner) const noexcept -> Ref<T> {
                return inner;
            }

            template <Sound T> constexpr auto operator()(T const& inner) const noexcept -> Ref<const T> {
                return inner;
            }
        };

        struct Move final {
            template <Sound T> constexpr auto operator()(T&& inner) const noexcept -> T&& {
                return std::move(inner);
            }
        };

        struct AsConst final {
            template <Sound T> constexpr auto operator()(T& inner) const noexcept -> T const& {
                return inner;
            }
        };

        template <Sound T> struct Flatten final {
            template <SizedSound U> constexpr T operator()(U const& self) const noexcept(noexcept(T::flatten(self)))
            requires
                PrimitiveSound<T, U>
            {
                return T::flatten(self);
            }
        };

        template <SizedSound S> struct Add final {
            S const& sound;

            constexpr Add(S const& sound) : sound(sound) {}

            template <typename T> constexpr T& operator()(T& self) const requires SizedMutableSound<T> {
                const auto count = std::min(self.count(), sound.count());

                for (usize i = 0; i < count; i += 1) {
                    self.set(i,
                        std::clamp(
                            self.get(i) + sound.get(i),
                            -1.f, 1.f
                        )
                    );
                }

                return self;
            }
        };

        struct Loop final {
            template <Sound T> constexpr auto operator()(T&& inner) const noexcept -> sound::Loop<std::decay_t<T>> {
                return sound::Loop<std::decay_t<T>>(std::forward<T>(inner));
            }
        };

        struct LoopFrom final {
            usize start;

            template <Sound T> constexpr auto operator()(T&& inner) const noexcept -> sound::LoopFrom<std::decay_t<T>> {
                return sound::LoopFrom<std::decay_t<T>>(std::forward<T>(inner), start);
            }
        };

        struct Trim final {
            usize front;
            usize back;

            template <Sound T> constexpr auto operator()(T&& inner) const noexcept -> sound::Slice<std::decay_t<T>> {
                return sound::Slice<std::decay_t<T>>(std::forward<T>(inner), front, inner.count() - front - back);
            }
        };
    }

    constexpr adapt::Slice slice(usize offset, usize count) noexcept {
        return adapt::Slice { offset, count };
    }

    constexpr adapt::Trim trim_front(usize count) noexcept {
        return adapt::Trim { count, 0 };
    }

    constexpr adapt::Trim trim_back(usize count) noexcept {
        return adapt::Trim { 0, count };
    }

    constexpr adapt::Trim trim(usize count) noexcept {
        return adapt::Trim { count / 2, count / 2 };
    }

    constexpr adapt::Shift shift(usize offset) noexcept {
        return adapt::Shift { offset };
    }

    constexpr adapt::AsSlice as_slice() noexcept {
        return adapt::AsSlice {};
    }

    constexpr adapt::AsRef as_ref() noexcept {
        return adapt::AsRef {};
    }

    constexpr adapt::Move move() noexcept {
        return adapt::Move {};
    }

    constexpr adapt::AsConst as_const() noexcept {
        return adapt::AsConst {};
    }

    template <typename T> constexpr adapt::Flatten<T> flatten() noexcept {
        return adapt::Flatten<T> {};
    }

    template <typename S> constexpr adapt::Add<S> add(S const& sound) {
        return adapt::Add { sound };
    }

    constexpr adapt::Loop loop() noexcept {
        return adapt::Loop {};
    }

    constexpr adapt::LoopFrom loop(usize start) noexcept {
        return adapt::LoopFrom { start };
    }
}

namespace sound {
    // Horrible virtual inheritance based existential containers.
    // Is this the best way to type erase sounds? Not really, but I don't feel like bothering today,
    // and realistically it really doesn't matter as the SoundStage won't be playing a lot of these at the same time.
    namespace any {
        class Sound {
          protected:
            void* data;

            auto (*deleter) (void*) -> void;
            auto (*getter) (void*, usize) -> f32;

          public:
            auto get(usize index) const -> f32 {
                return getter(data, index);
            }

            Sound(Sound const&) = delete;
            Sound& operator=(Sound const&) = delete;

            Sound(Sound&& other) noexcept
                : data(std::exchange(other.data, nullptr)), deleter(other.deleter), getter(other.getter) {}

            Sound& operator=(Sound&& other) noexcept {
                if (this != &other) {
                    if (data) deleter(data);
                    data = std::exchange(other.data, nullptr);
                    deleter = other.deleter;
                    getter = other.getter;
                }
                return *this;
            }

            template <sound::Sound S> explicit(false) Sound(S&& erasing) {
                using Decayed = std::decay_t<S>;

                data = new Decayed(std::forward<S>(erasing));

                deleter = [] (void* erased) {
                    delete static_cast<Decayed*>(erased);
                };

                getter = [] (void* erased, usize index) -> f32 {
                    return static_cast<Decayed*>(erased)->get(index);
                };
            }

            virtual ~Sound() {
                if (data) {
                    deleter(data);
                    data = nullptr;
                }
            }
        };

        class MutableSound : virtual public Sound {
          protected:
            auto (*setter) (void*, usize, f32) -> void;

          public:
            virtual void set(usize index, f32 value) {
                setter(data, index, value);
            }

            template <sound::MutableSound S> explicit(false) MutableSound(S&& erasing)
                : Sound(std::forward<S>(erasing))
            {
                setter = [] (void* erased, usize index, f32 value) {
                    static_cast<std::decay_t<S>*>(erased)->set(index, value);
                };
            }
        };

        class SizedSound : virtual public Sound {
          protected:
            auto (*counter) (void*) -> usize;

          public:
            virtual auto count() const -> usize {
                return counter(data);
            }

            template <sound::SizedSound S> explicit(false) SizedSound(S&& erasing)
                : Sound(std::forward<S>(erasing))
            {
                counter = [] (void* erased) -> usize {
                    return static_cast<std::decay_t<S>*>(erased)->count();
                };
            }
        };

        class SizedMutableSound : virtual public SizedSound, virtual public MutableSound {
          public:
            template <sound::SizedMutableSound S> explicit(false) SizedMutableSound(S&& erasing)
                : Sound(std::forward<S>(erasing))
            {
                this->counter = [] (void* erased) -> usize {
                    return static_cast<std::decay_t<S>*>(erased)->count();
                };

                this->setter = [] (void* erased, usize index, f32 value) {
                    static_cast<std::decay_t<S>*>(erased)->set(index, value);
                };
            }
        };
    }

    class Wave final {
        std::vector<f32> data;

        explicit Wave(std::vector<f32>&& data) : data(std::move(data)) {}

      public:
        Wave() {}

        Wave(Wave const&) = delete;
        auto operator=(Wave const&) -> Wave& = delete;

        Wave(Wave&& other) = default;
        auto operator=(Wave&& other) -> Wave& = default;

        template <typename F> Wave(usize count, F init) {
            data.reserve(count);
            data.resize(count);
            for (usize i = 0; i < count; i += 1) data[i] = init(i);
        }

        Wave(usize count, f32 value = .0f) : Wave(count, [value] (usize index) { return value; }) {}

        auto clone() const -> Wave {
            return Wave(count(), [this] (usize index) -> f32 {
                return this->get(index);
            });
        }

        void resize(usize count) {
            *this = Wave(count, [this] (usize index) -> f32 {
                return this->get(index);
            });
        }

        void clear(f32 value = .0f) {
            for (usize i = 0; i < data.size(); i += 1) {
                data[i] = value;
            }
        }

        auto count() const noexcept -> usize {
            return data.size();
        }

        auto get(usize index) const noexcept -> f32 {
            if (index >= 0 and index < count()) {
                return data[index];
            } else {
                return .0f;
            }
        }

        void set(usize index, f32 value) noexcept {
            if (index >= 0 and index < count()) {
                data[index] = value;
            }
        }

        auto raw() const noexcept -> f32 const* {
            return data.data();
        }

        auto raw() noexcept -> f32* {
            return data.data();
        }

        template <SizedSound U> static auto flatten(U const& other) -> Wave {
            return Wave(other.count(), [&] (usize index) -> f32 {
                return other.get(index);
            });
        }

        static auto from(std::vector<f32>&& data) -> Wave {
            return Wave(std::move(data));
        }
    };

    static_assert(SizedMutableSound<Wave>);

    template <const usize FREQUENCY = 48000> struct duration final {
        static auto seconds(usize s) -> usize {
            return s * FREQUENCY;
        }

        static auto milliseconds(usize s) -> usize {
            return s * FREQUENCY / 1000;
        }
    };

    class SoundStage final {
      public:
        using Id = usize;
        using Timestamp = u64;

        static constexpr auto FREQUENCY = 48000;
        using duration = sound::duration<FREQUENCY>;

        struct ActiveSound final {
            Box<any::Sound> exist;
            Timestamp start_time;
        };

        struct FinalizedSoundOutput final {
            f32 const* data;
            usize count;
        };

        SoundStage() = default;

        SoundStage(SoundStage const&) = delete;
        SoundStage(SoundStage&&) = default;
        SoundStage& operator=(SoundStage const&) = delete;
        SoundStage& operator=(SoundStage&&) = default;

      private:
        Wave buffer = Wave(4800);
        std::unordered_map<Id, ActiveSound> sounds;
        Id next_id = 1;
        Timestamp time_point = 0;

        void evaluate() {
            std::erase_if(sounds, [this] (auto const& pair) {
                auto const& sound = pair.second;

                if (const auto sized = dynamic_cast<any::SizedSound const*>(sound.exist.raw())) {
                    return time_point - sound.start_time >= sized->count();
                } else {
                    return false;
                }
            });

            for (auto const& [id, sound] : sounds) {
                usize local_time = time_point - sound.start_time;

                buffer | sound::add(
                    *sound.exist
                        | sound::as_ref()
                        | sound::slice(local_time, buffer.count())
                );
            }
        }

      public:
        template <Sound S> auto play(S&& sound) -> Id {
            Id id = next_id; next_id += 1;
            sounds.insert({
                id, ActiveSound {
                    .exist = Box<any::Sound>::make(std::forward<S>(sound)),
                    .start_time = time_point
                }
            });
            return id;
        }

        template <SizedSound S> auto play(S&& sound) -> Id {
            Id id = next_id; next_id += 1;
            sounds.insert({
                id, ActiveSound {
                    .exist = Box<any::SizedSound>::make(std::forward<S>(sound)),
                    .start_time = time_point
                }
            });
            return id;
        }

        void stop(Id id) {
            sounds.erase(id);
        }

        void stop(auto ids) {
            for (Id id : ids) stop(id);
        }

        void stop() {
            sounds.clear();
        }

        void advance_time_and_clear_buffer(u64 samples) {
            time_point += samples;
            buffer.clear();
        }

        auto finalize() -> FinalizedSoundOutput {
            evaluate(); return {
                .data = buffer.raw(),
                .count = buffer.count()
            };
        }
    };
}
