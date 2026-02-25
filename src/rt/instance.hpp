// Created by Lua (TeamPuzel) on May 29th 2025.
// Copyright (c) 2025 All rights reserved.
#pragma once
#include <primitive>
#include <io>
#include <draw>
#include <font>
#include <concepts>
#include <sstream>
#include <string>
#include <optional>
#include <atomic>
#include <unordered_set>
#include <iostream>
#include <chrono>
#include <numeric>
#include <SDL3/SDL.h>
#include "selector.hpp"

/// An implemenation of Io purely in terms of SDL3. This is very convenient because we don't need
/// to depend on the standard library or the operating system in SDL3 based projects.
class SdlIo final : public Io {
  public:
    class Error final : public Io::Error, public std::exception {
        std::string reason { SDL_GetError() };
      public:
        Error() {}

        const char * what() const noexcept override {
            return reason.c_str();
        }
    };

  private:
    auto perform_read_file(char const* path) -> std::vector<u8> override {
        usize count;
        const auto data = (u8*) SDL_LoadFile(path, &count);
        if (not data) throw Error();
        auto ret = std::vector(data, data + count);
        SDL_free(data);
        return ret;
    }

    void perform_write_file(char const* path, std::span<u8> data) override {
        SDL_SaveFile(path, data.data(), data.size());
    }

    /// A dynamic library loader in terms of SDL3.
    /// It offers little control but it happens to make the sensible choice of RTLD_NOW | RTLD_LOCAL which is
    /// exactly what we want and I will assume the semantics are preserved on other platforms or this would be a sad API.
    auto perform_open_library(char const* path) -> void* override {
        auto ret = SDL_LoadObject(path);
        if (not ret) throw Error();
        return ret;
    }

    void perform_close_library(void* library) override {
        SDL_UnloadObject((SDL_SharedObject*) library);
    }

    auto perform_load_symbol(void* library, char const* name) -> void* override {
        auto ret = SDL_LoadFunction((SDL_SharedObject*) library, name);
        if (not ret) throw Error();
        return (void*) ret;
    }
};


namespace rt {
    struct Mouse final {
        i32 x, y;
        bool left, right;
    };

    enum class Key : u32 {
        A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
        Backspace,
        Left, Right, Up, Down,
        Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
        Comma, Period, Slash,
        Backslash, Equals, Dash, BracketLeft, BracketRight,
        Semicolon, Quote,
        Space, Shift, Meta, Control, Option, Tab, Enter, Escape,
        F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,

        Plus = Equals,
        Minus = Dash,
    };

    struct KeyIterator final {
        class Impl final {
            u32 current;

            Impl(u32 at) : current(at) {}

          public:
            friend struct KeyIterator;

            auto operator==(Impl other) const -> bool {
                return current == other.current;
            }

            auto operator!=(Impl other) const -> bool {
                return current != other.current;
            }

            auto operator++() -> Impl& {
                ++current;
                return *this;
            }

            auto operator*() -> Key {
                return Key(current);
            }
        };

        auto begin() -> Impl {
            return Impl { 0 };
        }

        auto end() -> Impl {
            return Impl { u32(Key::F12) + 1 };
        }
    };

    constexpr auto all_keys() -> KeyIterator {
        return KeyIterator();
    }

    struct KeyState final {
        Key key;
        i32 pressed_for { 0 };

        static auto from(Key key) -> KeyState {
            return KeyState { key };
        }

        auto pressed() const -> KeyState {
            return KeyState { key, pressed_for + 1 };
        }

        friend constexpr auto operator==(rt::KeyState const& lhs, rt::KeyState const& rhs) noexcept -> bool {
            return lhs.key == rhs.key;
        }

        friend constexpr auto operator!=(rt::KeyState const& lhs, rt::KeyState const& rhs) noexcept -> bool {
            return !(lhs == rhs);
        }
    };
}

template <> struct std::hash<rt::KeyState> {
    std::size_t operator()(rt::KeyState const& state) const noexcept {
        return std::hash<u16>()(u16(state.key));
    }
};

namespace rt::detail {
    inline auto get_key(Key key) -> bool {
        const auto keys = SDL_GetKeyboardState(nullptr);
        switch (key) {
            case Key::A: return keys[SDL_SCANCODE_A];
            case Key::B: return keys[SDL_SCANCODE_B];
            case Key::C: return keys[SDL_SCANCODE_C];
            case Key::D: return keys[SDL_SCANCODE_D];
            case Key::E: return keys[SDL_SCANCODE_E];
            case Key::F: return keys[SDL_SCANCODE_F];
            case Key::G: return keys[SDL_SCANCODE_G];
            case Key::H: return keys[SDL_SCANCODE_H];
            case Key::I: return keys[SDL_SCANCODE_I];
            case Key::J: return keys[SDL_SCANCODE_J];
            case Key::K: return keys[SDL_SCANCODE_K];
            case Key::L: return keys[SDL_SCANCODE_L];
            case Key::M: return keys[SDL_SCANCODE_M];
            case Key::N: return keys[SDL_SCANCODE_N];
            case Key::O: return keys[SDL_SCANCODE_O];
            case Key::P: return keys[SDL_SCANCODE_P];
            case Key::Q: return keys[SDL_SCANCODE_Q];
            case Key::R: return keys[SDL_SCANCODE_R];
            case Key::S: return keys[SDL_SCANCODE_S];
            case Key::T: return keys[SDL_SCANCODE_T];
            case Key::U: return keys[SDL_SCANCODE_U];
            case Key::V: return keys[SDL_SCANCODE_V];
            case Key::W: return keys[SDL_SCANCODE_W];
            case Key::X: return keys[SDL_SCANCODE_X];
            case Key::Y: return keys[SDL_SCANCODE_Y];
            case Key::Z: return keys[SDL_SCANCODE_Z];

            case Key::Backspace: return keys[SDL_SCANCODE_BACKSPACE];
            case Key::Left:      return keys[SDL_SCANCODE_LEFT];
            case Key::Right:     return keys[SDL_SCANCODE_RIGHT];
            case Key::Up:        return keys[SDL_SCANCODE_UP];
            case Key::Down:      return keys[SDL_SCANCODE_DOWN];

            case Key::Num0: return keys[SDL_SCANCODE_0];
            case Key::Num1: return keys[SDL_SCANCODE_1];
            case Key::Num2: return keys[SDL_SCANCODE_2];
            case Key::Num3: return keys[SDL_SCANCODE_3];
            case Key::Num4: return keys[SDL_SCANCODE_4];
            case Key::Num5: return keys[SDL_SCANCODE_5];
            case Key::Num6: return keys[SDL_SCANCODE_6];
            case Key::Num7: return keys[SDL_SCANCODE_7];
            case Key::Num8: return keys[SDL_SCANCODE_8];
            case Key::Num9: return keys[SDL_SCANCODE_9];

            case Key::Comma:        return keys[SDL_SCANCODE_COMMA];
            case Key::Period:       return keys[SDL_SCANCODE_PERIOD];
            case Key::Slash:        return keys[SDL_SCANCODE_SLASH];
            case Key::Backslash:    return keys[SDL_SCANCODE_BACKSLASH];
            case Key::Equals:       return keys[SDL_SCANCODE_EQUALS];
            case Key::Dash:         return keys[SDL_SCANCODE_MINUS];
            case Key::BracketLeft:  return keys[SDL_SCANCODE_LEFTBRACKET];
            case Key::BracketRight: return keys[SDL_SCANCODE_RIGHTBRACKET];
            case Key::Semicolon:    return keys[SDL_SCANCODE_SEMICOLON];
            case Key::Quote:        return keys[SDL_SCANCODE_APOSTROPHE];
            case Key::Space:        return keys[SDL_SCANCODE_SPACE];
            case Key::Shift:        return keys[SDL_SCANCODE_LSHIFT] or keys[SDL_SCANCODE_RSHIFT];
            case Key::Meta:         return keys[SDL_SCANCODE_LGUI] or keys[SDL_SCANCODE_RGUI];
            case Key::Control:      return keys[SDL_SCANCODE_LCTRL] or keys[SDL_SCANCODE_RCTRL];
            case Key::Option:       return keys[SDL_SCANCODE_LALT] or keys[SDL_SCANCODE_RALT];
            case Key::Tab:          return keys[SDL_SCANCODE_TAB];
            case Key::Enter:        return keys[SDL_SCANCODE_RETURN];
            case Key::Escape:       return keys[SDL_SCANCODE_ESCAPE];

            case Key::F1:  return keys[SDL_SCANCODE_F1];
            case Key::F2:  return keys[SDL_SCANCODE_F2];
            case Key::F3:  return keys[SDL_SCANCODE_F3];
            case Key::F4:  return keys[SDL_SCANCODE_F4];
            case Key::F5:  return keys[SDL_SCANCODE_F5];
            case Key::F6:  return keys[SDL_SCANCODE_F6];
            case Key::F7:  return keys[SDL_SCANCODE_F7];
            case Key::F8:  return keys[SDL_SCANCODE_F8];
            case Key::F9:  return keys[SDL_SCANCODE_F9];
            case Key::F10: return keys[SDL_SCANCODE_F10];
            case Key::F11: return keys[SDL_SCANCODE_F11];
            case Key::F12: return keys[SDL_SCANCODE_F12];
        }
    }
}

namespace rt {
    class Input {
        std::optional<Mouse> mouse_state;
        std::unordered_set<KeyState> keys;
        usize poll_counter { 0 };

      protected:
        void press(Key key) {
            if (auto existing = keys.extract(KeyState::from(key))) {
                keys.insert(existing.value().pressed());
            } else {
                keys.insert(KeyState::from(key));
            }
        }

        void unpress(Key key) {
            (void) keys.extract(KeyState::from(key));
        }

        auto mouse() -> std::optional<Mouse>& {
            return mouse_state;
        }

        void advance_counter() {
            poll_counter += 1;
        }

      public:
        Input(Input const&) = delete;
        auto operator=(Input const&) -> Input& = delete;

        Input() {}

        auto mouse() const -> std::optional<Mouse> const& {
            return mouse_state;
        }

        auto key_pressed(Key key) const -> bool {
            if (auto k = keys.find(KeyState::from(key)); k != keys.end()) {
                return (*k).pressed_for == 0;
            } else {
                return false;
            }
        }

        auto key_held(Key key) const -> bool {
            return keys.find(KeyState::from(key)) != keys.end();
        }

        auto key_repeating(Key key, i32 delay, i32 interval) const -> bool {
            // If there is no matching key, it can't be pressed.
            const auto ki = keys.find(KeyState::from(key));
            if (ki == keys.end()) return false;
            const auto k = *ki;

            // If the key matched and it was just pressed it counts as the first press before the delay.
            if (k.pressed_for == 0) return true;
            // If we have not reached the delay the input doesn't count.
            if (k.pressed_for < delay) return false;

            // Now we can start repeating at the specified interval.
            const auto start = k.pressed_for - delay;
            return start % interval == 0; // start.isMultiple(of: interval)
        }

        auto counter() const -> usize {
            return poll_counter;
        }
    };

    class ManagedSdlInput final : public Input {
      public:
        void poll() {
            // TODO: Apply scale to mouse input. Unused in this game anyway so it's whatever.
            f32 x, y;
            auto btn = SDL_GetMouseState(&x, &y);
            i32 ix = i32(x), iy = i32(y);
            mouse() = Mouse {
                ix, iy,
                (btn & SDL_BUTTON_LMASK) ? true : false,
                (btn & SDL_BUTTON_RMASK) ? true : false,
            };

            for (const auto key : all_keys()) {
                if (detail::get_key(key)) {
                    this->press(key);
                } else {
                    this->unpress(key);
                }
            }

            advance_counter();
        }
    };

    /// Returns a concrete subtype of Input with a managed `poll()` interface.
    /// The exact type could change in the future.
    inline auto input() {
        return ManagedSdlInput {};
    }

    inline auto timestamp() -> f64 {
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration<f64, std::milli>(now).count();
    }

    class Timer final {
        f64 prev_time;

      public:
        Timer() : prev_time(timestamp()) {}

        auto elapsed() const -> f64 {
            return timestamp() - prev_time;
        }

        auto lap() -> f64 {
            const auto e = elapsed();
            prev_time = timestamp();
            return e;
        }
    };

    /// Accumulates time steps to guess the refresh rate since the SDL platform does not
    /// provide any way to query it otherwise.
    class HeuristicTimestampBasedRefreshRateLock final {
        std::vector<f64> acc;
        f64 previous_stamp { timestamp() };
        mutable f64 phase { 0.0 };

      public:
        enum CommonRate : u32 {
            Hz30  = 30,
            Hz60  = 60,
            Hz75  = 75,
            Hz90  = 90,
            Hz120 = 120,
            Hz144 = 144,
            Hz240 = 240,
            Hz360 = 360,
        };

        f64 estimated_millis;
        u32 estimated_hertz;
        std::optional<CommonRate> common_rate;

        void lap() {
            // SAFETY: Just an environment query.
            const f64 stamp = timestamp();
            const f64 elapsed = stamp - previous_stamp;

            acc.insert(acc.begin(), elapsed);

            previous_stamp = stamp;

            if (acc.size() > 120) acc.pop_back();

            const f64 average = std::accumulate(acc.begin(), acc.end(), 0.0)
                / (f64) acc.size();

            const auto common_rates = {
                Hz30, Hz60, Hz75, Hz90, Hz120, Hz144, Hz240, Hz360,
            };

            estimated_millis = average;
            estimated_hertz = u32(1000.0 / average);
            for (const auto rate : common_rates) {
                constexpr f64 ERROR = 0.5;
                const f64 closest = 1000.0 / (f64) (u32) rate;
                if (std::abs(average - closest) <= ERROR) {
                    common_rate = rate; break;
                } else {
                    common_rate = std::nullopt;
                }
            }
        }

        auto compatible_with_rate(u32 desired_rate) const -> bool {
            return (common_rate ? (u32) *common_rate : estimated_hertz) % desired_rate == 0;
        }

        /// Given a frame count and a target hertz makes an effort to invoke the provided callable
        /// at a consistent rate in sync with the display, ideally by alternating frames on which
        /// it attempts to draw when the rate is divisible or close to it.
        ///
        /// If the desired rate is incompatible and we can't evenly distribute frames it falls back on
        /// floating point accumulation which should maintain an erroneous but better than nothing sync.
        ///
        /// If the rate falls below the desired rate the callable will always be invoked since we are
        /// already failing to keep up and there is no way to provide a good experience anymore,
        /// which will slow down.
        ///
        /// Passing zero as the desired rate bypasses the lock.
        template <typename Fn> void sync(usize frame, u32 desired_rate, Fn f) const {
            const u32 actual_hertz = common_rate ? (u32) *common_rate : estimated_hertz;

            if (desired_rate == 0 or actual_hertz < desired_rate) {
                f();
            } else if (not compatible_with_rate(desired_rate)) {
                const f64 actual_rate = common_rate ? (f64) *common_rate : (1000.0 / estimated_millis);

                if (actual_rate < 1e-6) return; // avoid division by zero

                const f64 increment = (f64) desired_rate / actual_rate;
                phase += increment;

                if (phase >= 1.0) {
                    f();
                    phase -= 1.0;
                }
            } else {
                const u32 tick_frames = actual_hertz / desired_rate;

                if (tick_frames != 0 and frame % tick_frames == 0) {
                    f();
                }
            }
        }
    };

    /// Returns a concrete type implementing the refresh rate lock interface.
    /// The exact type could change in the future.
    inline auto refresh_rate_lock() {
        return HeuristicTimestampBasedRefreshRateLock {};
    }

    /// Defines a game runnable by a game executor. The default is `run(game)`.
    ///
    /// This does not use virtual dispatch because that would require the draw method
    /// to also be runtime polymorphic. This is opaque and prevents optimizing away the drawing
    /// for no benefit, as this type is never actually used in a runtime polymorphic way.
    /// It can simply be type erased and executed with the compiler having full insight.
    ///
    /// The catch is, without C++20 modules, this means the run implementation must be a template,
    /// so it is impossible for it to avoid exposing SDL includes, but that's an arbitrary
    /// issue caused by being forced to support old C++.
    template <typename Self>
    concept Instance = requires(Self const& self, Self& self_mut, draw::Ref<draw::Image> target, Io& io, Input const& input) {
        { self_mut.init(io) } -> std::same_as<void>;
        { self_mut.update(io, input) } -> std::same_as<void>;
        { self.draw(io, input, target) } -> std::same_as<void>;
    };

    /// An error raised while running the game using the default executor.
    struct RunError final {
        /// The cause of the error.
        enum class Reason {
            AlreadyRunning,
            CouldNotInitializeSdl,
            CouldNotCreateWindow,
            CouldNotCreateContext [[deprecated]],
            CouldNotCreateRenderer,
            CouldNotCreateTexture,
            CouldNotRenderTexture,
            CouldNotPresentToWindow,
        } reason;
        /// The description obtained from SDL, only present for SDL errors.
        std::optional<std::string> description { std::nullopt };
    };

    /// Runs a game in the environment.
    ///
    /// This method was moved from Game into an environment message.
    /// A game cannot run itself, it is run by the platform it's on
    /// and can be run in many ways, this is just one implementation.
    static void run(Instance auto& game, char const* title, i32 width, i32 height, i32 scale) {
        selector::unsafe_set_self(); // Bind the central selector runtime.

        static std::atomic<bool> is_running = false;

        if (is_running.load()) {
            throw RunError { RunError::Reason::AlreadyRunning };
        } else {
            is_running.store(true);
        }

        if (not SDL_Init(SDL_INIT_VIDEO)) {
            throw RunError {
                RunError::Reason::CouldNotInitializeSdl,
                SDL_GetError(),
            };
        }

        auto window = SDL_CreateWindow(
            title, width, height, SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY
        );
        if (not window) {
            throw RunError {
                RunError::Reason::CouldNotCreateWindow,
                SDL_GetError(),
            };
        }
        SDL_SetWindowMinimumSize(window, width, height);
        SDL_HideCursor();
        SDL_SyncWindow(window);

        // A simple SDL provided renderer.
        //
        // We are already using SDL and there is no need to bother setting up OpenGL just to
        // render to a texture and blit it to the screen. There's really no need for hardware
        // acceleration to begin with, the only API that should be used is a platform specific
        // synchronization primitive such as CADisplayLink on macOS. That would also properly
        // support variable refresh rates on such platforms which I am unsure SDL actually handles.
        auto renderer = SDL_CreateRenderer(window, nullptr);
        if (not renderer) {
            throw RunError {
                .reason = RunError::Reason::CouldNotCreateRenderer,
                .description = SDL_GetError(),
            };
        }
        // We can discard the error, it is inefficient not to use vsync but if a platform doesn't support it
        // we have much greater issues than rendering too fast and it's impressive we even got this far.
        // i.e. that kind of platform is probably better suited for a different runtime implementation.
        bool is_vsync = SDL_SetRenderVSync(renderer, 1);

        // Not const because we reallocate on window resize.
        //
        // Interesting note: stretching a pixel texture is the best way to get sharp pixels
        // with resizable windows on modern displays. You do very much notice blurry pixels,
        // you do not notice when the pixel ratio isn't quite square.
        //
        // Sure, on extremely low resolution displays this was once much more extreme, but no
        // operating system actually supports those anyhow. There is no reason to compromise here
        // as for whatever reason a lot of games seem to do.
        SDL_Texture* texture = nullptr;

        /// Reallocates the texture.
        auto const resize_texture = [&texture, renderer] (i32 w, i32 h) -> void {
            if (texture) {
                SDL_DestroyTexture(texture);
            }
            texture = SDL_CreateTexture(
                renderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING, w, h
            );
            if (not texture) {
                throw RunError {
                    RunError::Reason::CouldNotCreateTexture,
                    SDL_GetError(),
                };
            }
            SDL_SetTextureScaleMode(texture, SDL_SCALEMODE_NEAREST);
        };
        resize_texture(width / scale, height / scale);

        SdlIo io;
        Io::unsafe_push_threadlocal_io(&io);

        // Set up the proper, esoteric RAII cleanup in case of uncaught exceptions.
        // I don't feel like documenting this nonsense, but this is somehow the cleanest way to do this
        // until std::scope_exit is implemented >:(
        // Alternatively one could make all the code atrocious by wrapping SDL pointers in unique pointers
        // with deleters but that's rather unreadable.
        ScopeExit scope_exit = [=] {
            Io::unsafe_pop_threadlocal_io();
            SDL_DestroyTexture(texture);
            SDL_DestroyRenderer(renderer);
            SDL_DestroyWindow(window);
            SDL_Quit();
            is_running.store(false);
        };

        game.init(io);

        SDL_Event event;
        usize frame = 0;
        bool perf_overlay = true;
        bool heuristic_rate_lock = true;
        auto target = draw::Image(width / scale, height / scale);
        auto input = rt::input();
        auto rate = rt::refresh_rate_lock();

        const auto apply_window_size = [&] {
            // We are explicitly using the scaled window size and not the
            // GetWindowSizeInPixels(window:w:h:) call because we do actually want to scale
            // our own pixel scale by the display scale.
            // Effectively the game is always scaled twice on high density displays which will
            // give consistent sizing between devices.
            i32 w, h; SDL_GetWindowSizeInPixels(window, &w, &h);
            resize_texture(w / scale, h / scale);
            target.resize(w / scale, h / scale);
        };

        while (true) {
            while (SDL_PollEvent(&event)) {
                switch (event.type) {
                    case SDL_EVENT_QUIT: goto end;
                    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED: apply_window_size(); break;
                    default: break;
                }
            }

            // Ensure stable 60hz.
            rate.lap();

            const auto draw_perf_overlay = [&] {
                std::stringstream out;
                out << "Assumed rate: ";
                if (rate.common_rate) out << *rate.common_rate; else out << "Unknown";
                out << std::endl
                    << "Estimated rate: " << rate.estimated_hertz << std::endl
                    << "Average ms: " << rate.estimated_millis << std::endl
                    << "Vsync status: " << (is_vsync ? "Enabled" : "Disabled") << std::endl
                    << "Heuristic lock status: " << (heuristic_rate_lock ? "Enabled" : "Disabled") << std::endl
                    << "Scale: " << scale << 'x' << std::endl
                    << "Resolution: " << target.width() << 'x' << target.height() << std::endl;

                draw::MultilineText text { out.str(), font::mine() };
                target | draw::draw(text, target.width() - text.width() - 8, 8);
            };

            rate.sync(frame, heuristic_rate_lock ? 60 : 0, [&] {
                input.poll();

                { // Process runtime specific debug options.
                    if (input.key_pressed(Key::Num0)) {
                        is_vsync = !is_vsync;
                        SDL_SetRenderVSync(renderer, is_vsync);
                    }
                    if (input.key_pressed(Key::Num8)) heuristic_rate_lock = !heuristic_rate_lock;
                    if (input.key_pressed(Key::Num9)) perf_overlay = !perf_overlay;

                    if (bool p = input.key_pressed(Key::Plus), m = input.key_pressed(Key::Minus); p or m) {
                        if (p) scale = std::min(8, scale + 1);
                        if (m) scale = std::max(1, scale - 1);
                        target | draw::clear();
                        apply_window_size();
                    }

                    #ifdef _MSC_VER // Fullscreen button for a funny operating system.
                    if (input.key_pressed(Key::F1)) SDL_SetWindowFullscreen(window, true);
                    #endif
                }

                game.update(io, input);
            });
            game.draw(io, input, target);
            if (perf_overlay) draw_perf_overlay();

            SDL_RenderClear(renderer);

            SDL_UpdateTexture(texture, nullptr, target.raw(), u16(target.width() * sizeof(draw::Color)));

            if (not SDL_RenderTexture(renderer, texture, nullptr, nullptr)) {
                throw RunError {
                    RunError::Reason::CouldNotRenderTexture,
                    SDL_GetError(),
                };
            }

            if (not SDL_RenderPresent(renderer)) {
                throw RunError {
                    RunError::Reason::CouldNotPresentToWindow,
                    SDL_GetError(),
                };
            }

            frame += 1;
        }
    end:
    }

    /// Runs a game in the environment.
    ///
    /// This method was moved from Game into an environment message.
    /// A game cannot run itself, it is run by the platform it's on
    /// and can be run in many ways, this is just one implementation.
    ///
    /// This overload uses the default window size of 800x600.
    inline void run(Instance auto& game, char const* title, i32 scale = 1) {
        run(game, title, 800, 600, scale);
    }

    template <typename F> void run_io(F fn) requires requires (F fn, Io& io) { fn(io); } {
        selector::unsafe_set_self(); // Initialize the central selector runtime.

        SDL_Init(0);
        SdlIo io;
        Io::unsafe_push_threadlocal_io(&io);

        ScopeExit scope_exit = [] {
            Io::unsafe_pop_threadlocal_io();
            SDL_Quit();
        };

        fn(io);
    }
}
