// Created by Lua (TeamPuzel) on August 12th 2025.
// Copyright (c) 2025 All rights reserved.
//
// A simple convenience class for stateful object animation.
#pragma once
#include <primitive>
#include <type_traits>

namespace bubble {
    /// A simple animation system which scrolls through and loops in a range.
    ///
    /// The speed for the next iteration can be adjusted as the current iteration plays which can
    /// accurately recreate the behavior of animations from the classic sonic games.
    template <typename T, const T DEFAULT = T(0)> class Animator final {
        static_assert(std::is_enum<T>::value);
        T current { DEFAULT };

        /// The current frame of the animation.
        u32 frame { 0 };
        /// A counter counting down frames.
        u32 counter { 0 };
        /// How many frames are in this animation.
        u32 count { 1 };
        /// The frame the loop jumps back to.
        u32 loop { 0 };
        /// How many extra frames does it take to move on to the next frame.
        u32 speed { 0 };

      public:
        constexpr Animator() noexcept {}

        constexpr auto which() const noexcept -> T {
            return current;
        }

        constexpr auto is(T anim) const noexcept -> bool {
            return anim == current;
        }

        constexpr auto at() const noexcept -> u32 {
            return frame;
        }

        constexpr auto play(T anim, u32 count = 1, u32 speed = 0, u32 loop = 0) noexcept -> bool {
            if (anim == current) return false;

            this->current = anim;
            this->frame = 0;
            this->counter = speed;
            this->count = std::max(1u, count);
            this->loop = loop;
            this->speed = speed;

            return true;
        }

        constexpr void set_speed(u32 step) noexcept {
            this->speed = step;
        }

        /// Step through the animation.
        constexpr void update() noexcept {
            if (counter == 0) {
                frame += 1;
                counter = speed;
            } else {
                counter -= 1;
            }

            if (frame >= count) frame = loop;
        }
    };
}
