// Created by Lua (TeamPuzel) on May 26th 2025.
// Copyright (c) 2025 All rights reserved.
//
// A simple abstraction for switching scenes at runtime.
// Can be used for the stage, menu etc.
#pragma once
#include <primitive>
#include <draw>
#include <rt>

namespace sonic {
    using draw::Image;
    using draw::Ref;
    using draw::Color;

    /// A scene coroutine which can be run.
    ///
    /// Rendering is performed into an image rather than a dynamic target since this is
    /// significantly faster for obvous reasons and we control the runtime and can ensure it is one.
    struct Scene {
        /// Advances the state by 1/60 of a second.
        virtual void update(Io& io, rt::Input const& input) = 0;
        /// Called after update to mutate the render target.
        virtual void draw(
            Io& io, rt::Input const& input, Ref<Image> target, Ref<const Image> sheet, Ref<const Image> background
        ) const = 0;
        virtual ~Scene() noexcept {}

        virtual void hot_reload(Io& io) {}
    };
}
