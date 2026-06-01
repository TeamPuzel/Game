// Created by Lua (TeamPuzel) on May 26th 2025.
// Copyright (c) 2025 All rights reserved.
//
// A simple abstraction for switching scenes at runtime.
// Can be used for the stage, menu etc.
#pragma once
#include <primitive>
#include <draw>
#include <rt>

namespace bubble {
    using draw::Image;
    using draw::Ref;
    using draw::Grid;
    using draw::Color;
    using draw::ScaledPlane;
    using draw::MozaicPlane;

    /// A scene coroutine which can be run.
    ///
    /// Rendering is performed into an image rather than a dynamic target since this is
    /// significantly faster for obvous reasons and we control the runtime and can ensure it is one.
    class Scene {
        static Box<Scene>* root_ptr;
        static std::optional<Box<Scene>> scheduled_transition;

      protected:
        static auto transition(Box<Scene>&& scene) {
            if (scheduled_transition) throw std::runtime_error("double transition");
            scheduled_transition = std::move(scene);
        }

      public:
        /// Advances the state by 1/60 of a second.
        virtual void update(Io& io, rt::Input const& input, rt::SoundStage& sound) = 0;
        /// Called after update to mutate the render target.
        virtual void draw(Io& io, rt::Input const& input, Ref<Image> target) const = 0;

        virtual ~Scene() noexcept {}

        virtual void hot_reload(Io& io) {}

        // The root pointer must be pinned in memory for as long as scenes exist.
        static void unsafe_set_root_ptr(Box<Scene>* root_ptr) {
            Scene::root_ptr = root_ptr;
        }

        static void unsafe_apply_transition() {
            if (scheduled_transition) {
                *root_ptr = std::move(*scheduled_transition);
                scheduled_transition = std::nullopt;
            }
        }
    };
}
