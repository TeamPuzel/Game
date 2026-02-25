// Created by Lua (TeamPuzel) on May 26th 2025.
// Copyright (c) 2025 All rights reserved.
//
// This header defines everything about levels.
#pragma once
#include <primitive>
#include <rt>
#include <vector>
#include <unordered_set>
#include "scene.hpp"
#include "object.hpp"
#include "class_loader.hpp"

namespace bubble::temp {
    class SpinningPlayerThing final : public Object {
        static auto sprite_sheet() -> Grid<Ref<Image>> const& {
            // Since this returns a static value it's a rare case where it's good to return a reference as the calling
            // convention will simply yield the pointer rather than be forced to load the grid into registers.

            // Because objects are not created before the runtime is set up we can cheat and do some
            // impure IO for easy lazy initialization :)
            // Obviously though a runtime asset loader will be needed especially if it is to work with objects
            // loaded from shared libraries in the future, which can't use the threadlocal IO anyway.
            static auto source = draw::TgaImage::from(Io::unsafe_get_threadlocal_io().read_file("res/tiles.tga"))
                | draw::flatten<Image>();
            static auto grid = source
                | draw::as_ref()
                | draw::grid(16, 16);

            return grid;
        }

        // It's all static with no animation or anything, so this will do, no need for the Animator class yet.
        auto sprite() const -> draw::Slice<Ref<Image>> {
            return sprite_sheet().tile(0, 0);
        }

        math::angle angle;
        i32 radius;
        i32 speed;
        i32 center_x;
        i32 center_y;

      public:
        SpinningPlayerThing(i32 x, i32 y, math::angle angle = 0, i32 radius = 16, i32 speed = 2) : Object() {
            this->position.x = x;
            this->position.y = y - radius;
            this->center_x = x;
            this->center_y = y;
            this->radius = radius;
            this->speed = speed;
        }

        void update(Io& io, rt::Input const& input, Stage& stage) noexcept override {
            angle += i32(speed);

            position.x = center_x + radius * math::cos(angle);
            position.y = center_y + radius * math::sin(angle);
        }

        void draw(draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept override {
            // There isn't much logic to drawing this object I suppose.
            target | draw::draw(sprite());
        }
    };
}

namespace bubble {
    /// A coroutine class representing the state of a loaded stage.
    ///
    /// TODO: Most of the object logic can and probably should be made part of the supertype.
    class Stage final : public Scene {
        std::vector<Box<Object>> objects;
        std::unordered_set<Object*> removal_queue;
        Object* primary = nullptr;
        usize tick = 0;

      public:

        /// Schedules the object for removal at the end of the current update cycle.
        /// It remains valid until then.
        void remove(Object* object) noexcept {
            // TODO: This can throw, but it makes no sense to propagate to the object.
            // It would be ideal to implement a virtual handler in the scene itself for allocation failure,
            // but this is an internal queue. Doing this well would probably look similar to the iOS API where
            // the stage is notified that there isn't enough memory, with the scene assuming control over all
            // allocation. That would also be more efficient than using the default syscall allocator in the game loop.
            removal_queue.insert(object);
        }

        void add(Box<Object>&& object) noexcept {
            // TODO: This can throw, but it makes no sense to propagate to the object.
            objects.emplace_back(std::move(object));
        }

        Stage() {}

        ~Stage() noexcept {
            // Make sure that we no longer hold on to objects, we can't destroy them after clearing the class loader.
            // i.e. Letting them be destroyed naturally is undefined.
            // TODO: The class loader should just be an instance, why is it global lol.
            // Also, throw if someone tries to make two class loaders, idk if all platforms allow loading
            // the same library in multiple instances?
            objects.clear();
            class_loader::clear();
        }

        /// We need not remove inactive objects but we have no way of tracing this.
        /// This *is* optimizable if we manage sorting of objects sensibly and store active objects
        /// in the back of the object vector. The back because we wish to be able to reorder quickly without
        /// shifting the entire vector.
        ///
        /// For now though it should be fine even though it's not an efficient implementation at all.
        ///
        /// TODO: This horrible iterator mess can also be significantly cleaned up in C++20.
        void apply_removal_queue() {
            objects.erase(
                std::remove_if(objects.begin(), objects.end(),
                    [this] (Box<Object>& box) {
                        return removal_queue.find(box.raw()) != removal_queue.end();
                    }
                ),
                objects.end()
            );

            removal_queue.clear();

            // Keep the temporary queue allocation under control as otherwise a lot of removals would permanently
            // waste memory and we can't have that can we.
            if (removal_queue.bucket_count() > 1024) removal_queue.rehash(0);
        }

        void update(Io& io, rt::Input const& input) override {
            // The semantics are defined such that we handle collision first in sorting order on all active objects.
            // Updates follow in the same order but after all the collision. We iterate twice.
            // Also, the time complexity is silly here *but* it's just nearby objects so we should never hit
            // any actual scaling issues.
            //
            // There is a lifetime concern here. Remember that objects should be able to schedule themselves
            // for removal at the end of the update process, but must not be removed during the update
            // process itself.
            //
            // It is however unsound for any objects to ever reference each other directly at all. The reasons are many:
            // - Have fun serializing insane graphs.
            // - Lifetime and shared mutable state issues.
            //
            // For this reason, if any objects ever need to explicitly hold on to other objects between cycles,
            // a reference counting scheme shall be used:
            // - Smart, typed references can be requested from the stage.
            // - This smart (uniquely owned) reference object will internally hold on to the stage.
            // - When the smart reference object is destroyed it will automatically unregister itself.
            // - If any smart reference objects exists the object will be asked if it is okay to proceed with
            //   deletion.
            // - If it answers yes the object shall be destroyed and references invalidated.
            // - Unchecked access to an invalidated reference shall throw an exception which the stage will catch and
            //   the then immediately terminate the invalid object and any smart references to it.
            // This is the first and yet unimplemented draft of the approach.
            for (auto const& object : objects) object->update_tree(io, input, *this);

            apply_removal_queue();

            tick += 1;
        }

        void draw(Io& io, rt::Input const& input, Ref<Image> target) const override {
            // Let's just clear the screen here for now.
            target | draw::clear();

            // Render the game objects.
            // Objects more than a screen away from the edge are not drawn.
            //
            // TODO: Depth override with sorted drawing.
            const i32 buffer_x = target.width();
            const i32 buffer_y = target.height();

            constexpr i32 camera_x = 0;
            constexpr i32 camera_y = 0;

            // Visible rectangle in world coordinates.
            const i32 view_min_x = -camera_x - buffer_x;
            const i32 view_max_x = -camera_x + target.width() + buffer_x;
            const i32 view_min_y = -camera_y - buffer_y;
            const i32 view_max_y = -camera_y + target.height() + buffer_y;

            for (Box<Object> const& object : objects) {
                const auto [ox, oy] = object->pixel_pos();

                // TODO: Allow objects a force_draw override.
                if (ox >= view_min_x and ox <= view_max_x and oy >= view_min_y and oy <= view_max_y) {
                    // Align target with the object origin for relative drawing.
                    object->draw_tree(target | draw::shift(ox, oy), *this);
                }
            }
        }

        /// Loads a stage from a file using a provided object registry.
        /// Throws a runtime error if the object class does not exist.
        static auto load(Io& io, std::string_view filename) -> Box<Stage> {
            auto ret = Box<Stage>::make();

            // TODO: This temporarily hardcodes the scene.
            auto player = Box<temp::SpinningPlayerThing>::make(64, 64);
            player->add(Box<temp::SpinningPlayerThing>::make(0, 0));
            ret->add(std::move(player));

            return ret;
        }

        [[gnu::cold]] void hot_reload(Io& io) override {
            class_loader::swap_registry();
            for (Box<Object>& object : objects) { // Intentionally mutable for swap
                if (not object->is_dynobject()) {
                    // We must clear out objects of unknown provenance since they are likely
                    // to come from a dynamic library we are about to drop.
                    remove(object.raw());
                } else {
                    auto descriptor = class_loader::load(io, object->classname.view());
                    auto replacement = descriptor.rebuilder(*object);

                    replacement->position = object->position;
                    replacement->classname = object->classname;
                    if (object->classname == "Sonic") primary = replacement.raw();

                    std::swap(object, replacement);
                }
            }
            apply_removal_queue();
            class_loader::drop_old_object_classes();
        }
    };
}
