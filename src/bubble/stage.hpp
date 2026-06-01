// Created by Lua (TeamPuzel) on May 26th 2025.
// Copyright (c) 2025 All rights reserved.
//
// This header defines everything about levels.
#pragma once
#include <primitive>
#include <draw>
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

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound, Stage& stage) noexcept override {
            angle += i32(speed);

            position.x = center_x + radius * math::cos(angle);
            position.y = center_y + radius * math::sin(angle);
        }

        void draw(draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept override {
            // There isn't much logic to drawing this object I suppose.
            // target | draw::draw(sprite());

            target | draw::draw(
                draw::VStack(draw::VAlignment::Right, 16,
                    sprite(),
                    sprite() | draw::scale(2),
                    sprite() | draw::scale(3)
                )
            );
        }
    };
}

namespace bubble {
    class Editor;

    struct Tile final {
        bool is_empty : 1 = true;
        u8 id         : 7 = 0;

        enum Wind { Up, Down, Left, Right };
    };

    static_assert(sizeof(Tile) == 1);

    /// A coroutine class representing the state of a loaded stage.
    ///
    /// TODO: Most of the object logic can and probably should be made part of the supertype.
    class Stage final : public Scene {
        std::vector<Box<Object>> objects;
        std::unordered_set<Object*> removal_queue;
        usize tick = 0;
        std::string filename;
        Grid<Image> sheet;
        mutable Image nes_target = Image(32 * 8, 30 * 8);

        static constexpr auto WIDTH = 32;
        static constexpr auto HEIGHT = 30;

        std::array<Tile, WIDTH * HEIGHT> tiles;

      public:
        friend class Editor;

        auto tile(i32 x, i32 y) -> Tile& { return tiles.at(x + y * WIDTH); }
        auto tile(i32 x, i32 y) const -> Tile const& { return tiles.at(x + y * WIDTH); }

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

        Stage(Io& io, std::string filename, Grid<Image>&& sheet) : filename(filename), sheet(std::move(sheet)) {}

        Stage(Io& io, std::string filename) : Stage(io, filename,
            draw::TgaImage::from(io.read_file("res/tiles.tga"))
                | draw::flatten<Image>()
                | draw::grid(16, 16)
        ) {}

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

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound) override {
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
            for (auto const& object : objects) object->update(io, input, sound, *this);

            apply_removal_queue();

            tick += 1;
        }

        void draw(Io& io, rt::Input const& input, Ref<Image> target) const override {
            target | draw::clear();

            // NES resolution bounds
            constexpr i32 nes_width = 32 * 8, nes_height = 30 * 8; // 256x240

            // Figure out the greatest integer scale at which the NES resolution can fit.
            const i32 scale_x = target.width() / nes_width;
            const i32 scale_y = target.height() / nes_height;

            // Fall back to a scale of 1 if the screen is smaller than the NES base resolution
            // to prevent division by zero in MozaicPlane.
            const i32 scale = std::max(1, std::min(scale_x, scale_y));

            // Center the target area.
            const i32 scaled_width = nes_width * scale;
            const i32 scaled_height = nes_height * scale;
            const i32 offset_x = (target.width() - scaled_width) / 2;
            const i32 offset_y = (target.height() - scaled_height) / 2;

            draw_viewport(io, input, nes_target | draw::as_ref());

            // As inefficient as this entire process is (we kind of have to do this if we want virtual dispatch later)
            // the alternative is dispatch on individual pixels which is probably worse.
            //
            // Obligatory threading.
            target | draw::draw_threaded(
                nes_target
                    | draw::as_ref()
                    | draw::scale(scale),
                offset_x,
                offset_y
            );
        }

        // Inelegant but serviceable indirection to constrain the NES viewport without rewriting the code.
        void draw_viewport(Io& io, rt::Input const& input, Ref<Image> target) const {
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

            for (i32 x = 0; x < WIDTH; x += 1) {
                for (i32 y = 0; y < HEIGHT; y += 1) {
                    auto id = tile(x, y).id;

                    target | draw::draw(
                        sheet.tile_ref(-1 + id, 36),
                        x * 8, y * 8
                    );
                }
            }

            for (Box<Object> const& object : objects) {
                const auto [ox, oy] = object->pixel_pos();

                // TODO: Allow objects a force_draw override.
                if (ox >= view_min_x and ox <= view_max_x and oy >= view_min_y and oy <= view_max_y) {
                    // Align target with the object origin for relative drawing.
                    object->draw(target | draw::shift(ox, oy), *this);
                }
            }
        }

        /// Loads a stage from a little endian file.
        ///
        /// The stage format is very simple:
        /// - tile array (32 * 30 u8)
        /// - object count (u32)
        /// - object array (count Object)
        ///
        /// where an Object is:
        /// - classname (32 char cstring)
        /// - x (i32)
        /// - y (i32)
        /// - userdata (128 u8)
        static auto load(Io& io, std::string_view filename) -> Box<Stage> {
            auto ret = Box<Stage>::make(io, (std::string) filename);

            if (auto level_file = io.try_read_file(filename)) {
                BinaryReader reader { std::span(*level_file) };

                for (u32 i = 0; i < WIDTH * HEIGHT; i += 1) {
                    ret->tiles[i] = std::bit_cast<Tile>(reader.u8());
                }

                u32 object_count = reader.u32();

                for (u32 i = 0; i < object_count; i += 1) {
                    std::string classname = reader.cstr(32);
                    i32 x = reader.i32();
                    i32 y = reader.i32();

                    std::array<u8, 128> userdata;
                    for (u32 i = 0; i < 128; i += 1) userdata[i] = reader.u8();

                    BinaryReader userdata_reader { std::span(userdata) };

                    const auto descriptor = class_loader::load(io, classname);
                    auto instance = descriptor.deserializer(userdata_reader, x, y);
                    instance->classname = classname;
                    ret->objects.emplace_back(std::move(instance));
                }
            }

            return ret;
        }

      protected:
        /// Save the file to disk. This is a development feature invoked by the level editor.
        /// This function tries to write a file so it might throw an `Io::Error`.
        void store(Io& io) const {
            std::vector<u8> result;
            BinaryWriter writer { std::back_inserter(result) };

            for (auto tile : tiles) {
                writer.u8(std::bit_cast<u8>(tile));
            }

            writer.u32(0); // TODO: Write objects too.

            io.write_file(filename, result);
        }

      public:
        void hot_reload(Io& io) override {
            class_loader::swap_registry();
            for (Box<Object>& object : objects) { // Intentionally mutable for swap
                if (not object->is_dynobject()) {
                    // We must clear out objects of unknown provenance since they are likely
                    // to come from a dynamic library we are about to drop.
                    // This is safe because we manually apply the removal queue afterwards.
                    remove(object.raw());
                } else {
                    auto descriptor = class_loader::load(io, object->classname);
                    auto replacement = descriptor.rebuilder(*object);

                    replacement->position = object->position;
                    replacement->classname = object->classname;

                    std::swap(object, replacement);
                }
            }
            apply_removal_queue();
            class_loader::drop_old_object_classes();
        }
    };
}
