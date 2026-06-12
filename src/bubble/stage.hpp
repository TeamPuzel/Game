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
#include <ranges>
#include "scene.hpp"
#include "object.hpp"
#include "class_loader.hpp"

namespace bubble {
    enum class GameMode {
        OnePlayer,
        TwoPlayer,
        TwoPlayerVersus
    };

    struct Tile final {
        u8 id      : 5 = 0;
        u8 current : 3 = 0;

        enum class Current : u8 { Up, Down, Left, Right, Solid };

        auto is_empty() const -> bool { return id == 0; }
    };

    enum class SensorMode : u8 {
        Floor,
        RightWall,
        Ceiling,
        LeftWall,
    };

    enum class SensorDirection : u8 { Down, Right, Up, Left };

    struct SensorResult final {
        i32 distance;

        constexpr auto hit(i32 back, i32 forward) const -> bool {
            return distance > -back and distance < forward;
        };

        constexpr auto hit(i32 both) const -> bool {
            return hit(both, both);
        }
    };

    static_assert(sizeof(Tile) == 1);

    /// A coroutine class representing the state of a loaded stage.
    ///
    /// This MUSN'T be final. Compilers are too smart and WILL devirtualize calls to a final stage, which will
    /// cause linkage errors because objects are compiled to separate shared libraries and rely on vtables!
    class Stage : public Scene {
        std::vector<Box<Object>> objects;
        std::unordered_set<Object*> removal_queue;
        usize tick = 0;
        Grid<Image> sheet;
        Box<SoundLibrary> sounds;

        mutable Image nes_target = Image(32 * 8, 30 * 8);
        mutable Image pre_transition_nes_target = Image(32 * 8, 30 * 8);

        static constexpr auto WIDTH          = 32;
        static constexpr auto HEIGHT         = 30;
        static constexpr auto START_TICK     = 7 * 60; // 7 now that the intro has a transition, was 9 when not.
        static constexpr auto GAME_END_DELAY = 60 * 2;

        std::array<Tile, WIDTH * HEIGHT> tiles;

        GameMode mode;
        u8 bub_lives = 2;
        u8 bob_lives = 2;
        u32 bub_score = 0;
        u32 bob_score = 0;

        u32 transition_timer = 0;

        bool should_check_for_game_end = false;
        i32 game_end_timer = 0;

        u8 stage_index = 1;
        bool editor_mode = false;
        u8 editor_id = 0;
        Tile::Current editor_current = Tile::Current::Up;
        usize editor_object = 0;
        Box<Object> editor_object_temp;

        mutable i32 viewport_offset_x = 0, viewport_offset_y = 0;

        enum class EditorPane {
            Tile,
            Current,
            Object
        } editor_pane = EditorPane::Tile;

        auto editor_pane_str() const -> std::string_view {
            switch (editor_pane) {
                case EditorPane::Tile:    return "Tile";
                case EditorPane::Current: return "Current";
                case EditorPane::Object:  return "Object";
            }
        }

        auto editor_current_str() const -> std::string_view {
            switch (editor_current) {
                case Tile::Current::Up:    return "Up";
                case Tile::Current::Down:  return "Down";
                case Tile::Current::Left:  return "Left";
                case Tile::Current::Right: return "Right";
                case Tile::Current::Solid: return "Solid";
            }
        }

        auto editor_object_str() const -> std::string_view {
            return object_registry.at(editor_object);
        }

        static constexpr u8 END_TILE_ID = 18;

        static constexpr auto object_registry = std::to_array<std::string_view>({
            "Player",
            "ZenChan",
            "Maita",
            "StartPoint"
        });

      public:
        /// Loading a class can take a while so it might be useful to cache them ahead of time.
        /// This task also performs a sanity assertion that all objects in the registry have a valid classname.
        static rt::DetachedTask preload_object_classes(Io& io) {
            co_await rt::enqueue();

            for (auto classname : object_registry) {
                auto descriptor = class_loader::load(io, classname);
                auto test_instance = descriptor.initializer(0, 0);
                if (not test_instance->isa(classname)) throw std::runtime_error(
                    std::format("mismatched classname, expected: {} found: {}", classname, test_instance->classname())
                );
            }
        }

        bool begin_transition(Io* io, rt::SoundStage* sound);
        bool begin_transition(Io& io, rt::SoundStage& sound) { return begin_transition(&io, &sound); }
        void begin_transition() { begin_transition(nullptr, nullptr); }

        auto player_bubbles_should_move() const -> bool {
            return tick >= START_TICK;
        }

        auto done_transitioning() const -> bool {
            return transition_timer == 0;
        }

        auto tile(i32 x, i32 y) -> Tile& { return tiles.at(x + y * WIDTH); }
        auto tile(i32 x, i32 y) const -> Tile const& { return tiles.at(x + y * WIDTH); }

        auto tile_at(i32 x, i32 y) -> std::optional<Tile> {
            if (x >= 0 and x < WIDTH * 8 and y >= 0 and y < HEIGHT * 8) {
                return tile(x / 8, y / 8);
            } else {
                return std::nullopt;
            }
        }

        auto solid_at(i32 x, i32 y) const -> bool {
            if (x >= 0 and x < WIDTH * 8 and y >= 0 and y < HEIGHT * 8) {
                return tile(x / 8, y / 8).id != 0;
            } else {
                return false;
            }
        }

        auto super_solid_at(i32 x, i32 y) const -> bool {
            if (x >= 0 and x < WIDTH * 8 and y >= 0 and y < HEIGHT * 8) {
                return tile(x / 8, y / 8).current == (u8) Tile::Current::Solid;
            } else {
                return false;
            }
        }

        auto solid_at(Object* relative_space, i32 x, i32 y) const -> bool {
            auto [ox, oy] = relative_space->pixel_pos();
            return solid_at(x + ox, y + oy);
        }

        auto super_solid_at(Object* relative_space, i32 x, i32 y) const -> bool {
            auto [ox, oy] = relative_space->pixel_pos();
            return super_solid_at(x + ox, y + oy);
        }

        auto get_sheet() const -> Grid<Ref<const Image>> {
            return sheet.ref();
        }

        auto get_sounds() -> SoundLibrary& {
            return *sounds;
        }

        auto in_editor_mode() const -> bool { return editor_mode; }

        void editor_object_reload(Io& io) {
            auto classname = editor_object_str();

            const auto descriptor = class_loader::load(io, classname);
            auto instance = descriptor.initializer(0, 0);
            editor_object_temp = std::move(instance);
        }

        void editor_object_clear() {
            editor_object_temp.erase();
        }

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

        void force_remove(Object* object) {
            remove(object);
            apply_removal_queue();
        }

        template <typename T> auto add(Box<T> object) -> T* {
            objects.emplace_back(std::move(object));
            return (T*) objects.back().raw();
        }

        template <typename T> auto take(T* object) -> Box<T> {
            auto it = std::ranges::find(objects, object, &Box<Object>::raw);

            if (it != objects.end()) {
                removal_queue.erase(object);
                return it->template cast<T>();
            } else {
                throw std::logic_error("object not in stage");
            }
        }

        auto objs() {
            return objects
                | std::views::transform([] (Box<Object>& box) -> Object* { return box.raw(); })
                | std::views::filter([] (auto ptr) -> bool { return ptr; })
                | std::ranges::to<std::vector>();
        }

        auto objs() const {
            return objects
                | std::views::transform([](Box<Object> const& box) -> Object const* { return box.raw(); })
                | std::views::filter([] (auto ptr) -> bool { return ptr; })
                | std::ranges::to<std::vector>();
        }

        virtual void lose_life_bub();
        virtual void lose_life_bob();
        virtual void check_for_game_end();
        virtual void award_points_bub(u32 points);
        virtual void award_points_bob(u32 points);

        Stage(Io& io, u8 index, Grid<Image> sheet, Box<SoundLibrary> sounds, GameMode mode, bool start_as_editor = false)
            : sheet(std::move(sheet)), sounds(std::move(sounds)), mode(mode), stage_index(index), editor_mode(start_as_editor) {}

        ~Stage() noexcept {
            // Make sure that we no longer hold on to objects, we can't destroy them after clearing the class loader.
            // i.e. Letting them be destroyed naturally is undefined.
            // TODO: The class loader should just be an instance, why is it global lol.
            // Also, throw if someone tries to make two class loaders, idk if all platforms allow loading
            // the same library in multiple instances?
            editor_object_temp.erase();
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
                        return not box or removal_queue.find(box.raw()) != removal_queue.end();
                    }
                ),
                objects.end()
            );

            removal_queue.clear();

            // Keep the temporary queue allocation under control as otherwise a lot of removals would permanently
            // waste memory and we can't have that can we.
            if (removal_queue.bucket_count() > 1024) removal_queue.rehash(0);
        }

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound) override;

        void draw(Io& io, rt::Input const& input, Ref<Image> target) const override {
            target | draw::clear();

            if (editor_mode) {
                target | draw::draw(
                    draw::FilledRectangle(target.width(), target.height(), Color::gray(15))
                        | draw::dither()
                );
            }

            constexpr i32 nes_width = 32 * 8, nes_height = 30 * 8; // 256x240
            viewport_offset_x = (target.width() - nes_width) / 2;
            viewport_offset_y = (target.height() - nes_height) / 2;

            input.with_offset(-viewport_offset_x, -viewport_offset_y, [&] (rt::Input const& input) {
                draw_viewport(io, input, nes_target | draw::as_ref());
            });

            target | draw::draw(
                nes_target | draw::as_ref(),
                viewport_offset_x,
                viewport_offset_y
            );

            if (editor_mode) {
                switch (editor_pane) {
                    case EditorPane::Tile: {
                        target | draw::draw(
                            draw::VStack(draw::VAlignment::Left, 4,
                                draw::Text(
                                    std::format("Editor::Tile / Stage {}", stage_index),
                                    font::pico()
                                ),
                                draw::Text(std::format("Selected Tile Id: {}", editor_id), font::pico()),
                                sheet.tile_ref(editor_id - 1, 36)
                                    | draw::scale(2)
                            ),
                            8, 8
                        );
                    } break;
                    case EditorPane::Current: {
                        target | draw::draw(
                            draw::VStack(draw::VAlignment::Left, 4,
                                draw::Text(
                                    std::format("Editor::Current / Stage {}", stage_index),
                                    font::pico()
                                ),
                                draw::Text(std::format("Selected Current: {}", editor_current_str()), font::pico()),
                                sheet.tile_ref((u8) editor_current, 35)
                                    | draw::scale(2)
                            ),
                            8, 8
                        );
                    } break;
                    case EditorPane::Object: {
                        auto [px, py] = editor_object_temp->pixel_pos();

                        target | draw::draw(
                            draw::VStack(draw::VAlignment::Left, 4,
                                draw::Text(
                                    std::format("Editor::Object / Stage {}", stage_index),
                                    font::pico()
                                ),
                                draw::Text(std::format("Selected Object: {}", editor_object_str()), font::pico()),
                                draw::Text(std::format("x: {} y: {}", px, py), font::pico())
                            ),
                            8, 8
                        );
                    } break;
                }

                if (auto mouse = input.mouse()) {
                    target | draw::draw(
                        mouse->left or mouse->right ? sheet.tile_ref(17, 0) : sheet.tile_ref(16, 0),
                        mouse->x - 1, mouse->y - 1
                    );
                }
            }
        }

        // Inelegant but serviceable indirection to constrain the NES viewport without rewriting the code.
        void draw_viewport(Io& io, rt::Input const& input, Ref<Image> target) const;

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
        static auto load(Io& io, u8 index, Grid<Image> sheet, Box<SoundLibrary> sounds, GameMode mode, bool start_as_editor = false) -> Box<Stage> {
            auto ret = Box<Stage>::make(io, index, std::move(sheet), std::move(sounds), mode, start_as_editor);
            ret->reload(io);
            return ret;
        }

        /// The sensor logic is implemented differently, given the significant CPU improvement since then.
        /// We just analyze the height map in pixel-space instead of the 1991 logic of regressing through tiles.
        /// At the end of the day objects just want the distance and they do not care if the entire range is consistent
        /// as they always considered only the consistent subrange within. I can't believe I spent days on
        /// this nonsense instead of just doing the obious thing.
        auto sense(i32 x, i32 y, SensorDirection direction) const -> SensorResult {
            i32 cx = x, cy = y;

            const i32 max_distance = 32;

            const auto regress = [&] {
                switch (direction) {
                    case SensorDirection::Down:  cy -= 1; break;
                    case SensorDirection::Right: cx -= 1; break;
                    case SensorDirection::Up:    cy += 1; break;
                    case SensorDirection::Left:  cx += 1; break;
                }
            };

            const auto extend = [&] {
                switch (direction) {
                    case SensorDirection::Down:  cy += 1; break;
                    case SensorDirection::Right: cx += 1; break;
                    case SensorDirection::Up:    cy -= 1; break;
                    case SensorDirection::Left:  cx -= 1; break;
                }
            };

            const auto distance = [&] {
                switch (direction) {
                    case SensorDirection::Down:  return cy - y;
                    case SensorDirection::Right: return cx - x;
                    case SensorDirection::Up:    return y - cy;
                    case SensorDirection::Left:  return x - cx;
                }
            };

            const auto within_limit = [&] {
                const auto d = distance();
                return -32 <= d and d <= 32;
            };

            // There are two cases, either we are within terrain and we need to regress or we are not inside of terrain
            // and we need to extend.
            if (solid_at(cx, cy)) {
                do regress(); while (solid_at(cx, cy) and within_limit());
            } else {
                do extend(); while (not solid_at(cx, cy) and within_limit());
                regress();
            }

            return { distance() };
        }

        [[clang::always_inline]]
        auto sense(Object const* relative_space, i32 x, i32 y, SensorDirection direction) const -> SensorResult {
            auto [ox, oy] = relative_space->pixel_pos();
            return sense(x + ox, y + oy, direction);
        }

        [[clang::always_inline]]
        auto sense(Object const* relative_space, SensorDirection direction) const -> SensorResult {
            return sense(relative_space, 0, 0, direction);
        }

        [[clang::always_inline]]
        static auto rotate(SensorDirection direction, u32 by_steps) noexcept -> SensorDirection {
            return (SensorDirection) (((u32) direction + by_steps) % 4);
        }

        [[clang::always_inline]]
        static auto rotate(i32 x, i32 y, i32 steps) noexcept -> std::pair<i32, i32> {
            steps = ((steps % 4) + 4) % 4;

            switch (steps) {
                case 0: return { +x, +y };
                case 1: return { +y, -x };
                case 2: return { -x, -y };
                case 3: return { -y, +x };
            }

            std::unreachable();
        }

        [[clang::always_inline]]
        auto sense(Object const* relative_space, i32 x, i32 y, SensorDirection direction, SensorMode mode) const -> SensorResult {
            const auto [rx, ry] = rotate(x, y, (i32) mode);
            return sense(relative_space, rx, ry, rotate(direction, (u32) mode));
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

            u32 object_count = 0;
            for (Box<Object> const& object : objects) if (object->is_serial()) object_count += 1;
            writer.u32(object_count);

            for (Box<Object> const& object : objects) if (object->is_serial()) {
                // Write classname (32 char cstring padded with null terminators)
                std::string_view name = object->classname();

                for (u32 i = 0; i < 32; i += 1) if (i < name.size()) writer.u8(name[i]); else writer.u8(0);

                writer.i32((i32) object->position.x);
                writer.i32((i32) object->position.y);

                std::vector<u8> userdata;
                BinaryWriter userdata_writer { std::back_inserter(userdata) };

                const auto descriptor = class_loader::load(io, object->classname());
                descriptor.serializer(object.raw(), userdata_writer);

                for (u32 i = 0; i < 128; i += 1) {
                    if (i < userdata.size()) writer.u8(userdata[i]);
                    else writer.u8(0);
                }
            }

            io.write_file(std::format("../res/stage/{}.stage", stage_index), result);
        }

        /// Answers true if a file was loaded, false if a new empty stage was created.
        template <const bool DEV = false> auto reload(Io& io) -> bool {
            objects.clear();

            if (auto level_file = io.try_read_file(
                std::format(DEV ? "../res/stage/{}.stage" : "res/{}.stage", stage_index)
            )) {
                BinaryReader reader { std::span(*level_file) };

                for (u32 i = 0; i < WIDTH * HEIGHT; i += 1) {
                    this->tiles[i] = std::bit_cast<Tile>(reader.u8());
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

                    this->objects.emplace_back(std::move(instance));
                }

                return true;
            } else {
                for (Tile& tile : tiles) tile = {};

                return false;
            }
        }

      public:
        virtual void unsafe_hot_reload_child(Box<Object>& object);

        void hot_reload(Io& io) override {
            class_loader::swap_registry();
            for (Box<Object>& object : objects) { // Intentionally mutable for swap
                if (not object->is_dynobject()) {
                    // We must clear out objects of unknown provenance since they are likely
                    // to come from a dynamic library we are about to drop.
                    // This is safe because we manually apply the removal queue afterwards.
                    remove(object.raw());
                } else {
                    auto descriptor = class_loader::load(io, object->classname());
                    auto replacement = descriptor.rebuilder(object.raw(), *this);

                    replacement->position = object->position;

                    std::swap(object, replacement);
                }
            }
            editor_object_reload(io);

            apply_removal_queue();
            class_loader::drop_old_object_classes();
        }
    };
}
