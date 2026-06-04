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
#include "scoreboard.hpp"

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

        static constexpr auto WIDTH          = 32;
        static constexpr auto HEIGHT         = 30;
        static constexpr auto START_TICK     = 9 * 60; // 7-8 if the intro has a transition, 9 if not.
        static constexpr auto GAME_END_DELAY = 60 * 2;

        std::array<Tile, WIDTH * HEIGHT> tiles;

        GameMode mode;
        u8 bub_lives = 2;
        u8 bob_lives = 2;
        u32 bub_score = 0;
        u32 bob_score = 0;

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
        /// Loading a class can take a while so it is might be useful to cache them ahead of time.
        static rt::DetachedTask preload_object_classes(Io& io) {
            co_await rt::enqueue();
            for (auto classname : object_registry) class_loader::load(io, classname);
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
            instance->classname = classname;
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

        void add(Box<Object> object) {
            objects.emplace_back(std::move(object));
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
            return objects | std::views::transform([](Box<Object>& box) -> Object* {
                return box.raw();
            });
        }

        auto objs() const {
            return objects | std::views::transform([](Box<Object> const& box) -> Object const* {
                return box.raw();
            });
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

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound) override {
            if (editor_mode) tick = START_TICK;

            if (input.key_pressed(rt::Key::Tab)) editor_mode = not editor_mode;

            if (editor_mode) {
                if (input.key_pressed(rt::Key::Num1)) editor_pane = EditorPane::Tile;
                if (input.key_pressed(rt::Key::Num2)) editor_pane = EditorPane::Current;
                if (input.key_pressed(rt::Key::Num3)) {
                    editor_pane = EditorPane::Object;
                    editor_object_reload(io);
                }

                switch (editor_pane) {
                    case EditorPane::Tile:
                        if (input.key_repeating(rt::Key::Q)) editor_id = std::max(0, i32(editor_id) - 1);
                        if (input.key_repeating(rt::Key::E)) editor_id += 1;

                        editor_id = std::clamp<u8>(editor_id, 0, END_TILE_ID);

                        editor_object_clear();
                        break;
                    case EditorPane::Current:
                        editor_object_clear();
                        break;
                    case EditorPane::Object:
                        if (input.key_repeating(rt::Key::Q)) editor_object = std::max(0, i32(editor_object) - 1);
                        if (input.key_repeating(rt::Key::E)) editor_object += 1;

                        editor_object = std::clamp<u8>(editor_object, 0, object_registry.size() - 1);

                        if (input.key_repeating(rt::Key::Q) or input.key_repeating(rt::Key::E)) {
                            editor_object_reload(io);
                        }

                        if (input.key_pressed(rt::Key::F)) editor_object_temp->flip();
                        if (input.key_pressed(rt::Key::G)) editor_object_temp->alternate();

                        break;
                }

                if (input.key_pressed(rt::Key::BracketLeft)) {
                    stage_index = std::max(1, stage_index - 1);
                    reload<true>(io);
                }
                if (input.key_pressed(rt::Key::BracketRight)) {
                    stage_index += 1;
                    reload<true>(io);
                }

                if (input.key_pressed(rt::Key::Up))    editor_current = Tile::Current::Up;
                if (input.key_pressed(rt::Key::Down))  editor_current = Tile::Current::Down;
                if (input.key_pressed(rt::Key::Left))  editor_current = Tile::Current::Left;
                if (input.key_pressed(rt::Key::Right)) editor_current = Tile::Current::Right;
                if (input.key_pressed(rt::Key::Slash)) editor_current = Tile::Current::Solid;

                input.with_offset(-viewport_offset_x, -viewport_offset_y, [&] (rt::Input const& input) {
                    if (auto mouse = input.mouse()) {
                        if (editor_object_temp) {
                            editor_object_temp->position.x = mouse->x;
                            editor_object_temp->position.y = mouse->y;
                        }

                        auto mx = mouse->x - 1, my = mouse->y - 1;

                        if (mx >= 0 and mx < WIDTH * 8 and my >= 0 and my < HEIGHT * 8) {
                            // Snap to the nearest tile.
                            const i32 snapped_x = mx / 8;
                            const i32 snapped_y = my / 8;

                            switch (editor_pane) {
                                case EditorPane::Tile: {
                                    if (mouse->left) tile(snapped_x, snapped_y).id = editor_id;
                                    if (mouse->right) tile(snapped_x, snapped_y).id = 0;
                                } break;
                                case EditorPane::Current: {
                                    if (mouse->left) tile(snapped_x, snapped_y).current = (u8) editor_current;
                                } break;
                                case EditorPane::Object: {
                                    if (input.left_click()) {
                                        add(std::move(editor_object_temp));
                                        editor_object_reload(io);
                                    }

                                    if (input.right_click()) {
                                        Object* closest_object = nullptr;
                                        i32 min_dist_sq = 16 * 16;

                                        for (Box<Object> const& obj : objects) {
                                            auto [ox, oy] = obj->pixel_pos();

                                            i32 dx = ox - mouse->x;
                                            i32 dy = oy - mouse->y;
                                            i32 dist_sq = dx * dx + dy * dy;

                                            if (dist_sq < min_dist_sq) {
                                                min_dist_sq = dist_sq;
                                                closest_object = obj.raw();
                                            }
                                        }

                                        if (closest_object) {
                                            force_remove(closest_object);
                                        }
                                    }
                                } break;
                            }
                        }
                    }
                });

                if (input.key_pressed(rt::Key::S)) store(io);

                if (input.key_pressed(rt::Key::P)) {
                    objects.clear();
                    for (Tile& tile : tiles) tile = {};
                }

                if (input.key_pressed(rt::Key::R)) reload<true>(io);

                return; // Stop updates while editing.
            }

            if (tick == 0) sound.play(
                sounds->get("music::gameplay").clone()
                    | sound::trim_back(sound::duration<>::seconds(2) - sound::duration<>::milliseconds(500))
                    | sound::loop(sound::duration<>::seconds(9))
            );

            if (tick >= START_TICK) {
                if (should_check_for_game_end) {
                    check_for_game_end();
                    should_check_for_game_end = false;
                }

                if (game_end_timer) game_end_timer -= 1;

                if (game_end_timer == 1) {
                    std::queue<ScoreBoard::PendingScore> queue;

                    queue.push({ .character = ScoreBoard::Character::Bub, .score = bub_score });
                    if (mode == GameMode::TwoPlayer)
                        queue.push({ .character = ScoreBoard::Character::Bob, .score = bob_score });

                    sound.stop();
                    transition(Box<bubble::ScoreBoard>::make(
                        io, std::move(sheet), std::move(sounds), std::move(queue))
                    );
                }

                // We can add more objects during an object update so we can't use a range loop as that
                // could sometimes invalidate the iterator if the vector has to resize.
                for (usize i = 0; i < objects.size(); i += 1) if (objects[i]) {
                    objects[i]->update(io, input, sound, *this);
                }
            }

            apply_removal_queue();

            tick += 1;
        }

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
        void draw_viewport(Io& io, rt::Input const& input, Ref<Image> target) const {
            target | draw::clear(draw::color::BLACK);

            if (tick < START_TICK) {
                auto intro = draw::MultilineText(
                    "Now it is the beginning of\n"
                    "a fantastic story! Let us\n"
                    "make a journey to\n"
                    "the cave of monsters!\n\n"
                    "Good luck!",
                    font::pod(),
                    draw::VAlignment::Center
                );

                target | draw::draw(intro, draw::Origin::Center);

                return;
            }

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

            if (editor_mode) {
                target | draw::draw(
                    draw::FilledRectangle(WIDTH, HEIGHT, Color::gray(18))
                        | draw::dither()
                        | draw::scale(8)
                );
            }

            for (i32 x = 0; x < WIDTH; x += 1) {
                for (i32 y = 0; y < HEIGHT; y += 1) {
                    auto tile = Stage::tile(x, y);

                    if (editor_mode and editor_pane == EditorPane::Current) {
                        target | draw::draw(
                            sheet.tile_ref(tile.id - 1, 36)
                                | draw::map([] (Color c, i32 x, i32 y) -> Color {
                                    if (x < 8 and y < 8) {
                                        return c.a == 0 ? c : Color::gray(128);
                                    } else {
                                        return c.a == 0 ? c : Color::gray(64);
                                    }
                                })
                                | draw::dither(),
                            x * 8, y * 8
                        );
                    } else if (editor_mode) {
                        target | draw::draw(
                            sheet.tile_ref(tile.id - 1, 36)
                                | draw::resize_right(-8)
                                | draw::resize_bottom(-8),
                            x * 8, y * 8
                        );
                    } else {
                        target | draw::draw(
                            sheet.tile_ref(tile.id - 1, 36),
                            x * 8, y * 8
                        );
                    }
                }
            }

            if (editor_mode and editor_pane == EditorPane::Current) {
                for (i32 x = 0; x < WIDTH; x += 1) {
                    for (i32 y = 0; y < HEIGHT; y += 1) {
                        auto tile = Stage::tile(x, y);

                        target | draw::draw(
                            sheet.tile_ref(tile.current, 35)
                                | draw::map([tile] (Color c) -> Color {
                                    if (c.a == 0) return c;

                                    switch ((Tile::Current) tile.current) {
                                        case Tile::Current::Up:    return draw::color::pico::WHITE;
                                        case Tile::Current::Down:  return draw::color::pico::LIGHT_BLUE;
                                        case Tile::Current::Left:  return draw::color::pico::RED;
                                        case Tile::Current::Right: return draw::color::pico::GREEN;
                                        case Tile::Current::Solid: return draw::color::pico::YELLOW;
                                    }
                                }),
                            x * 8, y * 8
                        );
                    }
                }
            }

            if (not editor_mode or editor_mode and editor_pane == EditorPane::Object) {
                for (Box<Object> const& object : objects) {
                    const auto [ox, oy] = object->pixel_pos();

                    // TODO: Allow objects a force_draw override.
                    if (ox >= view_min_x and ox <= view_max_x and oy >= view_min_y and oy <= view_max_y) {
                        // Align target with the object origin for relative drawing.
                        object->draw(io, target | draw::shift(ox, oy), *this);
                    }
                }
            }

            if (editor_mode) {
                if (auto mouse = input.mouse()) {
                    if (editor_pane == EditorPane::Object) {
                        const auto [ox, oy] = editor_object_temp->pixel_pos();
                        editor_object_temp->draw(io, target | draw::shift(ox, oy), *this);
                    }

                    auto mx = mouse->x - 1, my = mouse->y - 1;

                    if (mx >= 0 and mx < WIDTH * 8 and my >= 0 and my < HEIGHT * 8) {
                        // Snap to the nearest tile.
                        const i32 snapped_x = (mx / 8) * 8;
                        const i32 snapped_y = (my / 8) * 8;

                        switch (editor_pane) {
                            case EditorPane::Tile: {
                                target | draw::draw(
                                    sheet.tile_ref(editor_id - 1, 36)
                                        | draw::resize_right(-8)
                                        | draw::resize_bottom(-8)
                                        | draw::dither(),
                                    snapped_x, snapped_y
                                );
                            } break;
                            case EditorPane::Current: {

                            } break;
                            case EditorPane::Object: {

                            } break;
                        }
                    }
                }
            }

            if (not editor_mode) { // HUD.
                auto above_space_target = (target | draw::as_slice())
                    .resize_bottom(-(HEIGHT - 4) * 8);

                // above_space_target | draw::clear(draw::color::BLACK);

                // Fit the HUD area and pad edges.
                auto hud_target = above_space_target.resize(-4);

                auto total_seconds = (tick - START_TICK) / 60;
                auto minutes = total_seconds / 60;
                auto seconds = total_seconds % 60;

                hud_target
                    | draw::draw(
                        draw::VStack(3,
                            draw::Text(std::format("STAGE {:02}", stage_index), font::pico()),
                            draw::Text(std::format("{:02}:{:02}", minutes, seconds), font::mine())
                        ) | draw::resize_bottom(-1),
                        draw::Origin::Bottom
                    );

                hud_target
                    | draw::draw(
                        draw::VStack(draw::VAlignment::Left, 2,
                            draw::Text("Bub", font::pico())
                                | draw::resize_left(2),
                            draw::HStack(draw::HAlignment::Bottom, 3,
                                sheet.tile_ref(0, 6).resize_bottom(-4),
                                draw::Text(std::format("x {}", bub_lives), font::pico()),
                                draw::HSpacer(3),
                                draw::Text(std::format("{:02}", bub_score), font::mine())
                                    | draw::resize_bottom(-1)
                            )
                        ),
                        draw::Origin::BottomLeft
                    );

                if (mode == GameMode::OnePlayer) {
                    auto hud_observer_target = (target | draw::as_slice())
                        .resize_bottom(-(HEIGHT - 4) * 8)
                        .resize_horizontal(-4);
                    hud_observer_target
                        | draw::draw(
                            sheet.tile_ref(tick / 30 % 2 == 0 ? 0 : 1, 0)
                                | draw::map([] (Color c) -> Color {
                                    if (c == Color::rgba(92, 230, 52)) return Color::rgba(76, 206, 220);
                                    if (c == Color::rgba(252, 130, 116)) return Color::rgba(196, 118, 252);
                                    return c;
                                }),
                            draw::Origin::BottomRight
                        );
                }

                if (mode == GameMode::TwoPlayer) {
                    hud_target
                        | draw::draw(
                            draw::VStack(draw::VAlignment::Right, 2,
                                draw::Text("Bob", font::pico())
                                    | draw::resize_right(2),
                                draw::HStack(draw::HAlignment::Bottom, 3,
                                    draw::Text(std::format("{:02}", bob_score), font::mine())
                                        | draw::resize_bottom(-1),
                                    draw::HSpacer(3),
                                    draw::Text(std::format("{} x", bob_lives), font::pico()),
                                    sheet.tile_ref(1, 6).resize_bottom(-4)
                                )
                            ),
                            draw::Origin::BottomRight
                        );
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
        [[gnu::const]] auto sense(i32 x, i32 y, SensorDirection direction) const -> SensorResult {
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

        [[clang::always_inline]] [[gnu::const]]
        auto sense(Object const* relative_space, i32 x, i32 y, SensorDirection direction) const -> SensorResult {
            auto [ox, oy] = relative_space->pixel_pos();
            return sense(x + ox, y + oy, direction);
        }

        [[clang::always_inline]] [[gnu::const]]
        auto sense(Object const* relative_space, SensorDirection direction) const -> SensorResult {
            return sense(relative_space, 0, 0, direction);
        }

        [[clang::always_inline]] [[gnu::const]]
        static auto rotate(SensorDirection direction, u32 by_steps) noexcept -> SensorDirection {
            return (SensorDirection) (((u32) direction + by_steps) % 4);
        }

        [[clang::always_inline]] [[gnu::const]]
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

        [[clang::always_inline]] [[gnu::const]]
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
            for (Box<Object> const& object : objects) if (object->is_dynobject()) object_count += 1;
            writer.u32(object_count);

            for (Box<Object> const& object : objects) if (object->is_dynobject()) {
                // Write classname (32 char cstring padded with null terminators)
                std::string_view name = object->classname;

                for (u32 i = 0; i < 32; i += 1) if (i < name.size()) writer.u8(name[i]); else writer.u8(0);

                writer.i32((i32) object->position.x);
                writer.i32((i32) object->position.y);

                std::vector<u8> userdata;
                BinaryWriter userdata_writer { std::back_inserter(userdata) };

                const auto descriptor = class_loader::load(io, object->classname);
                descriptor.serializer(object.raw(), userdata_writer);

                for (u32 i = 0; i < 128; i += 1) {
                    if (i < userdata.size()) writer.u8(userdata[i]);
                    else writer.u8(0);
                }
            }

            io.write_file(std::format("../res/stage/{}.stage", stage_index), result);
        }

        template <const bool DEV = false> void reload(Io& io) {
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
                    instance->classname = classname;
                    this->objects.emplace_back(std::move(instance));
                }
            } else {
                for (Tile& tile : tiles) tile = {};
            }
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
                    auto replacement = descriptor.rebuilder(object.raw());

                    replacement->position = object->position;
                    replacement->classname = object->classname;

                    std::swap(object, replacement);
                }
            }
            editor_object_reload(io);

            apply_removal_queue();
            class_loader::drop_old_object_classes();
        }
    };
}
