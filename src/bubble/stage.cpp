#include "stage.hpp"
#include "../object/Player.hpp"
#include "scoreboard.hpp"
#include <ranges>

using namespace bubble;

void Stage::begin_transition(Io* io) {
    transition_timer = HEIGHT * 8;

    auto target = pre_transition_nes_target | draw::as_ref();
    target | draw::clear();

    if (stage_index > 1) {
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
    }

    if (io) reload(*io);
}

void Stage::update(Io& io, rt::Input const& input, rt::SoundStage& sound) {
    if (transition_timer) {
        u32 player_bubble_count = 0;

        for (auto object : objs()) {
            if (auto player_bubble = isa_cast<PlayerBubble>("PlayerBubble", object)) {
                player_bubble->update(io, input, sound, *this);
                player_bubble_count += 1;
            }
        }

        apply_removal_queue();

        transition_timer -= 2;

        // Underflow sanity check.
        if (transition_timer > HEIGHT * 8) transition_timer = 0;

        return;
    } else {
        u32 transition_count = 0;
        u32 scoring_count = 0;

        for (auto object : objs()) {
            if (object->prevents_transition()) transition_count += 1;
            if (object->prevents_scoring()) scoring_count += 1;
        }

        // TODO: Also prevent post scoring bubbles from popping.
        if (scoring_count == 0) {
            auto consider_score = [&] (u32 score) -> std::optional<char> {
                auto decimal = std::format("{}", score)
                    | std::views::reverse;

                if (decimal.size() >= 3 and decimal[2] == decimal[3]) {
                    return decimal[2];
                }

                return std::nullopt;
            };

            // I could just chain the Alternative implementation (or_else)
            // but the signature seems very weird in C++ and I don't feel like figuring out the nested template errors.
            // This code looks really bad but it works.
            auto                   consideration = consider_score(bub_score);
            if (not consideration) consideration = consider_score(bob_score);

            for (auto object : objs()) {
                if (auto bubble = isa_cast<Bubble>("Bubble", object)) {
                    if (consideration) {
                        bubble->pop_special(*consideration, sound, *this);
                    } else {
                        bubble->pop(sound, *this);
                    }
                }
            }
        }

        if (tick > START_TICK and transition_count == 0) {
            std::vector<Box<Object>> players;
            for (auto object : objs()) {
                if (auto player = isa_cast<Player>("Player", object)) {
                    players.emplace_back(take(object));
                }
            }

            stage_index += 1;
            begin_transition(io);

            for (auto player : players | std::views::as_rvalue) {
                ((Player*) add(std::move(player)))->to_bubble(*this);
            }
        }
    }

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

                                for (auto obj : objs()) {
                                    auto [ox, oy] = obj->pixel_pos();

                                    i32 dx = ox - mouse->x;
                                    i32 dy = oy - mouse->y;
                                    i32 dist_sq = dx * dx + dy * dy;

                                    if (dist_sq < min_dist_sq) {
                                        min_dist_sq = dist_sq;
                                        closest_object = obj;
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

    if (tick == 0) {
        static constexpr auto EDGE_OFFSET = 64;
        static constexpr auto Y_POSITION = 140;

        auto player_descriptor = class_loader::load(io, "Player");

        auto bub = (Player*) add(player_descriptor.initializer(EDGE_OFFSET, Y_POSITION));
        bub->to_bubble(*this);

        if (mode == GameMode::TwoPlayer) {
            auto bob = (Player*) add(player_descriptor.initializer(WIDTH * 8 - EDGE_OFFSET, Y_POSITION));
            bob->alternate();
            bob->to_bubble(*this);
        }
    }

    if (tick == START_TICK - 1) {
        begin_transition();
    }

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
    } else {
        for (usize i = 0; i < objects.size(); i += 1) if (objects[i]) {
            if (isa_cast<PlayerBubble>("PlayerBubble", objects[i].raw())) {
                objects[i]->update(io, input, sound, *this);
            }
        }
    }

    apply_removal_queue();

    tick += 1;
}

void Stage::draw_viewport(Io& io, rt::Input const& input, Ref<Image> target) const {
    if (transition_timer) {
        target | draw::clear();

        target | draw::draw(
            pre_transition_nes_target
                | draw::as_ref()
                | draw::resize_top(-(HEIGHT * 8 - transition_timer))
        );

        // We will be drawing the new content at an offset (except for player transition bubbles).
        // This emulates the transition scroll effect which is normally performed more trivially with
        // the hardware scrolling feature of the NES. This isn't too complicated however.
        auto original_target = target;
        auto target = original_target
            | draw::resize_top(-transition_timer);

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

        // We draw just the transition bubbles independent of the transition scroll, the enemies scroll
        // with the level itself. This isn't quite what the original did but it works well enough.
        for (auto object : objs()) {
            const auto [ox, oy] = object->pixel_pos();

            if (isa_cast<PlayerBubble>("PlayerBubble", object)) {
                object->draw(io, original_target | draw::shift(ox, oy), *this);
            } else {
                if (input.counter() % 2 == 0)
                    object->draw(io, target.shift(ox, oy), *this);
            }
        }

        return;
    }

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

        for (auto object : objs()) if (isa_cast<PlayerBubble>("PlayerBubble", object)) {
            const auto [ox, oy] = object->pixel_pos();
            object->draw(io, target | draw::shift(ox, oy), *this);
        }

        target | draw::draw(std::move(intro) | draw::resize_bottom(100), draw::Origin::Center);

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
        for (auto object : objs()) {
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

void Stage::lose_life_bub() {
    if (not bub_lives)
        for (auto obj : objs())
            if (auto p = isa_cast<Player>("Player", obj); p and p->character == Player::Character::Bub)
                remove(obj);

    bub_lives = std::max(0, (i32) bub_lives - 1);

    should_check_for_game_end = true;
}

void Stage::lose_life_bob() {
    if (not bob_lives)
        for (auto obj : objs())
            if (auto p = isa_cast<Player>("Player", obj); p and p->character == Player::Character::Bob)
                remove(obj);

    bob_lives = std::max(0, (i32) bob_lives - 1);

    should_check_for_game_end = true;
}

void Stage::check_for_game_end() {
    if (not std::ranges::any_of(objs(), [] (auto p) -> bool { return isa_cast<Player>("Player", p); })) {
        game_end_timer = GAME_END_DELAY;
    }
}

void Stage::award_points_bub(u32 points) {
    bub_score += points;
}

void Stage::award_points_bob(u32 points) {
    bob_score += points;
}

void Stage::unsafe_hot_reload_child(Box<Object>& object) {
    if (not object) return;

    Io& io = Io::unsafe_get_threadlocal_io();

    if (not object->is_dynobject()) throw std::logic_error("tried reloading a non dynamic child object");

    auto descriptor = class_loader::load(io, object->classname());
    auto replacement = descriptor.rebuilder(object.raw(), *this);

    replacement->position = object->position;

    std::swap(object, replacement);
}
