#pragma once
#include "scene.hpp"
#include <algorithm>
#include <ranges>
#include <expected>
#include <vector>
#include <queue>

namespace bubble {
    class ScoreBoard final : public Scene {
      public:
        enum class Character : u8 { Bub, Bob };

        struct PendingScore final {
            Character character;
            u32 score;
        };

        static auto character_str(Character character) -> std::string_view {
            switch (character) {
                case Character::Bub: return "Bub";
                case Character::Bob: return "Bob";
            }
        }

      private:
        u32 tick = 0;
        Grid<Image> sheet;
        Box<SoundLibrary> sounds;

        // This is more than enough, and it fits the NES resolution in a pleasing way.
        static constexpr auto NAME_LENGTH = 18;

        bool score_input;
        i32 score_input_selection = 26;
        i32 score_input_cursor = 0;
        std::array<char, NAME_LENGTH> score_input_buffer;
        std::queue<PendingScore> pending;

        void reset_input_state() {
            score_input_selection = 26;
            score_input_cursor = 0;
            score_input_buffer.fill(' ');
        }

        struct Score final {
            bool is_empty;
            std::string name_storage;
            u32 score_storage;

            Score(std::string name, u32 score) : is_empty(false) {
                if (name.size() > NAME_LENGTH) throw std::runtime_error("name too large");
                this->name_storage = std::move(name);
                this->score_storage = score;
            }

            Score() : is_empty(true) {}

            auto name() const -> std::string_view {
                if (is_empty) throw std::out_of_range("name");
                return name_storage;
            }

            auto score() const -> u32 {
                if (is_empty) throw std::out_of_range("score");
                return score_storage;
            }

            operator bool() const { return not is_empty; }
        };

        std::vector<Score> scores;

        struct CrossChar { char character; i32 index; };

        static auto cross_input_range() {
            return std::views::iota(0, 53) | std::views::transform([] (i32 i) -> CrossChar {
                if (i < 26) {
                    return { char('Z' - i), i };
                } else if (i == 26) {
                    return { ' ', i };
                } else {
                    return { char('a' + (i - 27)), i };
                }
            });
        }

        // Compute initial selection from the buffer at cursor.
        void recompute_selection() {
            char c = score_input_buffer[score_input_cursor];

            if (c >= 'a' and c <= 'z') {
                score_input_selection = 'z' - c;
            } else if (c == ' ') {
                score_input_selection = 26;
            } else if (c >= 'A' and c <= 'Z') {
                score_input_selection = 27 + (c - 'A');
            } else {
                score_input_selection = 26;
            }
        }

        // Apply the selected glyph to the buffer at cursor.
        void apply_selection() {
            char c = ' ';

            if (score_input_selection < 26) {
                c = char('z' - score_input_selection);
            } else if (score_input_selection == 26) {
                c = ' ';
            } else {
                c = char('A' + (score_input_selection - 27));
            }

            score_input_buffer[score_input_cursor] = c;
        }

        enum class InputError {
            TooShort
        };

        // This will produce a string stripped of prefix and suffix spaces, and with multiple spaces
        // in the middle all shortened to just one.
        // This can fail when the input is invalid, such as being too short (less than two characters).
        auto sanitized_input() const -> std::expected<std::string, InputError> {
            std::string result;
            result.reserve(score_input_buffer.size());

            bool last_was_space = true;

            for (char c : score_input_buffer) {
                if (c == ' ') {
                    if (not last_was_space) {
                        result.push_back(' ');
                        last_was_space = true;
                    }
                } else {
                    result.push_back(c);
                    last_was_space = false;
                }
            }

            // Strip trailing space if one was added at the very end.
            if (not result.empty() and result.back() == ' ') {
                result.pop_back();
            }

            // Validate length.
            if (result.size() < 2) {
                return std::unexpected(InputError::TooShort);
            }

            return result;
        }

        void sort_scores() {
            std::ranges::sort(scores, std::greater(), &Score::score_storage);
        }

        void try_submit_score(Io& io) {
            auto name = sanitized_input();

            if (name) {
                scores.emplace_back(*name, pending.front().score);
                sort_scores();
                scores.resize(10);

                store(io);
                reset_input_state();

                pending.pop();

                if (pending.empty()) {
                    score_input = false;
                }
            } else {
                // TODO: Notify user.
            }
        }

      public:
        ScoreBoard(Io& io, Grid<Image> sheet, Box<SoundLibrary> sounds, std::queue<PendingScore> pending = {})
            : sheet(std::move(sheet)), sounds(std::move(sounds)), score_input(not pending.empty()), pending(std::move(pending))
        {
            reset_input_state();
            load(io);
        }

        ScoreBoard(Io& io, std::queue<PendingScore> pending = {}) : ScoreBoard(io,
            draw::TgaImage::from(io.read_file("res/tiles.tga"))
                | draw::flatten<Image>()
                | draw::grid(16, 16),
            {},
            std::move(pending)
        ) {}

        void load(Io& io) {
            scores.clear();
            scores.reserve(10);

            if (auto file = io.try_read_file(io.get_prefix_path(BUBBLE_TEAMNAME, BUBBLE_APPNAME) + "scores.bin")) {
                io::BinaryReader reader { std::span(*file) };

                u32 score_count = reader.u32();

                for (u32 i = 0; i < score_count; i += 1) {
                    scores.emplace_back(reader.cstr(NAME_LENGTH + 1), reader.u32());
                }
            }

            while (scores.size() != 10) scores.emplace_back();
        }

        void store(Io& io) {
            sort_scores();

            auto all_scores = scores
                | std::views::filter([] (auto const& score) -> bool { return not score.is_empty; })
                | std::ranges::to<std::vector>();

            std::vector<u8> result;
            io::BinaryWriter writer { std::back_inserter(result) };

            writer.u32(all_scores.size());

            for (auto score : all_scores) {
                for (u32 i = 0; i < NAME_LENGTH; i += 1) {
                    if (i < score.name_storage.size()) {
                        writer.u8(score.name_storage[i]);
                    } else {
                        writer.u8(0);
                    }
                }
                writer.u8(0); // The 25th character (fallback sentinel).
                writer.u32(score.score_storage);
            }

            io.write_file(io.get_prefix_path(BUBBLE_TEAMNAME, BUBBLE_APPNAME) + "scores.bin", result);
        }

        void return_to_title(Io& io);

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound) override {
            if (tick == 0 and sounds) sound.play(sounds->get("music::score").clone() | sound::loop());

            if (input.gamepad_pressed(rt::Button::A) or input.key_pressed(rt::Key::Enter)) {
                if (score_input) {
                    try_submit_score(io);
                } else {
                    sound.stop();
                    return_to_title(io);
                }
            }

            if (input.gamepad_pressed(rt::Button::B) or input.key_pressed(rt::Key::Escape)) {
                if (score_input) {
                    score_input = false;
                }
            }

            if (input.gamepad_pressed(rt::Button::Left) or input.key_repeating(rt::Key::Left)) {
                score_input_cursor = std::max(0, score_input_cursor - 1);
                recompute_selection();
            }

            if (input.gamepad_pressed(rt::Button::Right) or input.key_repeating(rt::Key::Right)) {
                score_input_cursor = std::min(NAME_LENGTH - 1, score_input_cursor + 1);
                recompute_selection();
            }

            if (input.gamepad_pressed(rt::Button::Up) or input.key_repeating(rt::Key::Up)) {
                score_input_selection = std::min(52, score_input_selection + 1);
                apply_selection();
            }

            if (input.gamepad_pressed(rt::Button::Down) or input.key_repeating(rt::Key::Down)) {
                score_input_selection = std::max(0, score_input_selection - 1);
                apply_selection();
            }

            tick += 1;
        }

        void draw(Io& io, rt::Input const& input, Ref<Image> target) const override {
            auto large_sheet = sheet.inner | draw::as_ref() | draw::grid(32, 32);

            if (score_input) {
                auto pending_score = pending.front();

                target
                    | draw::clear()
                    | draw::draw(
                        draw::FilledRectangle(target.width(), target.height(), Color::gray(15))
                            | draw::dither()
                    );

                    auto name_indices = std::views::iota(0, i32(score_input_buffer.size()));

                    target
                        | draw::draw(
                            draw::HForEach(2, name_indices, [&] (i32 i) {
                                const i32 font_height = font::pod().height;

                                static constexpr auto SPACING = 3;
                                static constexpr auto FADE_CHARS = 4;
                                static constexpr auto MAX_CHAR_WIDTH = 10;

                                // Inactive column.
                                char display_char = score_input_buffer[i];
                                Color char_color = draw::color::WHITE;

                                if (display_char == ' ') {
                                    display_char = '_';
                                    char_color = Color::gray(100);
                                }

                                auto inactive_plane = draw::Text(std::string(1, display_char), font::pod(), char_color)
                                    | draw::slice(0, 0, MAX_CHAR_WIDTH, font_height);

                                // Active column.
                                auto raw_active_plane = draw::VForEach(SPACING, cross_input_range(), [font_height] (auto c) {
                                    return draw::Text(std::string(1, c.character), font::pod(), draw::color::WHITE)
                                        | draw::slice(0, 0, MAX_CHAR_WIDTH, font_height);
                                });

                                const i32 inactive_y = (raw_active_plane.height() - inactive_plane.height()) / 2;
                                const i32 native_y = score_input_selection * (font_height + SPACING);
                                auto active_plane = std::move(raw_active_plane)
                                    | draw::offset(0, inactive_y - native_y)
                                    | draw::map([inactive_y, font_height] (draw::Color c, i32 x, i32 y) -> Color {
                                        i32 distance = std::abs(y - inactive_y);

                                        // Deadzone of half font height.
                                        i32 deadzone = font_height / 2;
                                        // Effective distance (0 if we are inside the deadzone)
                                        i32 effective_distance = std::max(0, distance - deadzone);

                                        // Dividing by height * FADE_CHARS fades out over FADE_CHARS characters.
                                        i32 alpha_drop = (effective_distance * 255)
                                            / ((font_height + SPACING) * FADE_CHARS);

                                        u8 alpha = u8(std::max(0, 255 - alpha_drop));

                                        return c.with_a(u8((c.a * alpha) / 255));
                                    });

                                using ResultPlane = draw::EitherPlane<decltype(active_plane), decltype(inactive_plane)>;

                                if (i == score_input_cursor) {
                                    return ResultPlane(std::move(active_plane));
                                } else {
                                    return ResultPlane(std::move(inactive_plane));
                                }
                            }),
                            draw::Origin::Center,
                            draw::blend::alpha
                        );

                    auto header_target = target | draw::shift(0, -72);
                    header_target
                        | draw::draw(
                            draw::VStack(4,
                                draw::Text(character_str(pending_score.character), font::pico()),
                                draw::Text(std::format("SCORE: {:04}", pending_score.score), font::pico()),
                                draw::Text("--- ENTER YOUR NAME ---", font::pod())
                            ),
                            draw::Origin::Center
                        );

                    auto help_target = target | draw::resize(-4);

                    help_target
                        | draw::draw(
                            draw::VStack(draw::VAlignment::Left, 3,
                                draw::HStack(3,
                                    sheet.tile_ref(14, 4)
                                        | draw::resize_right(-3)
                                        | draw::resize_bottom(-3),
                                    sheet.tile_ref(15, 4)
                                        | draw::resize_right(-3)
                                        | draw::resize_bottom(-3),
                                    sheet.tile_ref(16, 4)
                                        | draw::resize_right(-3)
                                        | draw::resize_bottom(-3),
                                    sheet.tile_ref(17, 4)
                                        | draw::resize_right(-3)
                                        | draw::resize_bottom(-3),
                                    draw::Text("or ARROW KEYS for entry", font::pico(), Color::gray(90))
                                ),
                                draw::HStack(3,
                                    sheet.tile_ref(16, 2)
                                        | draw::resize_right(-3)
                                        | draw::resize_bottom(-3),
                                    draw::Text("or RETURN to continue", font::pico(), Color::gray(90)),
                                    sheet.tile_ref(15, 2)
                                        | draw::resize_right(-3)
                                        | draw::resize_bottom(-3),
                                    draw::Text("or ESCAPE to abort", font::pico(), Color::gray(90))
                                )
                            ),
                            draw::Origin::BottomLeft
                        );
            } else {
                target
                    | draw::clear()
                    | draw::draw(
                        draw::FilledRectangle(target.width(), target.height(), Color::gray(15))
                            | draw::dither()
                    )
                    | draw::draw(
                        draw::VStack(8,
                            draw::HStack(148,
                                large_sheet.tile(tick / 30 % 2 == 0 ? 2 : 3, 2),
                                large_sheet.tile(tick / 30 % 2 == 0 ? 2 : 3, 2)
                                    | draw::map([] (Color c) -> Color {
                                        if (c == Color::rgba(92, 230, 52)) return Color::rgba(76, 206, 220);
                                        if (c == Color::rgba(252, 130, 116)) return Color::rgba(196, 118, 252);
                                        return c;
                                    })
                            ),
                            draw::VSpacer(4),
                            draw::Text("--- SCOREBOARD ---", font::pod()),
                            draw::VForEach(scores, [&] (Score const& score) {
                                auto ret = Image(256, 12);
                                auto lhs = (ret | draw::as_ref() | draw::as_slice()).resize_right(-8);
                                auto rhs = (ret | draw::as_ref() | draw::as_slice()).resize_left(-8);

                                if (score) {
                                    lhs | draw::draw(
                                            draw::Text(std::format("{}", score.name()), font::mine()),
                                            draw::Origin::Right, draw::Origin::Center
                                        );
                                    rhs | draw::draw(
                                            draw::Text(std::format("{}", score.score()), font::mine()),
                                            draw::Origin::Left, draw::Origin::Center
                                        );
                                } else {
                                    lhs | draw::draw(
                                            draw::Text("---", font::mine()),
                                            draw::Origin::Right, draw::Origin::Center
                                        );
                                    rhs | draw::draw(
                                            draw::Text("0", font::mine()),
                                            draw::Origin::Left, draw::Origin::Center
                                        );
                                }

                                return ret;
                            })
                        ),
                        draw::Origin::Center
                    );

                auto help_target = target | draw::resize(-4);

                help_target
                    | draw::draw(
                        draw::HStack(3,
                            sheet.tile_ref(16, 2)
                                | draw::resize_right(-3)
                                | draw::resize_bottom(-3),
                            draw::Text("or RETURN to continue", font::pico(), Color::gray(90))
                        ),
                        draw::Origin::BottomLeft
                    );
            }
        }
    };
}
