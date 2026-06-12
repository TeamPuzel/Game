#pragma once
#include "scene.hpp"
#include "stage.hpp"
#include "scoreboard.hpp"
#include "controls.hpp"
#include <rng>

namespace bubble {
    class Title final : public Scene {
        u32 tick = 0;
        Grid<Image> sheet;
        Box<SoundLibrary> sounds;
        i32 menu_selection = 0;

        static constexpr i32 MENU_SELECTION_END = 4;

        static constexpr auto palette = std::to_array<Color>({
            draw::color::pico::RED,
            draw::color::pico::ORANGE,
            draw::color::pico::YELLOW,
            draw::color::pico::GREEN,
            draw::color::pico::LIGHT_BLUE,
            draw::color::pico::LAVENDER,
            draw::color::pico::PINK
        });

        struct Bubble final {
            fixed x;
            fixed y;
            fixed vx;
            fixed vy;
            Color color;
            i32 ttl;
            i32 max_ttl;

            static constexpr fixed SPEED = fixed(0, 128);
            static constexpr i32 WIDTH_RADIUS = 7;
            static constexpr i32 HEIGHT_RADIUS = 7;

            Bubble(rng::Xoshiro256StarStar& rng, i32 w, i32 h) {
                x       = random_until(0, w, rng);
                y       = random_until(0, h, rng);
                vx      = random_to(-SPEED, SPEED, rng);
                vy      = random_to(-SPEED, SPEED, rng);
                color   = palette.at(random_until(0, 7, rng));
                ttl     = random_to(120, 160, rng);
                max_ttl = ttl;
            }

            void update(std::span<Bubble> bubbles) {
                x += vx;
                y += vy;
                ttl = std::max(0, ttl - 1);

                for (auto& bubble : bubbles) if (&bubble != this) {
                    // Calculate distance between bubbles.
                    auto dx = x - bubble.x;
                    auto dy = y - bubble.y;

                    if (math::abs(dx) < WIDTH_RADIUS * 2 and math::abs(dy) < HEIGHT_RADIUS * 2) {
                        if (dx != 0) x += math::sign(dx) / 2;
                        if (dy != 0) y += math::sign(dy) / 2;

                        if (dx == 0 and dy == 0) x += fixed(0, 128);
                    }
                }
            }
        };

        std::vector<Bubble> bubbles;
        mutable i32 last_width  = Stage::WIDTH  * 8;
        mutable i32 last_height = Stage::HEIGHT * 8;
        mutable bool should_init_bubbles = true;
        rng::Xoshiro256StarStar rng;

        void update_bubbles() {
            if (tick % 3) {
                for (u64 i = 0; i < 2; i += 1) {
                    bubbles.emplace_back(rng, last_width, last_height);
                }
            }

            for (auto& bubble : bubbles) bubble.update(bubbles);

            std::erase_if(bubbles, [] (auto b) { return b.ttl == 0; });
        }

        void init_bubbles() {
            for (u64 i = 0; i < 256; i += 1) update_bubbles();
        }

      public:
        Title(Io& io, Grid<Image> sheet, Box<SoundLibrary> sounds)
            : sheet(std::move(sheet)), sounds(std::move(sounds)), rng(io.get_random()) {}

        Title(Io& io) : sheet(
            draw::TgaImage::from(io.read_file("res/tiles.tga"))
                | draw::flatten<Image>()
                | draw::grid(16, 16)
        ), rng(io.get_random()) {
            sounds = Box<SoundLibrary>::make();
            using enum SoundLibrary::SoundRequest::Type;

            // The async loader is FIFO so these should be in the order they're likely first needed.
            sounds->enqueue("music::title",    "res/snes_bubble_bustin.ogg", Ogg);
            sounds->enqueue("music::gameplay", "res/snes_staff_roll.ogg",    Ogg);
            sounds->enqueue("music::score",    "res/snes_champion.ogg",      Ogg);
            sounds->enqueue("music::controls", "res/snes_pro_player.ogg",    Ogg);

            sounds->enqueue("sfx::launch",       "res/sfx_2.wav",  Wave);
            sounds->enqueue("sfx::jump",         "res/sfx_9.wav",  Wave);
            sounds->enqueue("sfx::death",        "res/sfx_13.wav", Wave);
            sounds->enqueue("sfx::enemy_launch", "res/sfx_17.wav", Wave);
            sounds->enqueue("sfx::pickup",       "res/sfx_6.wav",  Wave);
            sounds->enqueue("sfx::hit",          "res/sfx_3.wav",  Wave);

            sounds->fetch(io);
        }

        static constexpr auto menu = std::to_array<std::string_view>({
            "1 Player",
            "2 Player",
            "2 Player Versus",
            "Scoreboard",
            "Controls"
        });

        void start(Io& io) {
            switch (menu_selection) {
                case 0: transition(Stage::load(io, 1, std::move(sheet), std::move(sounds), GameMode::OnePlayer));       break;
                case 1: transition(Stage::load(io, 1, std::move(sheet), std::move(sounds), GameMode::TwoPlayer));       break;
                case 2: transition(Stage::load(io, 1, std::move(sheet), std::move(sounds), GameMode::TwoPlayerVersus)); break;
                case 3: transition(Box<ScoreBoard>::make(io, std::move(sheet), std::move(sounds))); break;
                case 4: transition(Box<Controls>::make(io, std::move(sheet), std::move(sounds)));   break;
            }
        }

        void start_editor(Io& io) {
            transition(Stage::load(io, 1, std::move(sheet), std::move(sounds), GameMode::OnePlayer, true));
        }

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound) override {
            using rt::Key;
            using rt::Button;

            if (tick == 0) {
                sound.play(sounds->get("music::title").clone() | sound::loop());
            }

            if (input.gamepad_pressed(Button::Up) or input.key_repeating(Key::Up)) {
                menu_selection = std::clamp(menu_selection - 1, 0, MENU_SELECTION_END);
            }
            if (input.gamepad_pressed(Button::Down) or input.key_repeating(Key::Down)) {
                menu_selection = std::clamp(menu_selection + 1, 0, MENU_SELECTION_END);
            }

            if (input.gamepad_pressed(Button::A) or input.key_pressed(Key::Enter)) {
                sound.stop();
                return start(io);
            }

            if (input.key_pressed(Key::Tab)) {
                sound.stop();
                return start_editor(io);
            }

            update_bubbles();

            if (should_init_bubbles) {
                should_init_bubbles = false;
                init_bubbles();
            }

            tick += 1;
        }

        void draw(Io& io, rt::Input const& input, Ref<Image> target) const override {
            target | draw::clear();

            target
                | draw::draw(
                    draw::FilledRectangle(target.width(), target.height(), Color::gray(15))
                        | draw::dither()
                );

            for (auto const& bubble : bubbles) {
                std::optional tile = sheet.tile_ref(4, 2);
                if (bubble.ttl < 20) tile.emplace(sheet.tile_ref(5, 2));
                if (bubble.ttl < 10)  tile.emplace(sheet.tile_ref(6, 2));

                static constexpr fixed MAX_ALPHA = 140;

                // Time alpha.
                fixed time_alpha = MAX_ALPHA;
                fixed fade_time = 20;

                if (bubble.ttl < 20) {
                    // Fade out while popping.
                    fixed t = fixed(bubble.ttl) / fade_time;
                    time_alpha = math::lerp(fixed(0), fixed(MAX_ALPHA), t);
                } else if (bubble.max_ttl - bubble.ttl < 20) {
                    // Fade in when new.
                    fixed age = fixed(bubble.max_ttl - bubble.ttl);
                    fixed t = age / fade_time;
                    time_alpha = math::lerp(fixed(0), fixed(MAX_ALPHA), t);
                }

                // Distance alpha for readability of menu text.
                auto center_x = fixed(target.width() / 2);
                auto center_y = fixed(target.height() / 2 + 64);

                auto dx = math::abs(center_x - bubble.x);
                auto dy = math::abs(center_y - bubble.y);
                auto d = (dx + dy) / fixed(2);

                fixed dist_alpha = MAX_ALPHA;
                fixed clear_radius = 164;

                if (d < clear_radius) {
                    fixed t = d / clear_radius;
                    dist_alpha = math::lerp(fixed(0), fixed(MAX_ALPHA), t);
                }

                i32 alpha = (i32) std::min(time_alpha, dist_alpha);

                target
                    | draw::draw(
                        *tile
                            | draw::map(Color::rgba(92, 230, 52), draw::color::pico::PINK)
                            | draw::map([&] (Color c) { return c.with_a(alpha); }),
                        (i32) bubble.x,
                        (i32) bubble.y,
                        draw::blend::alpha
                    );
            }

            auto title_card = sheet.inner
                | draw::as_ref()
                | draw::slice(96, 112, 192, 164)
                | draw::apply_if(tick / 12 % 2 == 0, draw::map([] (Color c) -> Color {
                    return c == draw::color::pico::PINK
                        ? draw::color::pico::LIGHT_PINK
                        : c;
                }));

            auto menu_item_color = [this] (i32 index) -> draw::Color {
                return index == menu_selection
                    ? draw::color::pico::PINK
                    : draw::color::pico::WHITE;
            };

            auto title_screen = draw::VStack(draw::VAlignment::Center, 1,
                title_card,
                draw::VForEach(draw::VAlignment::Center, 4,
                    // Cute how for all the advanced capabilities of LLVM, libc++ doesn't have enumerate before
                    // clang 23 which I am not currently using, this is absolutely lovely.
                    std::views::zip(menu, std::views::iota(0)),
                    [&] (auto element) { auto [text, index] = element;
                        auto sprite = sheet.tile_ref(2, 6)
                            .resize_bottom(-11)
                            .resize_right(-12)
                            .shift(4 * (input.counter() / 10 % 4), 0)
                                | draw::map(draw::color::WHITE, draw::color::pico::PINK);

                        auto selected = draw::HStack(4,
                            sprite,
                            draw::Text(text, font::mine(), draw::color::pico::PINK),
                            sprite | draw::mirror_x()
                        );
                        auto not_selected =
                            draw::Text(text, font::mine(), draw::color::pico::WHITE);

                        using ResultPlane = draw::EitherPlane<decltype(selected), decltype(not_selected)>;

                        if (index == menu_selection) {
                            return ResultPlane(std::move(selected));
                        } else {
                            return ResultPlane(std::move(not_selected));
                        }
                    }
                )
            );

            target
                | draw::draw(title_screen, draw::Origin::Center);

            if (target.width() != last_width or target.height() != last_height) {
                last_width = target.width();
                last_height = target.height();
                should_init_bubbles = true;
            }
        }
    };
}
