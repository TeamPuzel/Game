#pragma once
#include "scene.hpp"
#include "stage.hpp"
#include "scoreboard.hpp"
#include "controls.hpp"

namespace bubble {
    class Title final : public Scene {
        u32 tick = 0;
        Grid<Image> sheet;
        Box<SoundLibrary> sounds;
        i32 menu_selection = 0;

        static constexpr i32 MENU_SELECTION_END = 4;

      public:
        Title(Io& io, Grid<Image> sheet, Box<SoundLibrary> sounds) : sheet(std::move(sheet)), sounds(std::move(sounds)) {}

        Title(Io& io) : sheet(
            draw::TgaImage::from(io.read_file("res/tiles.tga"))
                | draw::flatten<Image>()
                | draw::grid(16, 16)
        ) {
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

            sounds->fetch(io);
        }

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

            tick += 1;
        }

        void draw(Io& io, rt::Input const& input, Ref<Image> target) const override {
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
                draw::VStack(draw::VAlignment::Center, 4,
                    draw::Text("1 Player", font::mine(), menu_item_color(0)),
                    draw::Text("2 Player", font::mine(), menu_item_color(1)),
                    draw::Text("2 Player Versus", font::mine(), menu_item_color(2)),
                    draw::Text("Scoreboard", font::mine(), menu_item_color(3)),
                    draw::Text("Controls", font::mine(), menu_item_color(4))
                )
            );

            target
                | draw::clear()
                | draw::draw(
                    draw::FilledRectangle(target.width(), target.height(), Color::gray(15))
                        | draw::dither()
                )
                | draw::draw(title_screen, draw::Origin::Center);
        }
    };
}
