#pragma once
#include "scene.hpp"
#include "stage.hpp"
#include "venue.hpp"

namespace bubble {
    class Title : public Scene {
        u32 tick = 0;
        Grid<Image> sheet;
        i32 menu_selection = 0;

      public:
        friend class Box<Editor>; // What.

        Title(Io& io) : sheet(
            draw::TgaImage::from(io.read_file("res/tiles.tga"))
                | draw::flatten<Image>()
                | draw::grid(16, 16)
        ) {}

        void start(Io& io) {
            switch (menu_selection) {
                case 0: transition(Venue::of(io, Play::OnePlayer,       std::move(sheet))); break;
                case 1: transition(Venue::of(io, Play::TwoPlayer,       std::move(sheet))); break;
                case 2: transition(Venue::of(io, Play::TwoPlayerVersus, std::move(sheet))); break;
            }
        }

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound) override {
            using rt::Key;
            using rt::Button;

            if (tick == 0) {
                sound.play(
                    sound::Wave::from(io.read_oggfile("res/the_secret_room.ogg"))
                        | sound::loop()
                );
            }

            if (input.gamepad_pressed(Button::Up) or input.key_pressed(Key::Up)) {
                menu_selection = std::clamp(menu_selection - 1, 0, 2);
            }
            if (input.gamepad_pressed(Button::Down) or input.key_pressed(Key::Down)) {
                menu_selection = std::clamp(menu_selection + 1, 0, 2);
            }

            if (input.gamepad_pressed(Button::A) or input.key_pressed(Key::Enter)) {
                sound.stop();
                start(io);
            }

            tick += 1;
        }

        void draw(Io& io, rt::Input const& input, Ref<Image> target) const override {
            auto title_card = sheet.inner
                | draw::as_ref()
                | draw::slice(96, 112, 192, 164)
                | draw::apply_if(tick / 12 % 2 == 0, draw::map([] (Color c, i32 x, i32 y) -> Color {
                    return c == draw::color::pico::PINK
                        ? draw::color::pico::LIGHT_PINK
                        : c;
                }));

            auto menu_item_color = [this] (i32 index) -> draw::Color {
                return index == menu_selection
                    ? draw::color::pico::PINK
                    : draw::color::pico::WHITE;
            };

            auto title_screen = draw::VStack(draw::VAlignment::Center, 4,
                title_card,
                draw::Text("1 Player", font::pod(), menu_item_color(0)),
                draw::Text("2 Player", font::pod(), menu_item_color(1)),
                draw::Text("2 Player Versus", font::pod(), menu_item_color(2))
            );

            target
                | draw::clear()
                | draw::draw(title_screen, draw::Origin::Center);
        }
    };
}
