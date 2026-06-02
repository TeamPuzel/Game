#pragma once
#include "scene.hpp"

namespace bubble {
    class Controls final : public Scene {
        u32 tick = 0;
        Grid<Image> sheet;

      public:
        Controls(Io& io, Grid<Image> sheet) : sheet(std::move(sheet)) {}

        Controls(Io& io) : Controls(io,
            draw::TgaImage::from(io.read_file("res/tiles.tga"))
                | draw::flatten<Image>()
                | draw::grid(16, 16)
        ) {}

        void return_to_title(Io& io);

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound) override {
            if (tick == 0) sound.play(
                sound::Wave::from(io.read_oggfile("res/snes_pro_player.ogg"))
                    | sound::loop()
            );

            if (input.key_pressed(rt::Key::Enter) or input.gamepad_pressed(rt::Button::A)) {
                sound.stop();
                return_to_title(io);
            }

            tick += 1;
        }

        void draw(Io& io, rt::Input const& input, Ref<Image> target) const override {
            auto large_sheet = sheet.inner | draw::as_ref() | draw::grid(32, 32);

            target
                | draw::clear()
                | draw::draw(
                    draw::FilledRectangle(target.width(), target.height(), Color::gray(15))
                        | draw::dither()
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
    };
}
