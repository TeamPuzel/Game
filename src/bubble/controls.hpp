#pragma once
#include "scene.hpp"

namespace bubble {
    class Controls final : public Scene {
        u32 tick = 0;
        Grid<Image> sheet;
        Box<SoundLibrary> sounds;

      public:
        Controls(Io& io, Grid<Image> sheet, Box<SoundLibrary> sounds)
            : sheet(std::move(sheet)), sounds(std::move(sounds)) {}

        void return_to_title(Io& io);

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound) override {
            if (tick == 0) sound.play(sounds->get("music::controls").clone() | sound::loop());

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

            auto secondary = Color::gray(180);

            auto gamepad =
                draw::VStack(draw::VAlignment::Left, 4,
                    draw::Text("GAMEPAD", font::pod()),
                    draw::HStack(3,
                        sheet.tile_ref(15, 2)
                            | draw::resize_right(-3)
                            | draw::resize_bottom(-3),
                        draw::Text("to jump", font::pico(), secondary)
                    ),
                    draw::HStack(3,
                        sheet.tile_ref(16, 2)
                            | draw::resize_right(-3)
                            | draw::resize_bottom(-3),
                        draw::Text("to attack", font::pico(), secondary)
                    ),
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
                        draw::Text("to move", font::pico(), secondary)
                    )
                );

            auto keyboard =
                draw::VStack(draw::VAlignment::Left, 4,
                    draw::Text("KEYBOARD", font::pod()),

                    draw::Text("P1", font::mine()),
                    draw::Text("PERIOD to jump", font::pico(), secondary),
                    draw::Text("COMMA to attack", font::pico(), secondary),
                    draw::HStack(3,
                        draw::Text("UP, DOWN, LEFT and RIGHT to move", font::pico(), secondary)
                    ),

                    draw::VSpacer(8),

                    draw::Text("P2", font::mine()),
                    draw::Text("B to jump", font::pico(), secondary),
                    draw::Text("V to attack", font::pico(), secondary),
                    draw::HStack(3,
                        draw::Text("W, S, A and D to move", font::pico(), secondary)
                    )
                );

            target
                | draw::draw(
                    draw::VStack(draw::VAlignment::Left, 8,
                        draw::HStack(12,
                            keyboard
                                | draw::as_ref(),
                            gamepad
                                | draw::as_ref()
                                | draw::resize_bottom(keyboard.height() - gamepad.height())
                        ),

                        draw::VSpacer(4),

                        draw::VStack(draw::VAlignment::Left, 4,
                            draw::Text("debug", font::pico(), draw::color::pico::YELLOW),
                            draw::Text("ENGINE", font::pod())
                        ),
                        draw::Text("+/- to change scale", font::pico(), secondary),
                        draw::Text("9 to toggle the performance overlay", font::pico(), secondary),
                        draw::Text("0 to toggle vsync", font::pico(), secondary)
                    ),
                    draw::Origin::Center
                );

            auto help_target = target | draw::as_ref() | draw::resize(-4);

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
