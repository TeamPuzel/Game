#pragma once
#include "scene.hpp"
#include "stage.hpp"

namespace bubble {
    enum class Mode {
        OnePlayer,
        TwoPlayer,
        TwoPlayerVersus
    };

    /// The NES game smoothly scrolls between levels (trivial on the NES hardware).
    /// This is not as trivial without such a hardware background, so we need a venue to manage the stage transitions.
    class Venue : public Scene {
        Mode mode;
        Grid<Image> sheet;
        u32 tick = 0;

        Venue(Io& io, Mode mode, Grid<Image>&& sheet) : mode(mode), sheet(std::move(sheet)) {}

      public:
        friend class Box<Venue>;

        static auto of(Io& io, Mode mode, Grid<Image>&& sheet) -> Box<Venue> {
            return Box<Venue>::make(io, mode, std::move(sheet));
        }

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound) override {
            if (tick == 0) sound.play(sound::Wave::from(io.read_oggfile("res/snes_staff_roll.ogg")));
            if (tick == 8 * 60) transition(Stage::load(io, 1, std::move(sheet)));

            tick += 1;
        }

        void draw(Io& io, rt::Input const& input, Ref<Image> target) const override {
            target | draw::clear();

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
        }
    };
}
