#pragma once
#include "scene.hpp"
#include "stage.hpp"

namespace bubble {
    enum class Play {
        OnePlayer,
        TwoPlayer,
        TwoPlayerVersus
    };

    /// The NES game smoothly scrolls between levels (trivial on the NES hardware).
    /// This is not as trivial without such a hardware background, so we need a venue to manage the stage transitions.
    class Venue : public Scene {
        Play play;
        Grid<Image> sheet;
        Box<Stage> stage;
        std::optional<Box<Stage>> outgoing_stage;
        u32 tick = 0;

        enum class State {
            Introduction, // The intro sequence.
            Action,       // Gameplay.
            Intermission, // Scene transition.
            Resolution    // Ending and score entry.
        } state = State::Introduction;

        Venue(Io& io, Play play, Grid<Image>&& sheet) : play(play), sheet(std::move(sheet)) {
            stage = Stage::load(io, "res/1.stage");
        }

      public:
        friend class Box<Venue>;
        friend class Editor;

        static auto of(Io& io, Play play, Grid<Image>&& sheet) -> Box<Venue> {
            return Box<Venue>::make(io, play, std::move(sheet));
        }

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound) override {
            switch (state) {
                case State::Introduction: {
                    if (tick == 0) sound.play(
                        sound::Wave::from(io.read_oggfile("res/the_quest_begins.ogg"))
                            | sound::loop()
                    );
                    if (tick == 8 * 60) state = State::Action;
                } break;
                case State::Action: {

                } break;
                case State::Intermission: {

                } break;
                case State::Resolution: {

                } break;
            }

            tick += 1;
        }

        void draw(Io& io, rt::Input const& input, Ref<Image> target) const override {
            target | draw::clear();

            switch (state) {
                case State::Introduction: {
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
                } break;
                case State::Action: {

                } break;
                case State::Intermission: {

                } break;
                case State::Resolution: {

                } break;
            }
        }

        void hot_reload(Io& io) override {
            if (state == State::Action) stage->hot_reload(io);
        }
    };
}
