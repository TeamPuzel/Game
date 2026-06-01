#pragma once
#include "scene.hpp"
#include "stage.hpp"

namespace bubble {
    class Editor : public Scene {
        Box<Stage> stage;

        explicit Editor(Box<Stage>&& stage) : stage(std::move(stage)) {}

      public:
        friend class Box<Editor>; // What.

        static auto of(Box<Stage>&& stage) -> Box<Editor> {
            return Box<Editor>::make(std::move(stage));
        }

        auto finalize() -> Box<Stage> {
            return std::move(stage);
        }

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound) override {

        }

        void draw(Io& io, rt::Input const& input, Ref<Image> target) const override {
            stage->draw(io, input, target);

            if (auto mouse = input.mouse()) {
                target | draw::draw(
                    mouse->left ? stage->sheet.tile_ref(17, 0) : stage->sheet.tile_ref(16, 0),
                    mouse->x - 1, mouse->y - 1
                );
            }
        }

        void hot_reload(Io& io) override {
            stage->hot_reload(io);
        }
    };
}
