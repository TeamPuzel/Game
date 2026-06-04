#pragma once
#include <bubble>
#include "Enemy.hpp"

namespace bubble {
    class Maita final : public Enemy, public DefaultCodable<Maita> {
      public:
        enum class Character : u8 { Bub, Bob } character = Character::Bub;

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound, Stage& stage) noexcept override {

        }

        void draw(Io& io, draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept override {
            target | draw::draw(stage.get_sheet().tile(0, 19), -8, -8);
        }

        auto bubble_sprite_pos() const -> BubbleSpritePosition override {
            return { .x = 6, .y = 19 };
        }
    };
}
