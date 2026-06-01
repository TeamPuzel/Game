#pragma once
#include <bubble>

namespace bubble {
    class Player final : public Object, public DefaultCodable<Player> {
      public:
        enum class Character : u8 { Bub, Bob } character = Character::Bub;

        void update(Io& io, rt::Input const& input, rt::SoundStage&, Stage& stage) noexcept override {

        }

        void draw(draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept override {

        }
    };
}
