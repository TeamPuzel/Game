#pragma once
#include <bubble>

namespace bubble {
    class Player final : public Object, public DefaultCodable<Player> {
      public:
        void update(Io& io, rt::Input const& input, Stage& stage) noexcept override {

        }

        void draw(draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept override {

        }
    };
}
