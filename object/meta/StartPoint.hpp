#pragma once
#include <bubble>

namespace bubble {
    class StartPoint final : public CodableObject<StartPoint> {
      public:
        enum class Facing : u8 { Left, Right } [[=serial]] facing = Facing::Right;
        enum class Character : u8 { Bub, Bob } [[=serial]] character = Character::Bub;

        auto facing_str() const -> std::string_view {
            switch (facing) {
                case Facing::Left:  return "Left";  break;
                case Facing::Right: return "Right"; break;
            }
        }

        auto character_str() const -> std::string_view {
            switch (character) {
                case Character::Bub: return "Bub"; break;
                case Character::Bob: return "Bob"; break;
            }
        }

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound, Stage& stage) noexcept override {

        }

        void draw(Io& io, draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept override {
            auto text_target = target.shift(0, -10);

            if (stage.in_editor_mode()) {
                target
                    | draw::draw(draw::Rectangle(16, 16), -8, -8)
                    | draw::pixel(0, 0, draw::color::WHITE);

                text_target
                    | draw::draw(
                        draw::MultilineText(
                            std::format("Start Point\n{} / {}", character_str(), facing_str()), font::pico(io),
                            draw::VAlignment::Center
                        ),
                        draw::Origin::Bottom, draw::Origin::TopLeft
                    );
            }
        }

        void flip() noexcept override {
            switch (facing) {
                case Facing::Left:  facing = Facing::Right; break;
                case Facing::Right: facing = Facing::Left;  break;
            }
        }

        void alternate() noexcept override {
            switch (character) {
                case Character::Bub: character = Character::Bob; break;
                case Character::Bob: character = Character::Bub; break;
            }
        }
    };
}
