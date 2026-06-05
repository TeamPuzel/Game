#pragma once
#include <bubble>
#include "Enemy.hpp"

namespace bubble {
    class ZenChan final : public CodableObject<ZenChan, Enemy> {
      public:
        enum class Facing : u8 { Left, Right } SERIAL facing = Facing::Right;
        RELOAD u8 jump_lock = 0;

        static constexpr fixed FALL_SPEED = 1;
        static constexpr fixed SPEED = 1;
        static constexpr i32 WIDTH_RADIUS = 7;
        static constexpr i32 HEIGHT_RADIUS = 7;
        static constexpr i32 SNAP_DISTANCE = 2;

        enum class State {
            Grounded,
            Airborne,
            Jumping,
            Leaping,
        } RELOAD state = State::Airborne;

        RELOAD usize tick = 0;

        auto facing_direction() const -> SensorDirection {
            switch (facing) {
                case Facing::Left:  return SensorDirection::Left;
                case Facing::Right: return SensorDirection::Right;
            }
        }

        void walk_forward() {
            switch (facing) {
                case Facing::Left:  position.x -= SPEED; break;
                case Facing::Right: position.x += SPEED; break;
            }
        }

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound, Stage& stage) noexcept override;

        void draw(Io& io, draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept override {
            target | draw::draw(
                stage.get_sheet().tile(tick / 6 % 2 == 0 ? 0 : 1, 18)
                    | draw::apply_if(facing == Facing::Right, draw::mirror_x()),
                -8, -8
            );
        }

        auto bubble_sprite_pos() const -> BubbleSpritePosition override {
            return { .x = 6, .y = 18 };
        }

        void flip() noexcept override {
            switch (facing) {
                case Facing::Left:  facing = Facing::Right; break;
                case Facing::Right: facing = Facing::Left;  break;
            }
        }
    };

    template <> struct FallbackCoder<ZenChan> {
        static void deserialize(Box<ZenChan>& self, BinaryReader& reader) {
            self->facing = (ZenChan::Facing) reader.u8();
        }
    };
}
