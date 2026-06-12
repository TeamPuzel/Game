#pragma once
#include <bubble>
#include "Enemy.hpp"

namespace bubble {
    class FireBall : public CodableObject<FireBall> {
      public:
        enum class Direction : u8 { Left, Right } RELOAD direction;
        RELOAD usize tick = 0;

        static constexpr i32 LIFETIME = 30;
        static constexpr i32 SPEED = 2;

        explicit FireBall(point<fixed> position, Direction direction) : direction(direction) {
            this->position = position;
        }

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound, Stage& stage) noexcept override;

        void draw(Io& io, draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept override {
            target | draw::draw(
                stage.get_sheet().tile(tick / 4 % 6, 20)
                    | draw::apply_if(direction == Direction::Right, draw::mirror_x()),
                -8, -8
            );
        }
    };

    class Maita : public CodableObject<Maita, Enemy> {
      public:
        enum class Facing : u8 { Left, Right } SERIAL facing = Facing::Right;
        RELOAD u8 jump_lock = 0;

        static constexpr fixed FALL_SPEED = 1;
        static constexpr fixed SPEED = 1;
        static constexpr i32 WIDTH_RADIUS = 7;
        static constexpr i32 HEIGHT_RADIUS = 7;
        static constexpr i32 SNAP_DISTANCE_BACK = 5;
        static constexpr i32 SNAP_DISTANCE_FORWARD = 2;
        static constexpr i32 FIRE_DISTANCE = FireBall::LIFETIME * FireBall::SPEED;

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

        void fire(rt::SoundStage& sound, Stage& stage) {
            switch (facing) {
                case Facing::Left:  stage.add(Box<FireBall>::make(position, FireBall::Direction::Left));
                case Facing::Right: stage.add(Box<FireBall>::make(position, FireBall::Direction::Right));
            }
        }

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound, Stage& stage) noexcept override;

        void draw(Io& io, draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept override {
            target | draw::draw(
                stage.get_sheet().tile(tick / 6 % 2 == 0 ? 0 : 1, 19)
                    | draw::apply_if(facing == Facing::Right, draw::mirror_x()),
                -8, -8
            );
        }

        auto bubble_sprite_pos() const -> SpritePosition override {
            return { .x = 6, .y = 19 };
        }

        auto particle_sprite_pos() const -> SpritePosition override {
            return { .x = 9, .y = 19 };
        }

        void reset() override {
            state = State::Airborne;
        }

        void flip() noexcept override {
            switch (facing) {
                case Facing::Left:  facing = Facing::Right; break;
                case Facing::Right: facing = Facing::Left;  break;
            }
        }
    };

    template <> struct FallbackCoder<Maita> {
        static void deserialize(Box<Maita>& self, BinaryReader& reader) {
            self->facing = (Maita::Facing) reader.u8();
        }
    };
}
