#pragma once
#include <bubble>

namespace bubble {
    class Bubble final : public Object {
      public:
        void update(Io& io, rt::Input const& input, rt::SoundStage&, Stage& stage) noexcept override {

        }

        void draw(Io& io, draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept override {
            target | draw::draw(
                stage.get_sheet().tile(4, 2),
                -8, -8
            );
        }
    };

    class Player final : public Object, public DefaultCodable<Player> {
      public:
        enum class Animation {
            None,
            Idle,
            Walk,
            Death
        };

        Animator<Animation> animator;

        enum class Facing : u8 { Left, Right } facing = Facing::Right;
        enum class Character : u8 { Bub, Bob } character = Character::Bub;

        enum class State : u8 {
            Grounded,
            Airborne,
            Jumping,
            Death
        } state = State::Airborne;

        static constexpr fixed FALL_SPEED = 1;
        static constexpr fixed SPEED = 1;
        static constexpr i32 WIDTH_RADIUS = 7;
        static constexpr i32 HEIGHT_RADIUS = 7;
        static constexpr i32 SNAP_DISTANCE = 2;

      private:
        auto get_input_left(rt::Input const& input) const -> bool {
            switch (character) {
                case Character::Bub:
                    return input.key_held(rt::Key::Left) or input.gamepad_held(rt::Button::Left, 0);
                case Character::Bob:
                    return input.key_held(rt::Key::A) or input.gamepad_held(rt::Button::Left, 1);
            }
        }

        auto get_input_right(rt::Input const& input) const -> bool {
            switch (character) {
                case Character::Bub:
                    return input.key_held(rt::Key::Right) or input.gamepad_held(rt::Button::Right, 0);
                case Character::Bob:
                    return input.key_held(rt::Key::D) or input.gamepad_held(rt::Button::Right, 1);
            }
        }

        auto get_input_attack(rt::Input const& input) const -> bool {
            switch (character) {
                case Character::Bub:
                    return input.key_held(rt::Key::Period) or input.gamepad_held(rt::Button::A, 0);
                case Character::Bob:
                    return input.key_held(rt::Key::V) or input.gamepad_held(rt::Button::A, 1);
            }
        }

        auto get_input_jump(rt::Input const& input) const -> bool {
            switch (character) {
                case Character::Bub:
                    return input.key_held(rt::Key::Comma) or input.gamepad_held(rt::Button::B, 0);
                case Character::Bob:
                    return input.key_held(rt::Key::B) or input.gamepad_held(rt::Button::B, 1);
            }
        }

      public:
        void update(Io& io, rt::Input const& input, rt::SoundStage&, Stage& stage) noexcept override {
            animator.update();

            if (animator.is(Animation::None)) {
                animator.play(Animation::Idle, 2, 12);
            }

            bool left = get_input_left(input);
            bool right = get_input_right(input);
            bool attack = get_input_attack(input);
            bool jump = get_input_jump(input);

            if (left and not right) facing = Facing::Left;
            if (right and not left) facing = Facing::Right;

            switch (state) {
                case State::Grounded: {
                    if (left and not right) position.x -= 1;
                    if (right and not left) position.x += 1;

                    if (left or right and not animator.is(Animation::Walk)) {
                        animator.play(Animation::Walk, 4, 6);
                    }

                    if (not left and not right and animator.is(Animation::Walk)) {
                        animator.play(Animation::Idle, 2, 12);
                    }
                } break;
                case State::Airborne: {
                    const auto sensor_a = stage.sense(this, -WIDTH_RADIUS, HEIGHT_RADIUS, SensorDirection::Down);
                    const auto sensor_b = stage.sense(this,  WIDTH_RADIUS, HEIGHT_RADIUS, SensorDirection::Down);

                    const auto sensor = sensor_b.distance < sensor_a.distance
                        ? sensor_b
                        : sensor_a;

                    if (sensor.distance > -SNAP_DISTANCE and sensor.distance < SNAP_DISTANCE) {
                        position.y += sensor.distance;
                        state = State::Grounded;
                    }

                    position.y += 1;
                } break;
                case State::Jumping: {

                } break;
                case State::Death: {

                } break;
            }
        }

        void draw(Io& io, draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept override {
            target | draw::draw(
                stage.get_sheet().tile(animator.at(), 0)
                    | draw::apply_if(facing == Facing::Right, draw::mirror_x())
                    | draw::apply_if(character == Character::Bob, draw::map([] (Color c) -> Color {
                        if (c == Color::rgba(92, 230, 52)) return Color::rgba(76, 206, 220);
                        if (c == Color::rgba(252, 130, 116)) return Color::rgba(196, 118, 252);
                        return c;
                    })),
                -8, -8
            );
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

        static void serialize(Object const* erased, BinaryWriter& writer) {
            auto self = flat_cast<Player>(erased);
            writer.u8((u8) self->facing);
            writer.u8((u8) self->character);
        }

        static auto deserialize(BinaryReader& reader, i32 x, i32 y) -> Box<Object> {
            auto self = initialize(x, y).cast<Player>();

            self->facing = (Facing) reader.u8();
            self->character = (Character) reader.u8();

            return self;
        }
    };
}
