#pragma once
#include <bubble>
#include "meta/StartPoint.hpp"

namespace bubble {
    class Bubble final : public Object {
      public:
        enum class LaunchDirection : u8 { Left, Right } launch_direction;
        u32 launch_timer = 25;
        u32 pop_timer = 0;
        Box<Object> held_object;

        explicit Bubble(point<fixed> position, LaunchDirection launch_direction) : launch_direction(launch_direction) {
            this->position = position;
        }

        static constexpr fixed LAUNCH_SPEED = 3;
        static constexpr i32 WIDTH_RADIUS = 7;
        static constexpr i32 HEIGHT_RADIUS = 7;

        auto is_popped() const -> bool {
            return pop_timer;
        }

        void apply_launch() {
            switch (launch_direction) {
                case LaunchDirection::Left:  position.x -= LAUNCH_SPEED; break;
                case LaunchDirection::Right: position.x += LAUNCH_SPEED; break;
            }
        }

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound, Stage& stage) noexcept override;

        void draw(Io& io, draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept override {
            auto tile = [&] {
                if (launch_timer > 15) return stage.get_sheet().tile(1, 2);
                if (launch_timer > 10) return stage.get_sheet().tile(2, 2);
                if (launch_timer > 1)  return stage.get_sheet().tile(3, 2);
                return stage.get_sheet().tile(4, 2);
            }();

            target | draw::draw(tile, -8, -8);
        }
    };

    class Player final : public Object, public DefaultCodable<Player> {
      public:
        enum class Animation {
            None,
            Idle,
            Walk,
            Attack,
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

        usize tick = 0;
        i32 attack_timer = 0;
        i32 jump_timer = 0;
        i32 death_timer = 0;
        i32 invulnerability_timer = 0;
        point<fixed> air_velocity;
        bool standing_jump = false;
        Facing jump_direction = Facing::Right;

        static constexpr fixed FALL_SPEED = 1;
        static constexpr fixed AIR_SPEED = 1;
        static constexpr fixed RESTRICTED_AIR_SPEED = fixed(0, 128);
        static constexpr fixed SPEED = 1;
        static constexpr fixed JUMP_FORCE = -3;
        static constexpr fixed GRAVITY_FORCE = fixed(0, 28);
        static constexpr fixed AIR_CONTROL_FORCE = fixed(0, 12);
        static constexpr fixed AIR_FRICTION = fixed(0, 8);
        static constexpr fixed FREE_AIR_FRICTION = fixed(0, 16);
        static constexpr i32 WIDTH_RADIUS = 7;
        static constexpr i32 HEIGHT_RADIUS = 7;
        static constexpr i32 SNAP_DISTANCE = 2;
        static constexpr i32 ATTACK_DELAY = 30;
        static constexpr i32 ATTACK_ANIMATION_TRIM = 10;
        static constexpr i32 JUMP_DELAY = 30;
        static constexpr i32 DEATH_DELAY = 60;
        static constexpr i32 INVULNERABILITY_DELAY = 60 * 4;

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

        auto bubble_launch_direction() const -> Bubble::LaunchDirection {
            switch (facing) {
                case Facing::Left:  return Bubble::LaunchDirection::Left;
                case Facing::Right: return Bubble::LaunchDirection::Right;
            }
        }

      public:
        void update(Io& io, rt::Input const& input, rt::SoundStage& sound, Stage& stage) noexcept override;

        void draw(Io& io, draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept override {
            auto sprite = [&] {
                if (state == State::Death)
                    return stage.get_sheet().tile(animator.at() + (death_timer <= 10 ? 4 : 0), 3);

                if (std::max(0, attack_timer - ATTACK_ANIMATION_TRIM)) return stage.get_sheet().tile(0, 2);

                return stage.get_sheet().tile(animator.at(), 0);
            }();

            target | draw::draw(
                sprite
                    | draw::apply_if(facing == Facing::Right, draw::mirror_x())
                    | draw::apply_if(invulnerability_timer, draw::map([this] (Color c) -> Color {
                        return tick % 2 == 0 ? c : draw::color::CLEAR;
                    }))
                    | draw::apply_if(character == Character::Bob, draw::map([] (Color c) -> Color {
                        if (c == Color::rgba(92, 230, 52)) return Color::rgba(76, 206, 220);
                        if (c == Color::rgba(252, 130, 116)) return Color::rgba(196, 118, 252);
                        return c;
                    })),
                -8, -8
            );
        }

        void move_to_start_point(Stage const& stage) {
            point<fixed> destination = { 64, 64 }; // Sane fallback.

            for (auto obj : stage.objs()) {
                if (auto start_point = flat_cast<StartPoint>(obj)) {
                    destination = start_point->position;

                    switch (start_point->facing) {
                        case StartPoint::Facing::Left:  facing = Facing::Left; break;
                        case StartPoint::Facing::Right: facing = Facing::Right; break;
                    }

                    break;
                }
            }

            position = destination;
        }

        void lose_life(Stage& stage) {
            switch (character) {
                case Character::Bub: (&stage)->lose_life_bub(); break;
                case Character::Bob: (&stage)->lose_life_bob(); break;
            }
        }

        void damage() noexcept {
            if (state != State::Death and not invulnerability_timer) {
                death_timer = DEATH_DELAY;
                state = State::Death;
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
