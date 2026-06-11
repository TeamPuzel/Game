#pragma once
#include <bubble>
#include "meta/StartPoint.hpp"
#include "enemy/Enemy.hpp"

namespace bubble {
    class Player;

    class Food : public CodableObject<Food> {
      public:
        enum class Source : u8 {
            Timer,
            Enemy,
            Clear,
            Bonus
        } RELOAD source;

        enum class Kind : u32 {
            GreenPepper,
            Orange,
            Cucumber,
            Tomato,
            Watermelon,
            Pear,
            Banana,

            FrenchFries,
            IceCream,
            Pudding,
            Hamburger,
            Shortcake,
            ChocolateCake,
            Beer,
            Frankfurter,
            SoftIceCream,
            SodaPopIce
        } RELOAD kind;

        RELOAD usize tick = 0;
        RELOAD bool awarded = false;

        explicit Food(point<fixed> position, Source source, Kind kind) : source(source), kind(kind) {
            this->position = position;
        }

        static constexpr i32 WIDTH_RADIUS = 7;
        static constexpr i32 HEIGHT_RADIUS = 7;
        static constexpr i32 TRANSITION_DELAY = 8;

        struct Sprite final {
            i32 x, y;
        };

        struct PointValue final {
            u32 timer = 0;
            u32 enemy = 0;
            u32 clear = 0;
            u32 bonus = 0;
        };

        auto sprite_pos() const -> Sprite {
            using enum Kind;

            switch (kind) {
                case GreenPepper:   return { 0, 37 };
                case Orange:        return { 15, 37 };
                case Cucumber:      return { 7, 37 };
                case Tomato:        return { 32, 38 };
                case Watermelon:    return { 19, 37 };
                case Pear:          return { 18, 37 };
                case Banana:        return { 17, 37 };

                case FrenchFries:   return { 25, 37 };
                case IceCream:      return { 4, 38 };
                case Pudding:       return { 26, 37 };
                case Hamburger:     return { 27, 37 };
                case Shortcake:     return { 28, 37 };
                case ChocolateCake: return { 29, 37 };
                case Beer:          return { 0, 38 };
                case Frankfurter:   return { 5, 38 };
                case SoftIceCream:  return { 3, 38 };
                case SodaPopIce:    return { 2, 38 };
            }
        }

        auto raw_point_value() const -> PointValue {
            using enum Kind;

            switch (kind) {
                case GreenPepper:   return { .enemy = 1000 };
                case Orange:        return { .enemy = 2000 };
                case Cucumber:      return { .enemy = 4000 };
                case Tomato:        return { .enemy = 8000 };
                case Watermelon:    return { .enemy = 16000 };
                case Pear:          return { .enemy = 32000 };
                case Banana:        return { .enemy = 64000 };

                case FrenchFries:   return { .timer = 1000, .clear = 700 };
                case IceCream:      return { .timer = 1000, .clear = 700 };
                case Pudding:       return { .timer = 2000, .clear = 700 };
                case Hamburger:     return { .timer = 2000, .clear = 700 };
                case Shortcake:     return { .timer = 2000, .clear = 700 };
                case ChocolateCake: return { .timer = 3000, .clear = 700 };
                case Beer:          return { .timer = 4000, .clear = 700 };
                case Frankfurter:   return { .timer = 2000, .clear = 700 };
                case SoftIceCream:  return { .timer = 950,  .clear = 700 };
                case SodaPopIce:    return { .timer = 700,  .clear = 700 };
            }
        }

        auto point_value() const -> u32 {
            auto raw = raw_point_value();

            switch (source) {
                case Source::Timer: return raw.timer;
                case Source::Enemy: return raw.enemy;
                case Source::Clear: return raw.clear;
                case Source::Bonus: return raw.bonus;
            }
        }

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound, Stage& stage) noexcept override;

        void draw(Io& io, draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept override {
            auto [tx, ty] = sprite_pos();
            target | draw::draw(stage.get_sheet().tile(tx, ty), -8, -8);
        }

        auto prevents_transition() const noexcept -> bool override {
            return tick < 60 * TRANSITION_DELAY;
        }

        auto prevents_scoring() const noexcept -> bool override {
            return source == Source::Clear;
        }
    };

    /// An enemy particle wrapper which implements launched enemies and drops fruit when landing.
    class EnemyParticle : public CodableObject<EnemyParticle> {
      public:
        enum class Direction : u8 { Left, Right };
        RELOAD Box<Enemy> held_enemy;
        RELOAD point<fixed> velocity;
        RELOAD usize tick = 0;
        RELOAD usize depth;

        static constexpr fixed GRAVITY_FORCE = fixed(0, 12);
        static constexpr fixed LAUNCH_FORCE = 4;
        static constexpr fixed FALL_SPEED = 4;
        static constexpr fixed SPEED = 1;

        explicit EnemyParticle(Box<Enemy> enemy, Direction direction, usize depth)
            : held_enemy(std::move(enemy)), depth(depth)
        {
            this->position = held_enemy->position;

            switch (direction) {
                case Direction::Left:  velocity.x = -SPEED;
                case Direction::Right: velocity.x =  SPEED;
            }

            velocity.y = -LAUNCH_FORCE;
        }

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound, Stage& stage) noexcept override {
            tick += 1;

            velocity.y += GRAVITY_FORCE;
            velocity.y = std::clamp(velocity.y, -FALL_SPEED, FALL_SPEED);

            position += velocity;
            wrap_position();

            if (stage.solid_at((i32) position.x + 9, (i32) position.y)) velocity.x = -SPEED;
            if (stage.solid_at((i32) position.x - 9, (i32) position.y)) velocity.x =  SPEED;

            if (stage.solid_at((i32) position.x, (i32) position.y + 9) and velocity.y >= 1) {
                using enum Food::Kind;

                // This is exceptionally contradictory and poorly documented in the resources
                // I could find so this ordering is just random guesswork.
                Food::Kind kind; switch (depth) {
                    case 0:  kind = GreenPepper; break;
                    case 1:  kind = Orange;      break;
                    case 2:  kind = Cucumber;    break;
                    case 3:  kind = Tomato;      break;
                    case 4:  kind = Watermelon;  break;
                    case 5:  kind = Pear;        break;
                    default: kind = Banana;      break;
                }

                stage.add(Box<Food>::make(position, Food::Source::Enemy, kind));
                stage.remove(this);
            }
        }

        void draw(Io& io, draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept override {
            auto [tx, ty] = held_enemy->particle_sprite_pos();
            target | draw::draw(
                stage.get_sheet().tile(tx, ty) | draw::rotate(tick / 4),
                -8, -8
            );
        }

        auto prevents_transition() const noexcept -> bool override {
            return held_enemy and held_enemy->prevents_transition();
        }
    };

    /// A bubble used to transport the player between stages.
    class PlayerBubble : public CodableObject<PlayerBubble> {
      public:
        RELOAD usize tick = 0;
        RELOAD Box<Player> held_player;

        explicit PlayerBubble(Box<Player> player);
        void update(Io& io, rt::Input const& input, rt::SoundStage& sound, Stage& stage) noexcept override;
        void draw(Io& io, draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept override;
    };

    /// A particle effect that appears after a bubble pops.
    class BubblePopParticle : public CodableObject<BubblePopParticle> {
      public:
        static constexpr u32 TIMER_DELAY = 10;

        RELOAD u32 timer = TIMER_DELAY;

        BubblePopParticle() = default;

        BubblePopParticle(point<fixed> position) {
            this->position = position;
        }

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound, Stage& stage) noexcept override {
            if (timer) timer -= 1; else stage.remove(this);
        }

        void draw(Io& io, draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept override {
            target | draw::draw(
                stage.get_sheet().tile(timer > (TIMER_DELAY / 2) ? 5 : 6, 2),
                -8, -8
            );
        }
    };

    /// A particle effect that appears when obtaining points.
    class PointParticle : public CodableObject<PointParticle> {
      public:
        static constexpr u32 TIMER_DELAY = 60;
        static constexpr fixed SPEED = fixed(0, 128);

        RELOAD u32 timer = TIMER_DELAY;
        RELOAD u32 points;

        PointParticle(point<fixed> position, u32 points) : points(points) {
            this->position = position;
        }

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound, Stage& stage) noexcept override {
            if (timer) timer -= 1; else stage.remove(this);
            position.y -= SPEED;
        }

        void draw(Io& io, draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept override {
            target | draw::draw(
                draw::Text(std::format("{}", points), font::pico(io), draw::color::pico::YELLOW),
                draw::Origin::Bottom,
                draw::Origin::TopLeft
            );
        }

        auto prevents_transition() const noexcept -> bool override {
            return true;
        }
    };

    class Bubble : public CodableObject<Bubble> {
      public:
        enum class LaunchDirection : u8 { Left, Right } RELOAD launch_direction;
        RELOAD usize tick = 0;
        RELOAD u32 launch_timer = 25;
        RELOAD Box<Enemy> held_enemy;
        RELOAD bool popped = false;

        Bubble() = default;

        explicit Bubble(point<fixed> position, LaunchDirection launch_direction) : launch_direction(launch_direction) {
            this->position = position;
        }

        static constexpr fixed LAUNCH_SPEED = 3;
        static constexpr i32 WIDTH_RADIUS = 7;
        static constexpr i32 HEIGHT_RADIUS = 7;
        static constexpr u32 BASE_POINT_VALUE = 10;

        auto point_value(usize depth) const -> u32 {
            if (held_enemy) return held_enemy->point_value(depth); else return BASE_POINT_VALUE;
        }

        auto pop_delay() const -> u32 {
            return 60 * 30;
        }

        void launch_enemy(Player* player, rt::SoundStage& sound, Stage& stage, usize depth);

        void apply_launch() {
            switch (launch_direction) {
                case LaunchDirection::Left:  position.x -= LAUNCH_SPEED; break;
                case LaunchDirection::Right: position.x += LAUNCH_SPEED; break;
            }
        }

        virtual void pop(Player* player, rt::SoundStage& sound, Stage& stage, usize depth = 0);
        virtual void pop(rt::SoundStage& sound, Stage& stage);
        virtual void pop_special(char match, rt::SoundStage& sound, Stage& stage);

        void update(Io& io, rt::Input const& input, rt::SoundStage& sound, Stage& stage) noexcept override;

        void draw(Io& io, draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept override {
            auto decay_color_map = [&] (Color c) -> Color {
                if (c == Color::rgba(92, 230, 52)) {
                    if (tick >= 60 * 25 and tick / 3 % 2 == 0) return Color::rgba(76, 206, 220);
                    if (tick >= 60 * 20) return Color::rgba(180, 30, 124);
                }
                return c;
            };

            if (held_enemy) {
                u32 sprite_offset = 0;
                if (tick >= 60 * 20) sprite_offset += 1;
                if (tick >= 60 * 25) sprite_offset += 1;

                auto [tx, ty] = held_enemy->bubble_sprite_pos();
                target | draw::draw(
                    stage.get_sheet().tile(tx, ty)
                        | draw::apply_if(tick / 6 % 2 == 0, draw::mirror_x())
                        | draw::map(decay_color_map),
                    -8, -8
                );
            } else {
                auto tile = [&] {
                    if (launch_timer > 15) return stage.get_sheet().tile(1, 2);
                    if (launch_timer > 10) return stage.get_sheet().tile(2, 2);
                    if (launch_timer > 1)  return stage.get_sheet().tile(3, 2);
                    return stage.get_sheet().tile(4, 2);
                }();

                target | draw::draw(tile | draw::map(decay_color_map), -8, -8);
            }
        }

        auto prevents_transition() const noexcept -> bool override {
            return held_enemy and held_enemy->prevents_transition();
        }

        auto prevents_scoring() const noexcept -> bool override {
            return held_enemy and held_enemy->prevents_scoring();
        }
    };

    class Player : public CodableObject<Player> {
      public:
        friend class Bubble;

        enum class Animation {
            None,
            Idle,
            Walk,
            Attack,
            Death
        };

        RELOAD Animator<Animation> animator;

        enum class Facing : u8 { Left, Right } SERIAL facing = Facing::Right;
        enum class Character : u8 { Bub, Bob } SERIAL character = Character::Bub;

        enum class State : u8 {
            Grounded,
            Airborne,
            Jumping,
            Death
        } RELOAD state = State::Airborne;

        RELOAD usize tick = 0;
        RELOAD i32 attack_timer = 0;
        RELOAD i32 jump_timer = 0;
        RELOAD i32 death_timer = 0;
        RELOAD i32 invulnerability_timer = 0;
        RELOAD point<fixed> air_velocity;
        RELOAD bool standing_jump = false;
        RELOAD Facing jump_direction = Facing::Right;

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
        static constexpr i32 SNAP_DISTANCE_BACK = 5;
        static constexpr i32 SNAP_DISTANCE_FORWARD = 2;
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

        virtual void move_to_start_point(Stage const& stage) {
            point<fixed> destination = { 64, 64 }; // Sane fallback.

            auto chr_eq = [this] (StartPoint::Character c) {
                switch (character) {
                    case Player::Character::Bub: return c == StartPoint::Character::Bub;
                    case Player::Character::Bob: return c == StartPoint::Character::Bob;
                }
            };

            for (auto obj : stage.objs()) {
                if (auto start_point = flat_cast<StartPoint>(obj); start_point and chr_eq(start_point->character)) {
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

        virtual void lose_life(Stage& stage) {
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

        virtual void to_bubble(Stage& stage) noexcept;
    };

    template <> struct FallbackCoder<Player> {
        static void deserialize(Box<Player>& self, BinaryReader& reader) {
            self->facing = (Player::Facing) reader.u8();
            self->character = (Player::Character) reader.u8();
        }
    };
}
