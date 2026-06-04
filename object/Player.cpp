#include "Player.hpp"

using namespace bubble;

void Bubble::update(Io& io, rt::Input const& input, rt::SoundStage& sound, Stage& stage) noexcept {
    wrap_position();

    if (launch_timer) launch_timer -= 1;

    if (stage.solid_at(this, -(WIDTH_RADIUS + 1), 0)) position.x += launch_timer ? 2 : 1;
    if (stage.solid_at(this,  (WIDTH_RADIUS + 1), 0)) position.x -= launch_timer ? 2 : 1;

    if (launch_timer) {
        apply_launch();
    } else {
        auto [x, y] = pixel_pos();
        auto tile = stage.tile_at(x, y);

        if (tile and input.counter() % 2 == 0) {
            auto current = (Tile::Current) tile->current;

            switch (current) {
                case Tile::Current::Up:    position.y -= 1; break;
                case Tile::Current::Down:  position.y += 1; break;
                case Tile::Current::Left:  position.x -= 1; break;
                case Tile::Current::Right: position.x += 1; break;
                case Tile::Current::Solid: position.y -= 1; break;
            }
        }

        if (not tile and input.counter() % 2 == 0) {
            position.x += 1;
        }

        for (auto obj : stage.objs()) {
            if (auto other = flat_cast<Bubble>(obj); other and other != this) {
                // Calculate distance between bubbles.
                auto dx = position.x - other->position.x;
                auto dy = position.y - other->position.y;

                if (math::abs(dx) < WIDTH_RADIUS * 2 and math::abs(dy) < HEIGHT_RADIUS * 2) {
                    if (dx != 0) position.x += math::sign(dx);
                    if (dy != 0) position.y += math::sign(dy);

                    // Unstuck perfectly overlapping bubbles.
                    if (dx == 0 and dy == 0) position.x += 1;
                }
            } else if (auto player = flat_cast<Player>(obj)) {

            }
        }
    }
}

void Player::update(Io& io, rt::Input const& input, rt::SoundStage& sound, Stage& stage) noexcept {
    tick += 1;
    animator.update();

    wrap_position();

    if (attack_timer != 0) attack_timer -= 1;
    if (jump_timer != 0) jump_timer -= 1;
    if (death_timer != 0) death_timer -= 1;
    if (invulnerability_timer != 0) invulnerability_timer -= 1;

    if (animator.is(Animation::None)) {
        animator.play(Animation::Idle, 2, 12);
    }

    bool left = get_input_left(input);
    bool right = get_input_right(input);
    bool attack = get_input_attack(input);
    bool jump = get_input_jump(input);

    if (left and not right) facing = Facing::Left;
    if (right and not left) facing = Facing::Right;

    if (attack and not attack_timer) {
        attack_timer = ATTACK_DELAY;
        sound.play(stage.get_sounds().get("sfx::launch").clone());
        stage.add(Box<Bubble>::make(position, bubble_launch_direction()));
    }

    if (jump and not jump_timer and state == State::Grounded) {
        jump_timer = JUMP_DELAY;
        sound.play(stage.get_sounds().get("sfx::jump").clone());

        air_velocity.y = JUMP_FORCE;
        state = State::Jumping;

        if (left and not right) standing_jump = false, air_velocity.x = -AIR_SPEED;
        if (right and not left) standing_jump = false, air_velocity.x = +AIR_SPEED;
        if (left == right) standing_jump = true, air_velocity.x = 0;

        jump_direction = facing;
    }

    auto air_control = [&] (fixed speed_cap) {
        auto friction = left == right
            ? FREE_AIR_FRICTION
            : AIR_FRICTION;

        if (left and not right) air_velocity.x -= AIR_CONTROL_FORCE;
        if (right and not left) air_velocity.x += AIR_CONTROL_FORCE;
        if (left == right and math::abs(air_velocity.x) > friction)
            air_velocity.x -= math::sign(air_velocity.x) * friction;

        air_velocity.x = std::clamp(air_velocity.x, -speed_cap, speed_cap);
    };

    switch (state) {
        case State::Grounded: {
            if (left and not right) position.x -= 1;
            if (right and not left) position.x += 1;

            if (left or right) {
                animator.play(Animation::Walk, 4, 6);
            }

            if (not left and not right) {
                animator.play(Animation::Idle, 2, 12);
            }

            const auto sensor_a = stage.sense(this, -WIDTH_RADIUS, HEIGHT_RADIUS, SensorDirection::Down);
            const auto sensor_b = stage.sense(this,  WIDTH_RADIUS, HEIGHT_RADIUS, SensorDirection::Down);
            const auto sensor = sensor_b.distance < sensor_a.distance
                ? sensor_b
                : sensor_a;

            if (sensor.distance > -SNAP_DISTANCE and sensor.distance < SNAP_DISTANCE) {
                position.y += sensor.distance;
            } else {
                air_velocity = { 0, 0 };
                state = State::Airborne;
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

            air_control(RESTRICTED_AIR_SPEED);

            position += air_velocity;
            position.y += 1;
        } break;
        case State::Jumping: {
            position += air_velocity;

            air_velocity.y += GRAVITY_FORCE;

            if (air_velocity.y > FALL_SPEED) air_velocity.y = FALL_SPEED;

            air_control(standing_jump ? RESTRICTED_AIR_SPEED : AIR_SPEED);

            if (
                math::abs(air_velocity.x) < fixed(0, 16) and
                (
                    left == right or
                    left and jump_direction == Facing::Right or
                    right and jump_direction == Facing::Left
                )
            ) standing_jump = true;

            if (not jump_timer) {
                const auto sensor_a = stage.sense(this, -WIDTH_RADIUS, HEIGHT_RADIUS, SensorDirection::Down);
                const auto sensor_b = stage.sense(this,  WIDTH_RADIUS, HEIGHT_RADIUS, SensorDirection::Down);

                const auto sensor = sensor_b.distance < sensor_a.distance
                    ? sensor_b
                    : sensor_a;

                if (sensor.distance > -SNAP_DISTANCE and sensor.distance < SNAP_DISTANCE) {
                    position.y += sensor.distance;
                    state = State::Grounded;
                }
            }
        } break;
        case State::Death: {
            if (death_timer == DEATH_DELAY - 1) sound.play(stage.get_sounds().get("sfx::death").clone());
            animator.play(Animation::Death, 4, 4);
            if (death_timer == 10) animator.replay(Animation::Death, 2, 4);

            if (death_timer == 0) {
                invulnerability_timer = INVULNERABILITY_DELAY;
                move_to_start_point(stage);
                lose_life(stage);
                state = State::Airborne;
            }
        } break;
    }

    if (stage.super_solid_at(this, -(WIDTH_RADIUS + 1), 0)) position.x += 1;
    if (stage.super_solid_at(this,  (WIDTH_RADIUS + 1), 0)) position.x -= 1;
}

EXPORT_GAME_OBJECT(Player);
