#include "Player.hpp"

using namespace bubble;

void Food::update(Io& io, rt::Input const& input, rt::SoundStage& sound, Stage& stage) noexcept {
    tick += 1;
    position.y += 1;

    wrap_position();

    const auto sensor_a = stage.sense(this, -WIDTH_RADIUS, HEIGHT_RADIUS, SensorDirection::Down);
    const auto sensor_b = stage.sense(this,  WIDTH_RADIUS, HEIGHT_RADIUS, SensorDirection::Down);
    const auto sensor = sensor_b.distance < sensor_a.distance
        ? sensor_b
        : sensor_a;

    if (sensor.hit(3, 3)) {
        position.y += sensor.distance;
    }

    if (stage.solid_at((i32) position.x, (i32) position.y)) position.x += 1;

    if (not awarded) {
        for (auto obj : stage.objs()) {
            if (auto player = flat_cast<Player>(obj)) {
                if (math::abs(player->position.x - position.x) < 8 and math::abs(player->position.y - position.y) < 8) {
                    u32 points = point_value();
                    switch (player->character) {
                        case Player::Character::Bub: stage.award_points_bub(points); break;
                        case Player::Character::Bob: stage.award_points_bob(points); break;
                    }
                    sound.play(stage.get_sounds().get("sfx::pickup").clone());
                    awarded = true; stage.remove(this);
                    stage.add(Box<PointParticle>::make(position, points));
                }
            }
        }
    }
}

PlayerBubble::PlayerBubble(Box<Player> player) : held_player(std::move(player)) {
    position = held_player->position;
}

void PlayerBubble::update(Io& io, rt::Input const& input, rt::SoundStage& sound, Stage& stage) noexcept {
    tick += 1;

    position.x = math::floor(position.x);
    position.y = math::floor(position.y);

    if (stage.player_bubbles_should_move()) {
        StartPoint* destination = nullptr;

        auto chr_eq = [this] (StartPoint::Character c) {
            switch (held_player->character) {
                case Player::Character::Bub: return c == StartPoint::Character::Bub;
                case Player::Character::Bob: return c == StartPoint::Character::Bob;
            }
        };

        for (auto object : stage.objs()) {
            if (auto start_point = flat_cast<StartPoint>(object); start_point and chr_eq(start_point->character)) {
                destination = start_point;
            }
        }

        switch (destination->facing) {
            case StartPoint::Facing::Left:  held_player->facing = Player::Facing::Left;  break;
            case StartPoint::Facing::Right: held_player->facing = Player::Facing::Right; break;
        }

        if (destination->position.x != position.x) position.x += math::sign(destination->position.x - position.x);
        if (destination->position.y != position.y) position.y += math::sign(destination->position.y - position.y);

        if (destination->position == position and stage.done_transitioning()) {
            held_player->position = position;
            held_player->state = Player::State::Airborne;
            held_player->air_velocity = {};
            stage.add(std::move(held_player));
            stage.remove(this);
        }
    }
}

void PlayerBubble::draw(Io& io, draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept {
    if (not held_player) return;

    auto large_sheet = stage.get_sheet().inner | draw::grid(32, 32);

    target | draw::draw(
        large_sheet.tile(tick / 10 % 2 == 0 ? 0 : 1, 2)
            | draw::apply_if(held_player->character == Player::Character::Bob, draw::map([] (Color c) -> Color {
                if (c == Color::rgba(92, 230, 52)) return Color::rgba(76, 206, 220);
                if (c == Color::rgba(252, 130, 116)) return Color::rgba(196, 118, 252);
                return c;
            })),
        -16, -16
    );
}

void Bubble::launch_enemy(Player* player, rt::SoundStage& sound, Stage& stage, usize depth) {
    if (not held_enemy) return;

    held_enemy->position = position;

    EnemyParticle::Direction direction = math::sign(position.x - player->position.x) == -1
        ? EnemyParticle::Direction::Left
        : EnemyParticle::Direction::Right;

    stage.add(Box<EnemyParticle>::make(std::move(held_enemy), direction, depth));

    sound.play(stage.get_sounds().get("sfx::enemy_launch").clone());
}

void Bubble::update(Io& io, rt::Input const& input, rt::SoundStage& sound, Stage& stage) noexcept {
    tick += 1;
    wrap_position();

    if (launch_timer) launch_timer -= 1;

    if (tick == 60 * 30) return pop(sound, stage);

    if (launch_timer) {
        apply_launch();

        for (auto obj : stage.objs()) {
            if (auto enemy = flat_cast<Enemy>(obj)) {
                auto dx = position.x - enemy->position.x;
                auto dy = position.y - enemy->position.y;

                if (math::abs(dx) < WIDTH_RADIUS * 2 and math::abs(dy) < HEIGHT_RADIUS * 2) {
                    position = enemy->position;
                    held_enemy = stage.take(enemy);
                    launch_timer = 0;
                    break;
                }
            }
        }
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
                case Tile::Current::Solid: break;
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
                auto dx = position.x - player->position.x;
                auto dy = position.y - player->position.y;

                if (math::abs(dx) < WIDTH_RADIUS * 2 and math::abs(dy) < HEIGHT_RADIUS * 2) {
                    if (math::abs(dx) > math::abs(dy)) {
                        position.x += math::sign(dx);
                    } else {
                        if (
                            dy > 0 and
                            (player->state == Player::State::Airborne or player->state == Player::State::Jumping) and
                            player->air_velocity.y >= 0 and
                            player->get_input_jump(input)
                        ) {
                            position.y += 2; // TODO: Maybe this shouldn't be duplicated.
                            player->air_velocity.y = Player::JUMP_FORCE;
                            player->state = Player::State::Jumping;
                            player->jump_timer = Player::JUMP_DELAY;

                            sound.play(stage.get_sounds().get("sfx::jump").clone());
                        } else {
                            pop(player, sound, stage);
                        }
                    }
                }
            }
        }
    }

    if (stage.solid_at(this, -(WIDTH_RADIUS + 1), 0)) position.x += launch_timer ? 2 : 1;
    if (stage.solid_at(this,  (WIDTH_RADIUS + 1), 0)) position.x -= launch_timer ? 2 : 1;
}

void Bubble::pop(Player* player, rt::SoundStage& sound, Stage& stage, usize depth) {
    if (popped) return;
    popped = true;

    u32 points = point_value(depth);

    switch (player->character) {
        case Player::Character::Bub: stage.award_points_bub(points); break;
        case Player::Character::Bob: stage.award_points_bob(points); break;
    }

    stage.add(Box<PointParticle>::make(position, points));

    usize next_depth = held_enemy ? depth + 1 : depth;

    // Moves enemy out of this object.
    launch_enemy(player, sound, stage, depth);

    stage.remove(this); stage.add(Box<BubblePopParticle>::make(position));

    for (auto obj : stage.objs()) {
        if (auto other = flat_cast<Bubble>(obj); other and other != this) {
            if (other->popped) continue;

            auto dx = position.x - other->position.x;
            auto dy = position.y - other->position.y;

            if (math::abs(dx) < WIDTH_RADIUS * 3 and math::abs(dy) < HEIGHT_RADIUS * 3) {
                other->pop(player, sound, stage, next_depth);
            }
        }
    }
}

void Bubble::pop(rt::SoundStage& sound, Stage& stage) {
    if (popped) return;
    popped = true;

    if (held_enemy) {
        held_enemy->position = position;
        held_enemy->reset();
        stage.add(std::move(held_enemy))->provoke(this);
    }

    stage.remove(this); stage.add(Box<BubblePopParticle>::make(position));
}

void Bubble::pop_special(char match, rt::SoundStage& sound, Stage& stage) {
    if (popped) return;
    popped = true;

    if (held_enemy) {
        held_enemy->position = position;
        stage.add(std::move(held_enemy))->provoke(this);
    } else {
        using enum Food::Kind;

        Food::Kind kind; switch (match) {
            case '0': kind = FrenchFries;   break;
            case '1': kind = IceCream;      break;
            case '2': kind = Pudding;       break;
            case '3': kind = Hamburger;     break;
            case '4': kind = Shortcake;     break;
            case '5': kind = ChocolateCake; break;
            case '6': kind = Beer;          break;
            case '7': kind = Frankfurter;   break;
            case '8': kind = SoftIceCream;  break;
            case '9': kind = SodaPopIce;    break;
            default: throw std::logic_error(std::format("invalid special match: {}", match));
        }

        stage.add(Box<Food>::make(position, Food::Source::Clear, kind));
    }

    stage.remove(this); stage.add(Box<BubblePopParticle>::make(position));
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

    if (attack and not attack_timer and not (state == State::Death)) {
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

            if (sensor.hit(SNAP_DISTANCE_BACK, SNAP_DISTANCE_FORWARD)) {
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

            if (sensor.hit(SNAP_DISTANCE_BACK, SNAP_DISTANCE_FORWARD)) {
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

                if (sensor.hit(SNAP_DISTANCE_BACK, SNAP_DISTANCE_FORWARD)) {
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

    bool fast = math::abs(air_velocity.x) > fixed(0, 64);

    if (fast and stage.solid_at(this, -(WIDTH_RADIUS + 1), 0)) position.x += 1;
    if (fast and stage.solid_at(this,  (WIDTH_RADIUS + 1), 0)) position.x -= 1;
}

void Player::to_bubble(Stage& stage) noexcept {
    stage.add(Box<PlayerBubble>::make(stage.take(this)));
}

EXPORT_GAME_OBJECT(Player);
EXPORT_GAME_OBJECT(Bubble);
EXPORT_GAME_OBJECT(BubblePopParticle);
