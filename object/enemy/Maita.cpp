#include "Maita.hpp"
#include "../Player.hpp"

using namespace bubble;

void Maita::update(Io& io, rt::Input const& input, rt::SoundStage& sound, Stage& stage) noexcept {
    tick += 1;

    wrap_position();

    for (auto obj : stage.objs()) {
        if (auto player = flat_cast<Player>(obj)) {
            if (math::abs(player->position.x - position.x) < 8 and math::abs(player->position.y - position.y) < 8)
                player->damage();
        }
    }

    switch (state) {
        case State::Grounded: {
            const u32 hash = std::max<u32>(1, (tick * 13) + (i32(position.x) * 7) + usize(this));

            if ((hash % 256) < 12) {
                const auto sensor_a =
                    stage.sense(this, -WIDTH_RADIUS, HEIGHT_RADIUS - 8 * 5, SensorDirection::Down);
                const auto sensor_b =
                    stage.sense(this,  WIDTH_RADIUS, HEIGHT_RADIUS - 8 * 5, SensorDirection::Down);

                const auto sensor = sensor_b.distance < sensor_a.distance
                    ? sensor_b
                    : sensor_a;

                if (
                    sensor.hit(SNAP_DISTANCE_BACK, SNAP_DISTANCE_FORWARD) and
                    not stage.solid_at((i32) position.x, (i32) position.y - HEIGHT_RADIUS - 8 * 5) and
                    (position.y - (HEIGHT_RADIUS - 8 * 5)) > (8 * 4) // Do not jump out of bounds.
                ) {
                    for (auto obj : stage.objs()) {
                        if (flat_cast<Player>(obj) and obj->position.y < position.y) {
                            state = State::Jumping;
                            jump_lock = 30;
                        }
                    }
                }
            } else if ((hash % 256) < 16) {
                for (auto obj : stage.objs()) {
                    if (
                        flat_cast<Player>(obj) and obj->position.y <= position.y and
                        (
                            obj->position.x < position.x and facing == Facing::Right or
                            obj->position.x > position.x and facing == Facing::Left
                        )
                    ) {
                        flip();
                    }
                }
            }

            const auto sensor_w = stage.sense(this, facing_direction());

            if (sensor_w.distance < WIDTH_RADIUS + 2) {
                flip();
            } else {
                walk_forward();
            }

            const auto sensor_a = stage.sense(this, -WIDTH_RADIUS, HEIGHT_RADIUS, SensorDirection::Down);
            const auto sensor_b = stage.sense(this,  WIDTH_RADIUS, HEIGHT_RADIUS, SensorDirection::Down);
            const auto sensor = sensor_b.distance < sensor_a.distance
                ? sensor_b
                : sensor_a;

            if (sensor.hit(SNAP_DISTANCE_BACK, SNAP_DISTANCE_FORWARD)) {
                position.y += sensor.distance;
            } else {
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

            position.y += 1;
        } break;
        case State::Jumping: {
            const auto sensor_a = stage.sense(this, -WIDTH_RADIUS, HEIGHT_RADIUS, SensorDirection::Down);
            const auto sensor_b = stage.sense(this,  WIDTH_RADIUS, HEIGHT_RADIUS, SensorDirection::Down);

            const auto sensor = sensor_b.distance < sensor_a.distance
                ? sensor_b
                : sensor_a;

            if (not jump_lock and sensor.hit(SNAP_DISTANCE_BACK, SNAP_DISTANCE_FORWARD)) {
                position.y += sensor.distance;
                state = State::Grounded;
            }

            if (jump_lock != 0) jump_lock -= 1;

            if (jump_lock < 10) position.y -= 1;
        } break;
        case State::Leaping: {

        } break;
    }
}

EXPORT_GAME_OBJECT(Maita);
