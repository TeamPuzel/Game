#include "ZenChan.hpp"
#include "../Player.hpp"

using namespace bubble;

void ZenChan::update(Io& io, rt::Input const& input, rt::SoundStage&, Stage& stage) noexcept {
    tick += 1;

    wrap_position();

    switch (state) {
        case State::Grounded: {
            const u32 hash = (tick * 13) + (i32(position.x) * 7);

            if ((hash % 256) < 12) {
                const auto sensor_a =
                    stage.sense(this, -WIDTH_RADIUS, HEIGHT_RADIUS - 8 * 5, SensorDirection::Down);
                const auto sensor_b =
                    stage.sense(this,  WIDTH_RADIUS, HEIGHT_RADIUS - 8 * 5, SensorDirection::Down);

                const auto sensor = sensor_b.distance < sensor_a.distance
                    ? sensor_b
                    : sensor_a;

                if (sensor.distance > -SNAP_DISTANCE and sensor.distance < SNAP_DISTANCE) {
                    for (auto obj : stage.objs()) {
                        if (flat_cast<Player>(obj) and obj->position.y < position.y) {
                            state = State::Jumping;
                            jump_lock = 30;
                        }
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

            if (sensor.distance > -SNAP_DISTANCE and sensor.distance < SNAP_DISTANCE) {
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

            if (sensor.distance > -SNAP_DISTANCE and sensor.distance < SNAP_DISTANCE) {
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

            if (not jump_lock and sensor.distance > -SNAP_DISTANCE and sensor.distance < SNAP_DISTANCE) {
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

EXPORT_GAME_OBJECT(ZenChan);
