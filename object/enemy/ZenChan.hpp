#pragma once
#include <bubble>

namespace bubble {
    class ZenChan final : public Object, public DefaultCodable<ZenChan> {
      public:
        enum class Facing : u8 { Left, Right } facing = Facing::Right;
        u8 jump_lock = 0;

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
        } state = State::Airborne;

        usize tick = 0;

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

        void flip() noexcept override {
            switch (facing) {
                case Facing::Left:  facing = Facing::Right; break;
                case Facing::Right: facing = Facing::Left;  break;
            }
        }

        static void serialize(Object const* erased, BinaryWriter& writer) {
            auto self = flat_cast<ZenChan>(erased);
            writer.u8((u8) self->facing);
        }

        static auto deserialize(BinaryReader& reader, i32 x, i32 y) -> Box<Object> {
            auto self = initialize(x, y).cast<ZenChan>();

            self->facing = (Facing) reader.u8();

            return self;
        }
    };
}
