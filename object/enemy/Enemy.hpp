#pragma once
#include <bubble>

namespace bubble {
    class Enemy : public Object {
      public:
        struct BubbleSpritePosition final { i32 x, y; };

        virtual auto bubble_sprite_pos() const -> BubbleSpritePosition = 0;

        virtual auto point_value(usize depth) const -> u32 {
            switch (depth) {
                case 0:  return 1000;
                case 1:  return 2000;
                case 2:  return 4000;
                case 3:  return 8000;
                case 4:  return 16000;
                case 5:  return 32000;
                default: return 64000;
            }
        }
    };
}
