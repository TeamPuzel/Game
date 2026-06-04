#include "stage.hpp"
#include "../object/Player.hpp"

using namespace bubble;

void Stage::lose_life_bub() {
    if (not bub_lives)
        for (auto obj : objs())
            if (auto p = isa_cast<Player>("Player", obj); p and p->character == Player::Character::Bub)
                remove(obj);

    bub_lives = std::max(0, (i32) bub_lives - 1);

    should_check_for_game_end = true;
}

void Stage::lose_life_bob() {
    if (not bob_lives)
        for (auto obj : objs())
            if (auto p = isa_cast<Player>("Player", obj); p and p->character == Player::Character::Bob)
                remove(obj);

    bob_lives = std::max(0, (i32) bob_lives - 1);

    should_check_for_game_end = true;
}

void Stage::check_for_game_end() {
    if (not std::ranges::any_of(objs(), [] (auto p) -> bool { return isa_cast<Player>("Player", p); })) {
        game_end_timer = GAME_END_DELAY;
    }
}
