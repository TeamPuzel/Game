#include "scoreboard.hpp"
#include "title.hpp"

using namespace bubble;

void ScoreBoard::return_to_title(Io& io) {
    transition(Box<Title>::make(io, std::move(sheet)));
}
