#include "controls.hpp"
#include "title.hpp"

using namespace bubble;

void Controls::return_to_title(Io& io) {
    transition(Box<Title>::make(io, std::move(sheet), std::move(sounds)));
}
