#include <primitive>
#include <draw>
#include <io>
#include <rt>
#include <bubble>

class Game final {
    Box<bubble::Scene> scene;

  public:
    Game() {

    }

    void init(Io& io) {
        scene = bubble::Stage::load(io, "1-1.stage");
    }

    void update(Io& io, rt::Input const& input) {
        scene->update(io, input);
    }

    void draw(Io& io, rt::Input const& input, draw::Ref<draw::Image> target) const {
        scene->draw(io, input, target);
    }
};

auto main() -> i32 {
    Game instance;
    rt::run(instance, "Game", 800, 600, 4);
}
