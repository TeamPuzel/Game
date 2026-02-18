#include <primitive>
#include <draw>
#include <io>
#include <rt>

// TODO: Rewrite serializable game object and component (trait) engine from sonic namespace
//       to avoid rewriting it from scratch yet again.
class Game final {
  public:
    Game() {

    }

    void init(Io& io) {

    }

    void update(Io& io, rt::Input const& input) {

    }

    void draw(Io& io, rt::Input const& input, draw::Ref<draw::Image> target) const {
        target | draw::clear();
    }
};

auto main() -> i32 {
    Game instance;
    rt::run(instance, "Presenter", 800, 600, 4);
}
