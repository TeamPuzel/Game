#include <primitive>
#include <draw>
#include <io>
#include <rt>
#include <bubble>

static std::atomic<bool> RELOAD_REQUESTED = false;
static std::atomic<bool> DID_PRELOAD = false;

#if defined(__APPLE__) || defined(__linux__)
#include <signal.h>
void reload_handler(i32 signal) {
    if (signal == SIGUSR1) RELOAD_REQUESTED = true;
}
#endif

class Game final {
    Box<bubble::Scene> scene;

  public:
    Game() {}

    void init(Io& io) {
        bubble::Scene::unsafe_set_root_ptr(&scene);
        scene = Box<bubble::Title>::make(io);
        // scene = Box<bubble::ScoreBoard>::make(io, std::queue<bubble::ScoreBoard::PendingScore>({
        //     { .character = bubble::ScoreBoard::Character::Bub, .score = 1000 }
        // }));
    }

    void deinit(Io& io) {
        scene.erase();
        bubble::class_loader::clear();
    }

    void update(Io& io, rt::Input const& input, rt::SoundStage& sound) {
        bubble::Scene::unsafe_apply_transition();

        if (not DID_PRELOAD) {
            bubble::Stage::preload_object_classes(io);
            DID_PRELOAD = true;
        }

        if (RELOAD_REQUESTED) {
            scene->hot_reload(io);
            RELOAD_REQUESTED = false;
        }

        scene->update(io, input, sound);
    }

    void draw(Io& io, rt::Input const& input, draw::Ref<draw::Image> target) const {
        scene->draw(io, input, target);
    }
};

auto main() -> i32 {
#if defined(__APPLE__) || defined(__linux__)
    struct sigaction sa;
    sa.sa_handler = reload_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART; // restart syscalls after signal
    sigaction(SIGUSR1, &sa, NULL);
#endif
    rt::run(Game(), "Bubble Bobble DX",
        32 * 8 * 2, // NES tiles to window size.
        30 * 8 * 2,
        2
    );
}
