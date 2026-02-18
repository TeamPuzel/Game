#include <renderer>
#include <primitive>
#include <draw>
#include <io>
#include <rt>
#include <format>

namespace nes {
    constexpr usize WIDTH = 256, HEIGHT = 240;

    constexpr draw::Color C0 = draw::Color::gray(0);
    constexpr draw::Color C1 = draw::Color::gray(102);
    constexpr draw::Color C2 = draw::Color::gray(187);
    constexpr draw::Color C3 = draw::Color::gray(255);
}

class Slide {
  public:
    using Target = draw::Slice<draw::Ref<draw::Image>>;
    using Sheet  = draw::Grid<draw::Ref<const draw::Image>>;

    std::optional<Sheet> sheet;

    virtual void update(Io& io, rt::Input const& input) {}
    virtual void draw(Io& io, rt::Input const& input, Target target) const = 0;
    virtual ~Slide() noexcept {}
};

class TitleSlide final : public Slide {
    std::string title;

  public:
    TitleSlide(std::string title) : title(title) {}

    void draw(Io& io, rt::Input const& input, Target target) const override {
        const auto txt = draw::Text(title, font::pod(), draw::color::pico::RED);
        const auto text = txt | draw::as_ref() | draw::scale(2);

        target | draw::draw(
            text,
            (target.width() - text.width()) / 2,
            (target.height() - text.height()) / 2
        );
    }
};

class ContentSlide final : public Slide {
    std::string title, content;

  public:
    ContentSlide(std::string title, std::string content) : title(title), content(content) {}

    void draw(Io& io, rt::Input const& input, Target target) const override {
        target | draw::draw(
            draw::Text(title, font::pod(), draw::color::pico::RED)
                | draw::as_ref()
                | draw::scale(2),
            8,
            8
        );
        target | draw::draw(draw::WrappingText(content, font::pod(), target.width() - 16), 8, 32);
    }
};

class Presenter final {
    std::vector<Box<Slide>> slides;
    usize current_slide { 0 };
    std::optional<draw::Image> source;
    std::optional<Slide::Sheet> sheet;

  public:
    Presenter() {
        slides.emplace_back(Box<TitleSlide>::make(
            "The Creature of Wellstown"
        ));

        slides.emplace_back(Box<ContentSlide>::make(
            "A Slide",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
            "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut "
            "aliquip ex ea commodo consequat.\n\n"
            "Duis aute irure dolor in reprehenderit "
            "in voluptate velit esse cillum dolore eu fugiat nulla pariatur"
        ));

        slides.emplace_back(Box<ContentSlide>::make(
            "Another Slide",
            ":)"
        ));
    }

    void init(Io& io) {
        source.emplace(draw::TgaImage::from(io.read_file("res/sheet.tga")) | draw::flatten<draw::Image>());
        sheet.emplace(
            *source
                | draw::as_const()
                | draw::as_ref()
                | draw::grid(16, 16)
        );

        for (auto& slide : slides) {
            slide->sheet.emplace(*sheet);
        }
    }

    void update(Io& io, rt::Input const& input) {
        if (input.key_pressed(rt::Key::Right) and not input.key_held(rt::Key::Left)) {
            if (slides.size() > current_slide + 1) current_slide += 1;
        }

        if (input.key_pressed(rt::Key::Left) and not input.key_held(rt::Key::Right)) {
            if (slides.size() > current_slide - 1) current_slide -= 1;
        }
    }

    void draw(Io& io, rt::Input const& input, draw::Ref<draw::Image> target) const {
        target | draw::clear();

        // target | draw::draw(sheet->tile(0, 0), 0, 0);
        // target | draw::draw(sheet->tile(0, 1).resize_right(16 * 16), 0, 16);

        if (slides.size() > current_slide) {
            slides.at(current_slide)->draw(
                io,
                input,
                target | draw::as_slice()
                // target | draw::slice(
                //     (target.width() - nes::WIDTH) / 2,
                //     (target.height() - nes::HEIGHT) / 2,
                //     nes::WIDTH,
                //     nes::HEIGHT
                // )
            );
        }

        const auto slide_text = draw::Text(std::format("Slide: {} of {}", current_slide + 1, slides.size()), font::pod());
        target | draw::draw(slide_text, 8, target.height() - slide_text.height() - 8);
    }
};

auto main() -> i32 {
    Presenter instance;
    rt::run(instance, "Presenter", 800, 700, 4);
}
