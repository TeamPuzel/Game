// Created by Lua (TeamPuzel) on May 26th 2025.
// Copyright (c) 2025 All rights reserved.
//
// This header defines game object functionality.
#pragma once
#include <primitive>
#include <math>
#include <io>
#include <rt>
#include <font>

namespace bubble {
    using draw::Image;
    using draw::Ref;
    using draw::Color;
    using draw::Text;
    using math::point;

    using BinaryReader = io::BinaryReader<std::span<const u8>>;
    using BinaryWriter = io::BinaryWriter<std::back_insert_iterator<std::vector<u8>>>;

    class Stage;

    /// A dynamic game object.
    class Object {
      public:
        virtual auto is_dynobject() const -> bool { return false; }

        virtual auto classname() const -> std::string_view {
            throw std::logic_error("only dynamic objects have classnames");
        }

        virtual auto is_serial() const -> bool { return false; }

        // TODO: Use generic math::Vector<fixed, 2> matrix type once it's adjusted to allow non-float element types.
        //       For now this old sonic::point type will work.
        point<fixed> position;

        Object() = default;
        Object(Object const&) = delete;
        Object(Object&&) = delete;
        auto operator=(Object const&) -> Object& = delete;
        auto operator=(Object&&) -> Object& = delete;
        virtual ~Object() noexcept {}

        auto isa(std::string_view name) const noexcept -> bool {
            return classname() == name;
        }

        void wrap_position() noexcept {
            static constexpr i32 screen_width = 32 * 8;                        // 256 px
            static constexpr i32 screen_height = 30 * 8;                       // 240 px
            static constexpr i32 top_margin = 4 * 8;                           // 32  px
            // static constexpr i32 top_margin = 0;                               // 0   px
            static constexpr i32 playable_height = screen_height - top_margin; // 208 px

            // Horizontal wrap.
            if (position.x < 0) {
                position.x += screen_width;
            } else if (position.x >= screen_width) {
                position.x -= screen_width;
            }

            // Vertical wrap.
            if (position.y < top_margin) {
                position.y += playable_height;
            } else if (position.y >= screen_height) {
                position.y -= playable_height;
            }
        }

        /// Called once every tick at 60hz carefully paced in sync with the display clock.
        /// The delta time is effectively constant and can be left out.
        virtual void update(Io& io, rt::Input const& input, rt::SoundStage& sound, Stage& stage) noexcept {}

        /// Called to draw the object with a target slice offset from the screen by the object position.
        ///
        /// The provided target slice retains the width and height of the scene target, so for objects at the origin
        /// it effectively wraps the scene target transparently in a slice, preserving its category.
        virtual void draw(Io& io, draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept {}

        auto pixel_pos() const noexcept -> math::point<i32> {
            return math::point { i32(position.x), i32(position.y) };
        }

        /// A request to flip horizontally.
        virtual void flip() noexcept {}

        /// An arbitrary editor toggle.
        virtual void alternate() noexcept {}

        /// This is really just a horrible hack because other than macOS dynamic_cast just does not
        /// work across shared libraries and I can't implement a proper portable cast without
        /// C++26 reflection to form the inheritance metadata.
        /// The reflection dependency must only be used for development features to maintain compatibility
        /// with the stripped down MSVC builds.
        virtual auto prevents_transition() const noexcept -> bool {
            return false;
        }

        virtual auto prevents_scoring() const noexcept -> bool {
            return false;
        }
    };

    template <typename T, typename U> [[deprecated]] auto flat_cast(U* ptr) noexcept -> T* {
        return dynamic_cast<T*>(ptr);
        // if (auto result = dynamic_cast<T*>(ptr)) return result;
        // std::string_view t = typeid(T).name(), u = typeid(*ptr).name();
        // if (t == u) return (T*) ptr; else return nullptr;
    }

    template <typename T, typename U> [[deprecated]] auto flat_cast(U const* ptr) noexcept -> T const* {
        return dynamic_cast<T const*>(ptr);
        // if (auto result = dynamic_cast<T const*>(ptr)) return result;
        // std::string_view t = typeid(T).name(), u = typeid(*ptr).name();
        // if (t == u) return (T const*) ptr; else return nullptr;
    }

    template <typename T> auto isa_cast(std::string_view isa, Object* ptr) noexcept -> T* {
        if (ptr->isa(isa)) return (T*) ptr; else return nullptr;
    }

    template <typename T> auto isa_cast(std::string_view isa, Object const* ptr) noexcept -> T const* {
        if (ptr->isa(isa)) return (T const*) ptr; else return nullptr;
    }

    /// A game object loadable from files and hot-reloadable during gameplay.
    /// Obviously don't attempt rebuilding if the ABI was broken between reloads.
    ///
    /// trait SerializableObject {
    ///     static rebuild(Self const*);
    ///     static serialize(Self const&, BinaryWriter&);
    ///     static deserialize(BinaryReader&) -> Self;
    /// }
    template <typename, typename = void> struct DynamicObject : std::false_type {};
    template <typename Self> struct DynamicObject<Self, std::enable_if_t<
        std::is_same<decltype(Self::rebuild(std::declval<Self*>(), std::declval<Stage&>())), Box<Object>>::value and
        std::is_same<decltype(Self::serialize(std::declval<Self const*>(), std::declval<BinaryWriter&>())), void>::value and
        std::is_same<decltype(Self::deserialize(std::declval<BinaryReader&>(), std::declval<i32>(), std::declval<i32>())), Box<Object>>::value and
        std::is_same<decltype(Self::initialize(std::declval<i32>(), std::declval<i32>())), Box<Object>>::value
    >> : std::true_type {};

    using ObjectRebuilder    = auto (*) (Object*, Stage& stage) -> Box<Object>;
    using ObjectSerializer   = auto (*) (Object const*, BinaryWriter&) -> void;
    using ObjectDeserializer = auto (*) (BinaryReader&, i32 x, i32 y) -> Box<Object>;
    using ObjectInitializer  = auto (*) (i32 x, i32 y) -> Box<Object>;

    // /// A game object loadable from files and hot-reloadable during gameplay.
    // /// Obviously don't attempt rebuilding if the ABI was broken between reloads.
    // template <typename Self> concept DynamicObject = requires(
    //     Object const& self, rt::BinaryReader& r, rt::BinaryWriter& w, i32 x, i32 y
    // ) {
    //     { &Self::rebuild } -> std::same_as<ObjectRebuilder>;
    //     { &Self::serialize } -> std::same_as<ObjectSerializer>;
    //     { &Self::deserialize } -> std::same_as<ObjectDeserializer>;
    // };
}
