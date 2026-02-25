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
        friend class Stage;

      private:
        /// The class name is used to determine the provenance of a dynamic object class.
        /// Effectively it relates it to a dynamic library name.
        ///
        /// This is derived during the deserialization process. Classes constructed otherwise will
        /// have an empty classname which makes their provenance uncertain. For this reason
        /// all unknown objects are erased on reload unless they manually assume a name which is error prone.
        ///
        /// The class name is an interned selector to replace RTTI which generally breaks across shared libraries
        /// (it doesn't on macOS when using Apple's fork of Clang but that's hardly useful).
        /// Truly the language specification of all time.
        rt::selector classname;

        auto is_dynobject() const -> bool {
            return not classname.empty();
        }

        std::vector<Box<Object>> children;

      protected:
        /// Assumes a classname. Assume the wrong classname and a reload is likely to end in undefined behavior.
        /// Not having one is fine but the object will not be reconstructed on a hot reload.
        /// Unfortunately until C++26 it will be impossible to automate this process.
        /// The good news is that C++26 has reflection and it will automate this process :D
        ///
        /// If a classname is already present it does not override it.
        void assume_classname(rt::selector new_classname) noexcept {
            if (classname.empty()) classname = new_classname;
        }

      public:
        // TODO: Use generic math::Vector<fixed, 2> matrix type once it's adjusted to allow non-float element types.
        //       For now this old sonic::point type will work.
        point<fixed> position;

        Object() = default;
        Object(Object const&) = delete;
        Object(Object&&) = delete;
        auto operator=(Object const&) -> Object& = delete;
        auto operator=(Object&&) -> Object& = delete;
        virtual ~Object() noexcept {}

        /// Called once every tick at 60hz carefully paced in sync with the display clock.
        /// The delta time is effectively constant and can be left out.
        virtual void update(Io& io, rt::Input const& input, Stage& stage) noexcept {}

        /// Called to draw the object with a target slice offset from the screen by the object position.
        ///
        /// The provided target slice retains the width and height of the scene target, so for objects at the origin
        /// it effectively wraps the scene target transparently in a slice, preserving its category.
        ///
        /// TODO: This variant of draw should
        virtual void draw(draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept {}

        auto pixel_pos() const noexcept -> math::point<i32> {
            return math::point { i32(position.x), i32(position.y) };
        }

        // TODO: This should schedule removal like the top level object list to preserve the same semantics.
        void remove(Object* object) noexcept {
            std::erase_if(children, [object](Box<Object>& e) { return e.raw() == object; });
        }

        void add(Box<Object>&& object) noexcept {
            children.emplace_back(std::move(object));
        }

      private:
        /// Recursively draws this object tree.
        ///
        /// TODO: This should probably account for the depth index sorting system once implemented.
        /// TODO: Cute, we're going to have to deserialize trees.
        void draw_tree(draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept {
            this->draw(target.shift(i32(position.x), i32(position.y)), stage);
            for (auto const& child : children)
                child->draw_tree(target.shift(i32(position.x), i32(position.y)), stage);
        }

        /// Recursively updates this object tree.
        void update_tree(Io& io, rt::Input const& input, Stage& stage) noexcept {
            this->update(io, input, stage);
            for (auto const& child : children)
                child->update_tree(io, input, stage);
        }

      public:
        /// Provides internal iteration of children.
        template <typename... Args> void for_children(auto const& fn, Args ...args) {
            for (auto const& child : children) std::invoke(fn, *child, args...);
        }

        /// Provides internal iteration of self and children.
        template <typename... Args> void for_all(auto const& fn, Args ...args) {
            std::invoke(fn, *this, args...);
            on_children(fn);
        }
    };

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
        std::is_same<decltype(Self::rebuild(std::declval<Self const&>())), Box<Object>>::value and
        std::is_same<decltype(Self::serialize(std::declval<Self const&>(), std::declval<BinaryWriter&>())), void>::value and
        std::is_same<decltype(Self::deserialize(std::declval<BinaryReader&>(), std::declval<i32>(), std::declval<i32>())), Box<Object>>::value
    >> : std::true_type {};

    using ObjectRebuilder    = auto (*) (Object const&) -> Box<Object>;
    using ObjectSerializer   = auto (*) (Object const&, BinaryWriter&) -> void;
    using ObjectDeserializer = auto (*) (BinaryReader&, i32 x, i32 y) -> Box<Object>;

    // /// A game object loadable from files and hot-reloadable during gameplay.
    // /// Obviously don't attempt rebuilding if the ABI was broken between reloads.
    // template <typename Self> concept DynamicObject = requires(
    //     Object const& self, rt::BinaryReader& r, rt::BinaryWriter& w, i32 x, i32 y
    // ) {
    //     { &Self::rebuild } -> std::same_as<ObjectRebuilder>;
    //     { &Self::serialize } -> std::same_as<ObjectSerializer>;
    //     { &Self::deserialize } -> std::same_as<ObjectDeserializer>;
    // };

    /// Provides default implementations of the dynamic object interface.
    /// It is advised to perform a super call when using this so that it can reconstruc the object base.
    ///
    /// TODO: This is stupid, just put it in the Object supertype and use newer C++ deducing this.
    ///       Silly language though, having no class ("static") inheritance.
    template <typename Self> struct DefaultCodable {
        static auto rebuild(Object const& existing) -> Box<Object> {
            auto ret = Box<Self>::make();
            ret->position = existing.position;
            return ret;
        }

        static auto deserialize(BinaryReader& reader, i32 x, i32 y) -> Box<Object> {
            auto ret = Box<Self>::make();
            ret->position = { x, y };
            return ret;
        }

        static void serialize(Object const& self, BinaryWriter& writer) {
            // TODO: Serialize basics and classname.
        }
    };
}
