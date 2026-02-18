// Created by Lua (TeamPuzel) on May 26th 2025.
// Copyright (c) 2025 All rights reserved.
#pragma once
#include <primitive>
#include <math>
#include <rt>
#include <font>
#include <sstream>
#include <string_view>
#include <typeindex>
#include <unordered_map>

namespace sonic {
    using draw::Image;
    using draw::Ref;
    using draw::Color;
    using draw::Text;
    using math::point;
    using math::angle;

    class Stage;

    /// A dynamic game object.
    ///
    /// Anything that isn't part of the tile grid.
    ///
    /// These assume 60Hz updates for multiple reasons:
    /// - 8 bit fixed point arithmetic doesn't work with delta time, I didn't add precision only
    ///   to mess with it using non-deterministic techniques.
    /// - The original games worked like this so it's more accurate.
    /// - It's easy to match updates perfectly evenly to any sane refresh rate (144Hz can go and disappear for all I care).
    /// - Graphics could be interpolated between updates which I might do later. At that point precision
    ///   is less important since it doesn't affect the simulation itself.
    ///
    /// The data of an object is a match of the state described in the Sonic Physics Guide.
    /// You could divide this data up over some overengineered inheritance hierarchy
    /// but I do not want to bother for literally 4 variables, I'll just store what the original game did.
    ///
    /// Inheritance is assumed and required to be trivial, preferably flat. Subclasses violating this
    /// property are likely to cause undefined behavior.
    class Object {
        friend class Stage;

      public:
        class Trait;

      private:

        /// The class name is used to determine the provenance of the dynamic object class.
        /// Determines the dynamic class object provenance (relates it to a library name).
        ///
        /// This is derived from the serialization process. Classes constructed otherwise will
        /// have an empty classname which makes their provenance uncertain. For this reason
        /// all unknown objects are erased on reload.
        std::string classname;

        auto is_dynobject() const -> bool {
            return not classname.empty();
        }

        /// We want dynamic traits on objects, and this is actually quite nice because an empty set does not allocate.
        /// At least I hope. This is C++ after all and strange decisions are made here. vector<bool>? :3
        /// For the price of storing it here we'd get the ability to associate dynamic functionality with objects
        /// without relying on virtual inheritance which is not something I wish to manage.
        ///
        /// These can also be moved around at runtime which is cool.
        ///
        /// Also, wtf C++ this class is 40 bytes?? This sad library is tempting me to write my own array and map again.
        std::unordered_map<std::type_index, Box<Trait>> traits;

      protected:
        /// Assumes a classname. Assume the wrong classname and a reload is likely to end in undefined behavior.
        /// Not having one is fine but the object will not be reconstructed on a hot reload.
        /// Unfortunately until C++26 it will be impossible to automate this process.
        /// The good news is that C++26 has reflection and it will automate this process :D
        ///
        /// If a classname is already present it does not override it.
        void assume_classname(std::string_view new_classname) noexcept {
            if (classname.empty()) {
                classname = new_classname;
            }
        }

      public:
        class Trait {
          public:
            /// Pointer to the object the trait is associated with or null if not associated with any.
            Object* object { nullptr };

            Trait() = default;
            Trait(Trait const&) = delete;
            Trait(Trait&&) = delete;
            auto operator=(Trait const&) -> Trait& = delete;
            auto operator=(Trait&&) -> Trait& = delete;
            virtual ~Trait() noexcept {}

            virtual void update(rt::Input const& input, Stage& stage) noexcept {}
        };

        template <typename T, typename... Args> void add(Args&&... args) {
            auto trait = Box<T>::make(std::forward<Args>(args)...);
            trait->object = this;
            traits[typeid(T)] = std::move(trait);
        }

        template <typename T> auto remove() -> Box<T> {
            if (auto trait = traits.extract(typeid(T))) {
                trait.mapped()->object = nullptr;
                return std::move(trait.mapped());
            } else {
                return Box<T>();
            }
        }

        template <typename T> auto trait() [[clang::lifetimebound]] -> T* {
            auto it = traits.find(typeid(T));
            if (it != traits.end()) {
                return static_cast<T*>(it->second.raw());
            } else {
                return nullptr;
            }
        }

        point<fixed> position;
        point<fixed> speed;
        fixed ground_speed;
        angle ground_angle;

      private:
        /// Internal supertype update which cannot be overriden.
        void managed_update(rt::Input const& input, Stage& stage) {
            for (auto& trait : traits) {
                trait.second->update(input, stage);
            }
        }

      public:
        Object() = default;
        Object(Object const&) = delete;
        Object(Object&&) = delete;
        auto operator=(Object const&) -> Object& = delete;
        auto operator=(Object&&) -> Object& = delete;
        virtual ~Object() noexcept {}

        /// Called once every tick at 60hz.
        virtual void update(rt::Input const& input, Stage& stage) noexcept {}

        /// If an instance answers true it will be treated as  active even when out of range.
        /// This is useful for more dynamic constructs.
        ///
        /// Another use would be an object flipping to true after it was initialized. That way it can remain inactive
        /// until first loaded in.
        virtual auto force_active() const -> bool {
            return false;
        }

        /// Called when the object detects some other object colliding with it.
        /// Notably we take the other object by pointer not reference, because dynamic casts
        /// of references throw rather than returning null. The provided pointer itself is never null.
        virtual void collide_with(Object* other) noexcept {}

        /// An object relative hitbox.
        struct Hitbox final {
            i32 x { 0 }, y { 0 }, w { 0 }, h { 0 };

            /// Convenience factory method which yields the only kind of hitbox the game supported,
            /// a width and height radius based hitbox.
            static constexpr auto of_radii(i32 w, i32 h) noexcept -> Hitbox {
                return { -w, -h, 2 * w, 2 * h };
            }

            /// A convenience which shifts the hitbox in relative space since this is something
            /// pretty common among object hitboxes.
            constexpr auto shift(i32 ox, i32 oy) const noexcept -> Hitbox {
                return { x + ox, y + oy, w, h };
            }
        };

        virtual auto hitbox() const noexcept -> Hitbox {
            return {};
        }

      private:
        struct AbsoluteHitbox final {
            i32 x, y, w, h;

            constexpr auto overlaps(AbsoluteHitbox other) const noexcept -> bool {
                if (w == 0 or h == 0 or other.w == 0 or other.h == 0) return false;
                return not (
                    x + w < other.x or
                    other.x + other.w < x or
                    y + h < other.y or
                    other.y + other.h < y
                );
            }
        };

        auto absolute_hitbox() const -> AbsoluteHitbox {
            const auto local = hitbox();
            const auto [px, py] = pixel_pos();
            return { local.x + px, local.y + py, local.w, local.h };
        }

      public:
        enum class Mode : u8 {
            Floor,
            RightWall,
            Ceiling,
            LeftWall,
        };

        struct Sprite final {
            i32 x { 0 }, y { 0 }, w { 0 }, h { 0 };
            bool mirror_x { false }, mirror_y { false };
            u8 rotation { 0 };
        };

        virtual auto sprite(rt::Input const& input) const noexcept -> Sprite {
            return Sprite {};
        };

        auto tile_pos() const noexcept -> math::point<i32> {
            return math::point { i32(position.x) / 16, i32(position.y) / 16 };
        }

        auto pixel_pos() const noexcept -> math::point<i32> {
            return math::point { i32(position.x), i32(position.y) };
        }

        auto is_underwater() const noexcept -> bool {
            return false;
        }

        struct CameraBuffer final { i32 width, height, speed_cap; };

        /// The object can determine the area buffer within which the camera does not scroll.
        /// Once the camera leaves this buffer it is scrolled at a speed matching the object exactly,
        /// except capped to 16px per frame. This was a technical limitation but it's pretty cool
        /// as it gives a sense of speed since Sonic is outrunning the camera :)
        ///
        /// The returned width and height represent extents in both directions on their axis.
        virtual auto camera_buffer() const noexcept -> CameraBuffer {
            return { 0, 0, 16 };
        }

        /// Called on the primary, meant for drawing the object specific UI.
        /// The object receives the screen slice to draw into freely.
        virtual void hud_draw(Io& io, Ref<Image> target, Stage const& stage) const noexcept {}

        /// Called when debug drawing is enabled, meant for visualising collision etc.
        /// The object receives the global debug overlay output and the camera slice to draw into freely.
        virtual void debug_draw(Io& io, std::stringstream& out, draw::Slice<Ref<Image>> target, Stage const& stage) const noexcept {}
    };

    /// Provides default implementations of the dynamic object interface.
    /// It is advised to perform a super call when using this so that it can reconstruc the object base.
    ///
    /// TODO: This is stupid, just put it in the Object supertype and use newer C++ deducing this.
    /// I was made to use C++17 though so we shall still use silly CRTP patterns :)
    template <typename Self> struct DefaultCodable {
        static auto rebuild(Object const& existing) -> Box<Object> {
            auto ret = Box<Self>::make();
            ret->position = existing.position;
            ret->speed = existing.speed;
            ret->ground_speed = existing.ground_speed;
            ret->ground_angle = existing.ground_angle;
            return ret;
        }

        static auto deserialize(rt::BinaryReader& reader, i32 x, i32 y) -> Box<Object> {
            auto ret = Box<Self>::make();
            ret->position = { x, y };
            return ret;
        }

        static void serialize(Object const& self, rt::BinaryWriter& writer) {
            // TODO: Serialize basics and classname.
        }
    };

    /// An object which damages the player on collision.
    class DamagesPlayer final : public Object::Trait {
      public:
        enum Severity {
            UnprotectedOnly,
            BypassProtection,
        };

      private:
        Severity s;

      public:
        DamagesPlayer(Severity s) : s(s) {}

        auto severity() const -> Severity {
            return s;
        }

        auto unprotected_only() const -> bool {
            return s == UnprotectedOnly;
        }

        auto bypass_protection() const -> bool {
            return s == BypassProtection;
        }
    };

    /// An object which takes damage from the player on collision.
    class TakesDamageFromPlayer final : public Object::Trait {
      public:
        using DamageHandler = auto (Object::*) () -> void;

        DamageHandler handler;

        template <typename T> TakesDamageFromPlayer(T handler) : handler(static_cast<DamageHandler>(handler)) {}

        void damage() {
            (object->*handler)();
        }
    };
}
