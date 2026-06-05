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
#include <meta>

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

      protected:
        virtual auto is_dynobject() const -> bool { return false; }

        virtual auto classname() const -> std::string_view {
            throw std::logic_error("only dynamic objects have classnames");
        }

        virtual auto is_serial() const -> bool { return false; }

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

        auto isa(std::string_view name) const noexcept -> bool {
            return classname() == name;
        }

        void wrap_position() noexcept {
            constexpr i32 screen_width = 32 * 8;                        // 256 px
            constexpr i32 screen_height = 30 * 8;                       // 240 px
            constexpr i32 top_margin = 4 * 8;                           // 32  px
            // constexpr i32 top_margin = 0;                               // 0   px
            constexpr i32 playable_height = screen_height - top_margin; // 208 px

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
    };

    template <typename T, typename U> auto flat_cast(U* ptr) noexcept -> T* {
        return dynamic_cast<T*>(ptr);
        // if (auto result = dynamic_cast<T*>(ptr)) return result;

        // std::string_view t = typeid(T).name(), u = typeid(*ptr).name();
        // if (t == u) return (T*) ptr; else return nullptr;
    }

    template <typename T, typename U> auto flat_cast(U const* ptr) noexcept -> T const* {
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
        std::is_same<decltype(Self::rebuild(std::declval<Self const*>())), Box<Object>>::value and
        std::is_same<decltype(Self::serialize(std::declval<Self const*>(), std::declval<BinaryWriter&>())), void>::value and
        std::is_same<decltype(Self::deserialize(std::declval<BinaryReader&>(), std::declval<i32>(), std::declval<i32>())), Box<Object>>::value and
        std::is_same<decltype(Self::initialize(std::declval<i32>(), std::declval<i32>())), Box<Object>>::value
    >> : std::true_type {};

    using ObjectRebuilder    = auto (*) (Object const*) -> Box<Object>;
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

    struct serial_t {
        constexpr bool operator<=>(serial_t const&) const = default;
    };
    struct reload_t {
        constexpr bool operator<=>(reload_t const&) const = default;
    };

    /// Annotates a property for serialization and hot reloading by Codable.
    constexpr serial_t serial;
    /// Annotates a property for hot reloading by Codable.
    constexpr reload_t reload;

    template <const usize N> requires (N > 0) struct meta_info_array {
        std::meta::info data[N];
        constexpr std::meta::info const* begin() const { return data; }
        constexpr std::meta::info const* end() const { return data + N; }
    };

    template <typename T> consteval auto has_reflected_members() {
        return std::meta::nonstatic_data_members_of(^^T).size() > 0;
    }

    template <typename T> consteval auto reflected_members() {
        constexpr auto size = std::meta::nonstatic_data_members_of(^^T).size();
        meta_info_array<size> arr;
        auto vec = std::meta::nonstatic_data_members_of(^^T);
        for (usize i = 0; i < size; i += 1) arr.data[i] = vec[i];
        return arr;
    }

    template <typename T> constexpr void write_reflected_member(BinaryWriter& writer, T const& value) {
        using Self = std::remove_cvref_t<T>;

        if constexpr (std::is_enum_v<Self>) {
            using IntType = std::underlying_type_t<Self>;
            writer.template write<IntType>(static_cast<IntType>(value));
        } else if constexpr (std::integral<Self>) {
            writer.template write<Self>(value);
        } else if constexpr (std::same_as<Self, bool>) {
            writer.boolean(value);
        } else if constexpr (std::same_as<Self, fixed>) {
            writer.u32(std::bit_cast<u32>(value));
        } else if constexpr (std::same_as<Self, point<fixed>>) {
            write_reflected_member(writer, value.x);
            write_reflected_member(writer, value.y);
        } else if constexpr (std::same_as<Self, f32>) {
            writer.f32(value);
        } else if constexpr (std::same_as<Self, f64>) {
            writer.f64(value);
        } else {
            static_assert(sizeof(Self) == 0, "unsupported serialization type");
        }
    }

    template <typename T> constexpr void read_reflected_member(BinaryReader& reader, T& value) {
        using Self = std::remove_reference_t<T>;

        if constexpr (std::is_enum_v<Self>) {
            using IntType = std::underlying_type_t<Self>;
            value = static_cast<Self>(reader.template read<IntType>());
        } else if constexpr (std::integral<Self>) {
            value = reader.template read<Self>();
        } else if constexpr (std::same_as<Self, bool>) {
            value = reader.boolean();
        } else if constexpr (std::same_as<Self, fixed>) {
            value = std::bit_cast<Self>(reader.u32());
        } else if constexpr (std::same_as<Self, point<fixed>>) {
            read_reflected_member(reader, value.x);
            read_reflected_member(reader, value.y);
        } else if constexpr (std::floating_point<Self>) {
            value = reader.template read<Self>();
        } else {
            static_assert(sizeof(Self) == 0, "unsupported serialization type");
        }
    }

    /// Provides default implementations of the dynamic object serial interface using reflection.
    /// Shadowing with custom implementations is possible but preferably avoided.
    template <typename Self, typename Base = Object>
        requires std::is_base_of_v<Object, Base>
    class CodableObject : public Base {
      public:
        auto is_dynobject() const -> bool final override { return true; }

        auto classname() const -> std::string_view final override { return std::meta::identifier_of(^^Self); }

        /// By default an object is serial if it has any serialized properties.
        /// An object with no serialized properties of its own can still opt in by being annotated as serial itself.
        auto is_serial() const -> bool override {
            constexpr auto annotation = std::meta::annotation_of_type<serial_t>(^^Self);
            if constexpr (annotation) return true;

            if constexpr (has_reflected_members<Self>()) template for (constexpr auto member : reflected_members<Self>()) {
                constexpr auto annotation = std::meta::annotation_of_type<serial_t>(member);
                if constexpr (annotation) return true;
            }

            return false;
        }

        static auto rebuild(Object const* existing) -> Box<Object> {
            auto ret = Box<Self>::make();
            ret->position = existing->position;

            auto self = static_cast<Self const*>(existing);

            if (self) {
                if constexpr (has_reflected_members<Self>()) template for (constexpr auto member : reflected_members<Self>()) {
                    constexpr auto serial = std::meta::annotation_of_type<serial_t>(member);
                    constexpr auto reload = std::meta::annotation_of_type<reload_t>(member);
                    if constexpr (serial or reload) ret.raw()->[:member:] = self->[:member:];
                }
            }

            return ret;
        }

        static auto initialize(i32 x, i32 y) -> Box<Object> {
            auto ret = Box<Self>::make();
            ret->position = { x, y };
            return ret;
        }

        static void serialize(Object const* erased, BinaryWriter& writer) {
            auto self = static_cast<Self const*>(erased);

            if (not self->is_serial()) throw std::logic_error("attempted to serialize a non serial object");

            if constexpr (has_reflected_members<Self>()) template for (constexpr auto member : reflected_members<Self>()) {
                constexpr auto annotation = std::meta::annotation_of_type<serial_t>(member);
                if constexpr (annotation) write_reflected_member(writer, self->[:member:]);
            }
        }

        static auto deserialize(BinaryReader& reader, i32 x, i32 y) -> Box<Object> {
            auto base_box = initialize(x, y);
            auto self = box_cast<Self>(base_box);

            if constexpr (has_reflected_members<Self>()) template for (constexpr auto member : reflected_members<Self>()) {
                constexpr auto annotation = std::meta::annotation_of_type<serial_t>(member);
                if constexpr (annotation) read_reflected_member(reader, self.raw()->[:member:]);
            }

            return self;
        }
    };
}
