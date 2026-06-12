// A simple game object serialization system using C++26 reflection.
// There is a fallback coder to specialize for C++23 support.
#pragma once
#include "stage.hpp"

namespace bubble {
    namespace detail {
        /// C++23 fallback for class names, which are required for the isa cast.
        template <typename T> consteval auto extract_classname() -> std::string_view {
#ifdef BUBBLE_MSVC_COMPAT
            std::string_view func = __FUNCSIG__; // "... extract_classname<class bubble::Player>(void)"
            usize first = func.find_first_of('<') + 1;
            usize last = func.find_last_of('>');
            std::string_view type = func.substr(first, last - first);
#else
            std::string_view func = __PRETTY_FUNCTION__; // "... extract_classname() [with T = bubble::Player]"
            usize first = func.find_first_of('=') + 2;
            usize last = func.find_last_of(']');
            std::string_view type = func.substr(first, last - first);
#endif
            // Strip namespace.
            usize colon = type.find_last_of(':');
            if (colon != std::string_view::npos) return type.substr(colon + 1);

            // Strip struct/class tags if present ("class Player" -> "Player")
            usize space = type.find_last_of(' ');
            if (space != std::string_view::npos) return type.substr(space + 1);

            return type;
        }
    }

    /// Specialize this template to provide fallback deserialization when C++26 reflection is not available.
    template <typename Self> struct FallbackCoder {
        static void deserialize(Box<Self>& self, BinaryReader& reader) {
            throw std::logic_error(
                std::format("fallback coder not implemented for {}", detail::extract_classname<Self>())
            );
        }
    };
}

#ifdef BUBBLE_REFLECTION_ENABLED
#include <meta>

#define SERIAL [[=serial]]
#define RELOAD [[=reload]]

namespace bubble {
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
        auto is_dynobject() const -> bool override { return true; }

        auto classname() const -> std::string_view override { return identifier_of(^^Self); }

        /// By default an object is serial if it has any serialized properties.
        /// An object with no serialized properties of its own can still opt in by being annotated as serial itself.
        auto is_serial() const -> bool override {
            constexpr auto annotation = annotation_of_type<serial_t>(^^Self);
            if constexpr (annotation) return true;

            if constexpr (not std::is_same_v<Base, Object>) {
                if constexpr (has_reflected_members<Base>()) template for (constexpr auto member : reflected_members<Base>()) {
                    constexpr auto annotation = annotation_of_type<serial_t>(member);
                    if constexpr (annotation) return true;
                }
            }

            if constexpr (has_reflected_members<Self>()) template for (constexpr auto member : reflected_members<Self>()) {
                constexpr auto annotation = annotation_of_type<serial_t>(member);
                if constexpr (annotation) return true;
            }
            return false;
        }

        static auto rebuild(Object* existing, Stage& stage) -> Box<Object> {
            auto ret = Box<Self>::make();
            ret->position = existing->position;

            auto self = static_cast<Self*>(existing);

            if (self) {
                if constexpr (not std::is_same_v<Base, Object>) {
                    if constexpr (has_reflected_members<Base>()) template for (constexpr auto member : reflected_members<Base>()) {
                        constexpr auto serial = annotation_of_type<serial_t>(member);
                        constexpr auto reload = annotation_of_type<reload_t>(member);

                        if constexpr (serial or reload) {
                            if constexpr (is_convertible_type(type_of(member), ^^Box<Object>)) {
                                if (self->[:member:]) {
                                    Box<Object> erased = std::move(self->[:member:]);
                                    stage.unsafe_hot_reload_child(erased);
                                    using Dest = [:type_of(member):]::Pointee;
                                    ret.raw()->[:member:] = erased.template cast<Dest>();
                                }
                            } else {
                                ret.raw()->[:member:] = std::move(self->[:member:]);
                            }
                        }
                    }
                }

                if constexpr(has_reflected_members<Self>()) template for (constexpr auto member : reflected_members<Self>()) {
                    constexpr auto serial = annotation_of_type<serial_t>(member);
                    constexpr auto reload = annotation_of_type<reload_t>(member);

                    if constexpr (serial or reload) {
                        if constexpr (is_convertible_type(type_of(member), ^^Box<Object>)) {
                            if (self->[:member:]) {
                                Box<Object> erased = std::move(self->[:member:]);
                                stage.unsafe_hot_reload_child(erased);
                                using Dest = [:type_of(member):]::Pointee;
                                ret.raw()->[:member:] = erased.template cast<Dest>();
                            }
                        } else {
                            ret.raw()->[:member:] = std::move(self->[:member:]);
                        }
                    }
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
                constexpr auto annotation = annotation_of_type<serial_t>(member);
                if constexpr (annotation) write_reflected_member(writer, self->[:member:]);
            }
        }

        static auto deserialize(BinaryReader& reader, i32 x, i32 y) -> Box<Object> {
            auto base_box = initialize(x, y);
            auto self = box_cast<Self>(base_box);

            if constexpr (has_reflected_members<Self>()) template for (constexpr auto member : reflected_members<Self>()) {
                constexpr auto annotation = annotation_of_type<serial_t>(member);
                if constexpr (annotation) read_reflected_member(reader, self.raw()->[:member:]);
            }

            return self;
        }
    };
}

#else

#define SERIAL
#define RELOAD

namespace bubble {
    /// Provides bare minimum implementations of the dynamic object serial interface required for production
    /// builds without reflection, but support for hot reloading and saving files is not available.
    template <typename Self, typename Base = Object>
        requires std::is_base_of_v<Object, Base>
    class CodableObject : public Base {
      public:
        auto is_dynobject() const -> bool override { return true; }

        auto classname() const -> std::string_view override { return detail::extract_classname<Self>(); }

        auto is_serial() const -> bool override { return false; }

        static auto rebuild(Object* existing, Stage& stage) -> Box<Object> {
            throw std::logic_error("rebuilding not available in this engine build");
        }

        static auto initialize(i32 x, i32 y) -> Box<Object> {
            auto ret = Box<Self>::make();
            ret->position = { x, y };
            return ret;
        }

        static void serialize(Object const* erased, BinaryWriter& writer) {
            throw std::logic_error("serialization not available in this engine build");
        }

        static auto deserialize(BinaryReader& reader, i32 x, i32 y) -> Box<Object> {
            auto box = initialize(x, y).template cast<Self>();
            FallbackCoder<Self>::deserialize(box, reader);
            return box;
        }
    };
}

#endif
