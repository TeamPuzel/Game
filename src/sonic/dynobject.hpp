// Created by Lua (TeamPuzel) on August 12th 2025.
// Copyright (c) 2025 All rights reserved.
//
// This header defines utilities for safely reloading object classes at runtime.
#pragma once
#include <rt>

#if defined(_MSC_VER)
#define DLLEXPORT [[gnu::dllexport]]
#else
#define DLLEXPORT
#endif

// A type-safe and cross-platform object exporter.
// This is not for use in headers but rather the object entry point to expose it
// from the shared object file.
//
// This very elaborate (weird) with member pointers. Oh well. I don't feel like
// bothering to handle the botched variance in the Microsoft's implementation (C++ is such a well standardized language)
// The serializer will just be static instead of a member. Whatever. I hate this language.
#define EXPORT_SONIC_OBJECT(CLASSNAME)                                 \
static_assert(DynamicObject<CLASSNAME>::value);                        \
extern "C" DLLEXPORT ObjectRebuilder __sonic_object_rebuild() {        \
    return (ObjectRebuilder) &CLASSNAME::rebuild;                      \
}                                                                      \
extern "C" DLLEXPORT ObjectSerializer __sonic_object_serialize() {     \
    return (ObjectSerializer) &CLASSNAME::serialize;                   \
}                                                                      \
extern "C" DLLEXPORT ObjectDeserializer __sonic_object_deserialize() { \
    return (ObjectDeserializer) &CLASSNAME::deserialize;               \
}
#define OBJECT_REBUILD "__sonic_object_rebuild"
#define OBJECT_SERIALIZE "__sonic_object_serialize"
#define OBJECT_DESERIALIZE "__sonic_object_deserialize"

namespace sonic {
    class Object;

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
        std::is_same<decltype(Self::serialize(std::declval<Self const&>(), std::declval<rt::BinaryWriter&>())), void>::value and
        std::is_same<decltype(Self::deserialize(std::declval<rt::BinaryReader&>(), std::declval<i32>(), std::declval<i32>())), Box<Object>>::value
    >> : std::true_type {};

    using ObjectRebuilder    = auto (*) (Object const&) -> Box<Object>;
    using ObjectSerializer   = auto (*) (Object const&, rt::BinaryWriter&) -> void;
    using ObjectDeserializer = auto (*) (rt::BinaryReader&, i32 x, i32 y) -> Box<Object>;

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
