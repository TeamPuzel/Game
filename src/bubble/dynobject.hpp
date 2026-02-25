// Created by Lua (TeamPuzel) on August 12th 2025.
// Copyright (c) 2025 All rights reserved.
//
// This header defines utilities for safely reloading object classes at runtime.
#pragma once
#include <io>

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
#define EXPORT_GAME_OBJECT(CLASSNAME)                                 \
static_assert(DynamicObject<CLASSNAME>::value);                       \
extern "C" DLLEXPORT ObjectRebuilder __game_object_rebuild() {        \
    return (ObjectRebuilder) &CLASSNAME::rebuild;                     \
}                                                                     \
extern "C" DLLEXPORT ObjectSerializer __game_object_serialize() {     \
    return (ObjectSerializer) &CLASSNAME::serialize;                  \
}                                                                     \
extern "C" DLLEXPORT ObjectDeserializer __game_object_deserialize() { \
    return (ObjectDeserializer) &CLASSNAME::deserialize;              \
}
#define OBJECT_REBUILD "__game_object_rebuild"
#define OBJECT_SERIALIZE "__game_object_serialize"
#define OBJECT_DESERIALIZE "__game_object_deserialize"
