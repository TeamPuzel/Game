// Created by Lua (TeamPuzel) on August 12th 2025.
// Copyright (c) 2025 All rights reserved.
//
// A dynamic class loader for very late binding of game objects.
#pragma once
#include <rt>
#include <unordered_map>
#include "dynobject.hpp"

namespace sonic::class_loader {
    static bool SWAPPED_REGISTRY = false;
    static std::unordered_map<std::string, Io::DynamicLibrary> REGISTRY_0;
    static std::unordered_map<std::string, Io::DynamicLibrary> REGISTRY_1;

    struct DynamicObjectDescriptor final {
        ObjectRebuilder rebuilder;
        ObjectSerializer serializer;
        ObjectDeserializer deserializer;
    };

    template <typename F> using Stub = auto (*) () -> F;

    /// We can use this to load objects from shared libraries. Once there is a built-in level editor
    /// It will be possible to hot reload classes on the fly very conveniently.
    static auto load(Io& io, std::string_view classname) -> DynamicObjectDescriptor {
        auto& reg = SWAPPED_REGISTRY ? REGISTRY_0 : REGISTRY_1;

        std::stringstream library_path;
        library_path << "obj/" << classname << ".object";

        Stub<ObjectRebuilder> rebuilder { nullptr };
        Stub<ObjectSerializer> serializer { nullptr };
        Stub<ObjectDeserializer> deserializer { nullptr };

        const auto fill = [&] (Io::DynamicLibrary const& obj) {
            rebuilder = (decltype(rebuilder)) obj.symbol("__sonic_object_rebuild");
            serializer = (decltype(serializer)) obj.symbol("__sonic_object_serialize");
            deserializer = (decltype(deserializer)) obj.symbol("__sonic_object_deserialize");
        };

        if (const auto it = reg.find(library_path.str()); it != reg.end()) {
            fill(it->second);
        } else {
            auto obj = io.open_library(library_path.str());
            fill(obj);
            reg.emplace(library_path.str(), std::move(obj));
        }

        return { rebuilder(), serializer(), deserializer() };
    }

    /// Swaps which registry is active, since we need to be able to destroy old objects while creating new ones.
    static void swap_registry() {
        SWAPPED_REGISTRY = !SWAPPED_REGISTRY;
    }

    /// Closes shared libraries for the inactive registry.
    static void drop_old_object_classes() {
        auto& reg = SWAPPED_REGISTRY ? REGISTRY_1 : REGISTRY_0;
        reg.clear();
    }

    static void clear() {
        REGISTRY_0.clear();
        REGISTRY_1.clear();
    }
}
