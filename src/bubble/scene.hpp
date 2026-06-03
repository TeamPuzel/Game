// Created by Lua (TeamPuzel) on May 26th 2025.
// Copyright (c) 2025 All rights reserved.
//
// A simple abstraction for switching scenes at runtime.
// Can be used for the stage, menu etc.
#pragma once
#include <primitive>
#include <draw>
#include <sound>
#include <rt>

namespace bubble {
    using draw::Image;
    using draw::Ref;
    using draw::Grid;
    using draw::Color;
    using draw::ScaledPlane;
    using draw::MozaicPlane;

    struct SoundLibrary final {
        struct Tsh {
            using is_transparent = void;
            std::size_t operator()(std::string_view txt) const { return std::hash<std::string_view>()(txt); }
        };

      public:
        struct SoundRequest final {
            std::string name;
            std::string filename;
            enum class Type: u8 { Wave, Ogg } type;

            SoundRequest(std::string name, std::string filename, Type type)
                : name(name), filename(filename), type(type)
            {
                if (name.empty()) throw std::logic_error("unnamed sound request");
                if (filename.empty()) throw std::logic_error("unnamed file request");
            }
        };

      private:
        std::queue<SoundRequest> queue;
        std::unordered_map<std::string, sound::Wave, Tsh, std::equal_to<>> sound_storage;
        std::mutex sound_storage_mutex;
        std::condition_variable cv;
        std::atomic<bool> fetching = false;

      public:
        SoundLibrary() {}

        void enqueue(SoundRequest request) { queue.push(std::move(request)); }

        void enqueue(std::string name, std::string filename, SoundRequest::Type type) {
            enqueue({ std::move(name), std::move(filename), type });
        }

        rt::DetachedTask fetch(Io& io) {
            std::queue<SoundRequest> queue;
            queue.swap(this->queue);
            fetching = true;

            ScopeExit scope_exit([&] {
                fetching = false;
                cv.notify_all();
            });

            co_await rt::enqueue();

            while (not queue.empty()) {
                auto [name, filename, type] = std::move(queue.front()); queue.pop();

                sound::Wave sound; switch (type) {
                    case SoundRequest::Type::Wave: sound = sound::Wave::from(io.read_wavefile(filename)); break;
                    case SoundRequest::Type::Ogg:  sound = sound::Wave::from(io.read_oggfile(filename));  break;
                }

                std::lock_guard lock(sound_storage_mutex);
                sound_storage.insert({ name, std::move(sound) });

                cv.notify_all(); // We have a new sound someone might be waiting for.
            }
        }

        /// This will return a reference to a sound if it exists.
        /// If it doesn't and there is a fetch in progress it will block until it is found or the fetch ends.
        /// If the fetch ended but the sound doesn't exist, or there was no fetch to begin with,
        /// the function will instead throw std::logic_error.
        ///
        /// The use case for this object is primarily to prefetch sounds asynchronously before they are needed
        /// to hopefully avoid blocking the game when trying to load a sound.
        auto get(std::string_view name) -> sound::Wave const& {
            std::unique_lock lock(sound_storage_mutex);

            cv.wait(lock, [&] {
                return sound_storage.contains(name) or not fetching.load();
            });

            auto search = sound_storage.find(name);
            if (search != sound_storage.end()) {
                return search->second;
            } else {
                throw std::logic_error("impossible get of a missing sound without a fetch in progress");
            }
        }
    };

    /// A scene coroutine which can be run.
    ///
    /// Rendering is performed into an image rather than a dynamic target since this is
    /// significantly faster for obvous reasons and we control the runtime and can ensure it is one.
    class Scene {
        static Box<Scene>* root_ptr;
        static std::optional<Box<Scene>> scheduled_transition;

      protected:
        static auto transition(Box<Scene>&& scene) {
            if (scheduled_transition) throw std::logic_error("double transition");
            scheduled_transition = std::move(scene);
        }

      public:
        /// Advances the state by 1/60 of a second.
        virtual void update(Io& io, rt::Input const& input, rt::SoundStage& sound) = 0;
        /// Called after update to mutate the render target.
        virtual void draw(Io& io, rt::Input const& input, Ref<Image> target) const = 0;

        virtual ~Scene() noexcept {}

        virtual void hot_reload(Io& io) {}

        // The root pointer must be pinned in memory for as long as scenes exist.
        static void unsafe_set_root_ptr(Box<Scene>* root_ptr) {
            Scene::root_ptr = root_ptr;
        }

        static void unsafe_apply_transition() {
            if (scheduled_transition) {
                *root_ptr = std::move(*scheduled_transition);
                scheduled_transition = std::nullopt;
            }
        }

        static auto transitioning() -> bool {
            return (bool) scheduled_transition;
        }
    };
}
