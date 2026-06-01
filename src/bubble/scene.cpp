#include "scene.hpp"

using namespace bubble;

Box<Scene>* Scene::root_ptr = nullptr;
std::optional<Box<Scene>> Scene::scheduled_transition;
