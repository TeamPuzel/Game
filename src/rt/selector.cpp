#include "selector.hpp"

std::unordered_set<std::string> rt::selector::pool;
std::mutex rt::selector::sync;

std::unordered_set<std::string>* rt::selector::pool_ref = nullptr;
std::mutex* rt::selector::sync_ref = nullptr;
