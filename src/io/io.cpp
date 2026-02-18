#include "io.hpp"

thread_local std::vector<Io*> Io::threadlocal_io_stack;
