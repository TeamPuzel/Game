# Game

### Project structure

- `src` contains the engine.
- `object` contains game objects.
- `lib` contains minimal vendor code.
- `cross` contains cross compilation files.

Game objects are implemented seperately because they are a shared library live-reloadable at runtime.

### Source structure

- `bubble` the game.
- `draw` a std::ranges like library for functional programming of general raster graphics and layout.
- `sound` a std::ranges like library for functional programming of sound.
- `io` a small library providing a runtime io interface and binary coding.
- `rt` an SDL3 based runtime providing graphics output, input, tasks, executors and an implementation of the io interface.
- `primitive` implements aliases, a box and a 24.8 signed fixed point type for retro games.
- `math` implements a safe angle type and a unified matrix/vector type.
- `font` implements a few variable width pixel fonts for use with the draw library.

### Special notes

- Classes used across ABI boundaries are intentionally not marked final as that would allow the compiler
to perform devirtualization of methods intentionally made virtual to work across shared libraries and
update when performing a hot reload even when defined in a header file.
- Downcasting across ABI in C++ is a mess and relies on a fallback cast. This is solvable but would require
a hard dependency on C++26 reflection to form structured metadata accessible through the vtables,
which only works when using clang.
- C++26 reflection is used for hot reloading and serialization of objects to level files, this is
a soft dependency and without it fallback deserializers are used (MSVC builds can play stages, they can't edit them).
- C++20 coroutines are used to context switch between two executors, moving tasks to the background and back.
