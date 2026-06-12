# Game

https://github.com/TeamPuzel/Game

## Project structure

- `src` contains the engine.
- `object` contains game objects.
- `lib` contains minimal vendor code.
- `cross` contains cross compilation files.

Game objects are implemented seperately because they are a shared library live-reloadable at runtime.

## Source structure

- `bubble` the game.
- `draw` a std::ranges like library for functional programming of general raster graphics and layout.
- `sound` a std::ranges like library for functional programming of sound.
- `io` a small library providing a runtime io interface and binary coding.
- `rt` an SDL3 based runtime providing graphics output, input, tasks, executors and an implementation of the io interface.
- `primitive` implements aliases, a box and a 24.8 signed fixed point type for retro games.
- `math` implements a safe angle type and a unified matrix/vector type (unused).
- `font` implements a few variable width pixel fonts for use with the draw library.

## Windows

Because Windows does not have a standard fullscreen toggle, Windows builds support using
either F11 or Alt + Enter to toggle it.

## Special notes

- Classes used across ABI boundaries are intentionally not marked final as that would allow the compiler
to perform devirtualization of methods intentionally made virtual to work across shared libraries and
update when performing a hot reload even when defined in a header file.
- Downcasting across ABI in C++ is a mess and relies on a fallback cast. This is solvable but would require
a hard dependency on C++26 reflection to form structured metadata accessible through the vtables,
which only works when using clang.
- C++26 reflection is used for hot reloading and serialization of objects to level files, this is
a soft dependency and without it fallback deserializers are used (MSVC builds can play stages, they can't edit them).
- C++20 coroutines are used to context switch between two executors, moving tasks to the background and back.

### Engine base

I was not able to use the recommended engine base because it was fundamentally
incompatible with my preferred programming patterns :(

- Floating point math is not deterministic and often differs between architectures or even CPU vendors. They also
fundamentally don't have almost any important numeric properties forcing all physics code to deal with the error
accumulating on almost every operation; floats are severely overused for purposes they are not fit for and
nothing upsets me more than misaligned pixel sprites in games that use this horrible numeric standard for 2d.
- Textures are not great for complex raster effects without fragment shaders, which are highly unportable and
completely avoidable trivial 2d which can just be rendered in software.
- I prefer function composition of lazy wrappers to state management, side effect purity greatly limits logic bugs.
- I value purity, so I don't like to use global IO operations, everything has to use dependency injection if possible.
- I like my hot reloading setup as it allows me to iterate mechanics and visuals quickly.

Ideally I would be able to integrate the vblank interrupt with C++20 async/await to more optimally schedule
tasks for as long as vblank lasts but not any longer, however SDL did not expose that kind of callback based API when
I was writing the run procedure. For the purposes of this game however there is not enough background activity
for this to cause any issues (excess pressure on the main executor could get it stuck in an infinite loop blocking
the SDL event loop completely).
Async/await would also be a good fit for WASM support, however C++ is fundamentally
terrible for embedded targets like that (unportable standard library) compared to Swift or Rust which are capable
of compiling to pure libc-free wasm and not the severely jank emscripten target.
