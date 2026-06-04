# Game

### File structure

- `src` contains the engine
- `object` contains game objects
- `lib` contains minimal vendor code
- `cross` contains cross compilation files

Game objects are implemented seperately because they are a shared library live-reloadable at runtime.

### Source structure

- `bubble` the game
- `draw` a std::ranges like library for functional programming of general raster graphics and layout
- `sound` a std::ranges like library for functional programming of sound
- `io` a small library providing a runtime io interface
- `rt` an SDL3 based runtime providing graphics output, input, tasks, executors and an implementation of the io interface
- `primitive` implements a 24.8 signed fixed point type for retro games
- `math` implements a safe angle type and a unified matrix/vector type
- `font` implements a few variable width pixel fonts for use with the draw library
