# Game

File structure

- `src` contains the engine
- `object` contains game objects

Game objects are implemented seperately because they are compiled as shared libraries and reloaded at runtime.

For now they are hardcoded in the stage file as there is no runtime asset loader yet.
