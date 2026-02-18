# A makefile containing some commands I use a lot.
# It is not actually used for building anything, all of that is specified with CMake.
# These are mostly used as Zed tasks but I define them here for compatibility with other workflows.

all: setup

# Counts the lines of code :)
# Requires cloc to be installed of course.
cloc:
	@cloc src

# Switch clangd to the native build.
clangd-build:
	@echo "CompileFlags:" > .clangd
	@echo "  CompilationDatabase: build" >> .clangd

# Switch clangd to the cross-build.
clangd-cross:
	@echo "CompileFlags:" > .clangd
	@echo "  CompilationDatabase: build-cross" >> .clangd

setup: clangd-build
	@rm -rf build
	@cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

build: setup
	@cd build; ninja

# Runs the game natively.
run: build
	@cd build; ./renderer

# A convenience for building the binary for Windows from UNIX operating systems.
# It's not even that hard, I feel bad for people who think they need to use Windows for anything.
# If anything supporting MSVC is more difficult due to how different it is and how sad the C++ standard is.
cross-setup: clangd-cross
	@rm -rf build-cross
	@cmake -B build-cross -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	    -DCMAKE_TOOLCHAIN_FILE=cross/windows-toolchain.cmake

cross-build: cross-setup
	@cd build-cross; ninja

cross-run: cross-build
	@cd build-cross; wine renderer.exe
