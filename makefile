# A makefile containing some commands I use a lot.
# It is not actually used for building anything, all of that is specified with CMake.
# These are mostly used as Zed tasks but I define them here for compatibility with other workflows.

BUILD_TYPE ?= Release

REFLECTION_LLVM_DIR ?= /Users/teampuzel/OpenSource/clang-p2996/build
HOMEBREW_LLVM_DIR   ?= /opt/homebrew/Cellar/llvm/21.1.3

COMPILER_FLAGS = \
	-DCMAKE_C_COMPILER=$(REFLECTION_LLVM_DIR)/bin/clang \
	-DCMAKE_CXX_COMPILER=$(REFLECTION_LLVM_DIR)/bin/clang++ \
	-DCMAKE_CXX_FLAGS="-fsanitize=bounds,local-bounds -fsanitize-trap=bounds,local-bounds" \
	-DCUSTOM_CXX_INCLUDE_DIR=$(REFLECTION_LLVM_DIR)/include/c++/v1 \
	-DCUSTOM_CXX_LIB_DIR=$(REFLECTION_LLVM_DIR)/lib

all: bundle

build/build.ninja: CMakeLists.txt
	@cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON $(COMPILER_FLAGS)

# Counts the lines of code :)
# Requires cloc to be installed of course.
cloc:
	@cloc src object

# Switch clangd to the native build.
clangd-build:
	@echo "CompileFlags:" > .clangd
	@echo "  CompilationDatabase: build" >> .clangd

# Switch clangd to the cross-build.
clangd-cross:
	@echo "CompileFlags:" > .clangd
	@echo "  CompilationDatabase: build-cross" >> .clangd

setup: clangd-build build/build.ninja
	@rm -rf build
	@cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON $(COMPILER_FLAGS)

build: build/build.ninja
	@ninja -C build

# Bundles the application.
bundle: build/build.ninja
	@cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUNDLE_RELEASE=ON $(COMPILER_FLAGS)
	@ninja -C build

# Runs the game natively.
run: build
	@cd build; ./bubble

profile: build
	@cd build; xcrun xctrace record --template 'Game' --launch -- ./bubble

# A simple command for recompiling parts of the game while it's running.
# Because most of the engine is just headers this inherently includes hot reloading those parts of the engine.
reload: build/build.ninja
	@cmake -B build -DHOT_RELOAD=ON
	@ninja -C build
	@pkill -USR1 bubble

# A convenience for building the binary for Windows from UNIX operating systems.
# It's not even that hard, I feel bad for people who think they need to use Windows for anything.
# If anything supporting MSVC is more difficult due to how different it is and how sad the C++ standard is.
cross-setup: clangd-cross
	@rm -rf build-cross
	@cmake -B build-cross -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	    -DCMAKE_TOOLCHAIN_FILE=cross/windows-toolchain.cmake

cross-build: cross-setup
	@cd build-cross; ninja

cross-run: cross-build
	@cd build-cross; wine bubble.exe
