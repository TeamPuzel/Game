# The main build file for the project

# Project --------------------------------------------------------------------------------------------------------------

# This is important otherwise macOS will choose AppleClang and Linux will generally choose GCC.
# Technically this is meaningless on Windows (when used with Visual Studio) as per typical
# Microsoft nonsense they use a dedicated JSON file to declare the compiler.
# Just to be safe this is will remain conditional.
# AppleClang does have a myesterious benefit of making RTTI work across shared libraries but that's hardly
# useful as it's definitely not a portable behavior as the C++ standard doesn't standardize anything at all here.
if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
endif()

cmake_minimum_required(VERSION 3.21)
project(Game C CXX)

# Language -------------------------------------------------------------------------------------------------------------

# Technically the codebase uses C++23 with C++26 extensions enabled, but last time I tried MSVC was far behind
# so while not much is changed in the configurations that code had to be temporarily disabled.
# I cross compile to Windows, while for now it still has some shrinking market share, it is
# purely a target platform, not a serious development environment. This requires the STL and Windows libraries to be
# present on unix systems which as far as I can tell is easiest to do by just copying them over from Windows.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -Wno-unused -Wpedantic -Wno-logical-op-parentheses -Wno-c++26-extensions -Wno-pre-c++26-compat")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ftrivial-auto-var-init=pattern")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wno-deprecated-declarations")
else()
    # AI generated MSVC flag equivalent nonsense:

    # /W4 is the practical equivalent of -Wall. (MSVC's actual /Wall is extremely noisy).
    # /Zi generates debugging information (like -g).
    # /permissive- enforces strict C++ standard conformance (like -Wpedantic).
    # /wd4100 /wd4101 /wd4189 disable specific "unused" warnings.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /Zi /permissive- /wd4100 /wd4101 /wd4189")

    # /RTC1 enables runtime checks, including catching uninitialized locals by filling them
    # with a pattern (0xCC), which is the MSVC equivalent of -ftrivial-auto-var-init=pattern.
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /RTC1")

    # /O2 is MSVC's equivalent to -O3 (maximizes speed).
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")

    # Toggles off some features which make MSVC sad, such as use of C++26 extensions in C++23 mode.
    add_compile_definitions(MSVC_COMPAT)

    # Force MSVC to place executables sanely in the root of the build folder.
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_BINARY_DIR}")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_BINARY_DIR}")
endif()

# Depencencies ---------------------------------------------------------------------------------------------------------

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    # Other operating systems just have development packages, windows requires an extra step.
    set(SDL3_DIR "lib/SDL3-3.2.24/cmake")
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    # Can't just link the system library anymore due to ABI breakage in libc++, how cute, isn't C++ just so great.
    # Technically C++ does not have a stable ABI in the first place so the fact that this is not the default
    # behavior is obviously a huge mistake, but that is the case for most of the C++ ecosystem.
    link_libraries(/opt/homebrew/Cellar/llvm/21.1.3/lib/c++/libc++.dylib)
endif()

# Depencencies ---------------------------------------------------------------------------------------------------------

# Only SDL3 is a dependency, text and images which are handled by their other libraries are easily implemented by hand.
# SDL2 is obviously not used because it is end-of-life and only further supported through an SDL3 compatibility wrapper.
# SDL is used as a convenient platform-generic IO, not a game library, outputting the software pixel renderer, the
# software sound renderer, and serving as a source for input logic. It is also used to load runtime libraries.
find_package(SDL3 REQUIRED)

# Targets --------------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE GAME_SOURCES CONFIGURE_DEPENDS src/*.cpp src/*.mm)
file(GLOB_RECURSE GAME_OBJECT_SOURCES CONFIGURE_DEPENDS object/*.cpp)

if(NOT HOT_RELOAD)
    add_executable(bubble ${GAME_SOURCES})
    target_include_directories(bubble PRIVATE include lib/stb)
    target_link_libraries(bubble PRIVATE SDL3::SDL3)
endif()

foreach(obj_src ${GAME_OBJECT_SOURCES})
    get_filename_component(obj_name ${obj_src} NAME_WE)

    add_library(${obj_name} SHARED ${obj_src})

    target_link_libraries(${obj_name} PRIVATE SDL3::SDL3)
    target_include_directories(${obj_name} PRIVATE include lib/stb)

    # Set output directory for shared objects, also for Windows dll files.
    set_target_properties(${obj_name} PROPERTIES
        PREFIX ""
        SUFFIX ".object"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/obj"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/obj"

        # Overrides for esoteric build environments (MSVC).
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}/obj"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}/obj"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}/obj"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}/obj"
    )
endforeach()

# Resources ------------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE RESOURCES res/*)

foreach(RESOURCE ${RESOURCES})
    file(COPY ${RESOURCE} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/res")
endforeach()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    file(COPY "lib/SDL3-3.2.24/lib/x64/SDL3.dll" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()
