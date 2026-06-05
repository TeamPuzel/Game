# The main build file for the project

# Project --------------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.21)
project(Game C CXX)

# Language -------------------------------------------------------------------------------------------------------------

# Technically the codebase uses C++23 with C++26 extensions enabled, but last time I tried MSVC was far behind
# so while not much is changed in the configurations that code had to be temporarily disabled.
# I cross compile to Windows, while for now it still has some shrinking market share, it is
# purely a target platform, not a serious development environment. This requires the STL and Windows libraries to be
# present on unix systems which as far as I can tell is easiest to do by just copying them over from Windows.
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -Wno-unused -Wunused-value -Wpedantic -Wno-logical-op-parentheses -Wno-c++26-extensions -Wno-pre-c++26-compat -Wno-tautological-compare")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ftrivial-auto-var-init=pattern")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wno-deprecated-declarations")
    add_compile_definitions(_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE)

    # Reflection.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -nostdinc++ -fno-blocks -stdlib=libc++ -freflection -freflection-latest -fexpansion-statements")

    if(DEFINED CUSTOM_CXX_INCLUDE_DIR AND DEFINED CUSTOM_CXX_LIB_DIR)
        set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES "${CUSTOM_CXX_INCLUDE_DIR}")
        link_directories("${CUSTOM_CXX_LIB_DIR}")

        if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
            link_libraries(
                "${CUSTOM_MAC_LIBCXX}"
                "${CUSTOM_CXX_LIB_DIR}/libunwind.1.0.dylib"
            )
        endif()
    endif()
else()
    # MSVC flag equivalent nonsense.

    # /W4 is the practical equivalent of -Wall. (MSVC's actual /Wall is extremely noisy).
    # /Zi generates debugging information (like -g). (UPDATE: needs Z7 or the dumb thing will race condition and die).
    # /permissive- enforces strict C++ standard conformance (like -Wpedantic).
    # /wd4100 /wd4101 /wd4189 disable specific "unused" warnings.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8 /W4 /Z7 /permissive- /wd4100 /wd4101 /wd4189")

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

    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ENTRY:mainCRTStartup")
endif()

# Depencencies ---------------------------------------------------------------------------------------------------------

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    # Other operating systems just have development packages, windows requires an extra step.
    set(SDL3_DIR "lib/SDL3-3.2.24/cmake")
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
    add_executable(bubble WIN32 ${GAME_SOURCES})
    target_include_directories(bubble PRIVATE include lib/stb)
    target_link_libraries(bubble PRIVATE SDL3::SDL3)
endif()

add_library(dyn MODULE ${GAME_OBJECT_SOURCES})

target_link_libraries(dyn PRIVATE SDL3::SDL3)
target_include_directories(dyn PRIVATE include lib/stb)

set_target_properties(dyn PROPERTIES
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

# Resources ------------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE RESOURCES res/*)

foreach(RESOURCE ${RESOURCES})
    file(COPY ${RESOURCE} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/res")
endforeach()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    file(COPY "lib/SDL3-3.2.24/lib/x64/SDL3.dll" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()
