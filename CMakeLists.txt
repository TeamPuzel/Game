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
#
# The MSVC build conditionally avoids compiling everything guarded by BUBBLE_REFLECTION_ENABLED.
# That is because the development setup requires clang-p2996 to use C++26 reflection for automatic
# derivation of `serialize` and `rebuild` which are used for saving files through the stage editor
# and hot reloading of game objects at runtime respectively. The MSVC build is usable but not intended for development.

set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_OPENGL_CRT_EFFECT "Use a CRT fragment shader" OFF)

if(NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC")
    set(CMAKE_CXX_STANDARD 26)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -Wno-unused -Wunused-value -Wpedantic -Wno-logical-op-parentheses -Wno-pre-c++26-compat -Wno-tautological-compare")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ftrivial-auto-var-init=pattern")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wno-deprecated-declarations")
    add_compile_definitions(_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE)

    if (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
        # We are completely static and pure, there is no reason to arbitrarily restrict the versions we can
        # run on based on the SDK. This overrides the configuration to allow all arm64 versions of macOS.
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmacosx-version-min=11")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-w")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-w")
    endif()

    if(DEFINED CUSTOM_CXX_INCLUDE_DIR AND DEFINED CUSTOM_CXX_LIB_DIR)
        # Reflection.
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -nostdlib++ -nostdinc++ -fno-blocks -freflection -freflection-latest -fexpansion-statements")
        add_compile_definitions(BUBBLE_REFLECTION_ENABLED)

        set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES "${CUSTOM_CXX_INCLUDE_DIR}")

        if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
            link_libraries(
                "${CUSTOM_CXX_LIB_DIR}/libc++.a"
                "${CUSTOM_CXX_LIB_DIR}/libc++abi.a"
                "${CUSTOM_CXX_LIB_DIR}/libunwind.a"
            )
        endif()
    endif()
else()
    set(CMAKE_CXX_STANDARD 23)
    # MSVC flag equivalent nonsense.

    # /W4 is the practical equivalent of -Wall.
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
    add_compile_definitions(BUBBLE_MSVC_COMPAT)

    # Force MSVC to place executables sanely in the root of the build folder.
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_BINARY_DIR}")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_BINARY_DIR}")

    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ENTRY:mainCRTStartup")
endif()

add_compile_definitions(BUBBLE_TEAMNAME="TeamPuzel")
add_compile_definitions(BUBBLE_APPNAME="Bubble Bobble DX")

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

if(USE_OPENGL_CRT_EFFECT)
    find_package(OpenGL REQUIRED)
    add_compile_definitions(USE_OPENGL_CRT_EFFECT)
endif()

# Targets --------------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE GAME_SOURCES CONFIGURE_DEPENDS src/*.cpp src/*.mm)
file(GLOB_RECURSE GAME_OBJECT_SOURCES CONFIGURE_DEPENDS object/*.cpp)

if(NOT HOT_RELOAD)
    if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
        add_executable(bubble WIN32 ${GAME_SOURCES} templates/windows/windows_resources.rc)
    else()
        add_executable(bubble ${GAME_SOURCES})
    endif()

    target_include_directories(bubble PRIVATE include lib/stb)
    target_link_libraries(bubble PRIVATE SDL3::SDL3)

    if(USE_OPENGL_CRT_EFFECT)
        target_link_libraries(bubble PRIVATE OpenGL::GL)
    endif()
endif()

add_library(dyn MODULE ${GAME_OBJECT_SOURCES})

target_link_libraries(dyn PRIVATE SDL3::SDL3)
target_include_directories(dyn PRIVATE include)

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

# Bundle ---------------------------------------------------------------------------------------------------------------

if(BUNDLE_RELEASE AND NOT HOT_RELOAD AND ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    add_dependencies(bubble dyn)

    set(APP_RESOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/res")
    set(APP_OBJECT_DIR "${CMAKE_CURRENT_BINARY_DIR}/obj")
    set(APP_TEMPLATE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/templates/Bubble Bobble DX App")
    set(APP_BUNDLE_DIR "${CMAKE_CURRENT_BINARY_DIR}/Bubble Bobble DX.app")
    set(HOMEBREW_SDL "/opt/homebrew/opt/sdl3/lib/libSDL3.0.dylib")

    add_custom_command(TARGET bubble POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${APP_TEMPLATE_DIR}" "${APP_BUNDLE_DIR}"

        COMMAND ${CMAKE_COMMAND} -E make_directory "${APP_BUNDLE_DIR}/Contents/Frameworks"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${APP_BUNDLE_DIR}/Contents/MacOS"

        COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:bubble>" "${APP_BUNDLE_DIR}/Contents/MacOS/bubble"
        COMMAND ${CMAKE_COMMAND} -E copy "${HOMEBREW_SDL}" "${APP_BUNDLE_DIR}/Contents/Frameworks/"

        COMMAND ${CMAKE_COMMAND} -E copy_directory "${APP_RESOURCE_DIR}" "${APP_BUNDLE_DIR}/Contents/Resources/res"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${APP_OBJECT_DIR}" "${APP_BUNDLE_DIR}/Contents/Resources/obj"

        COMMAND install_name_tool -change "${HOMEBREW_SDL}" "@executable_path/../Frameworks/libSDL3.0.dylib" "${APP_BUNDLE_DIR}/Contents/MacOS/bubble"

        COMMAND codesign --force --deep --sign - "${APP_BUNDLE_DIR}"

        COMMENT "Bundling for macOS..."
    )
endif()
