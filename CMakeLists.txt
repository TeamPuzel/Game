# The main build file for the project

# Project --------------------------------------------------------------------------------------------------------------

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

cmake_minimum_required(VERSION 3.21)
project(Game C CXX)

# Language -------------------------------------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -Wno-unused -Wpedantic -Wno-logical-op-parentheses -Wno-c++26-extensions -Wno-pre-c++26-compat")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ftrivial-auto-var-init=pattern")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

# Depencencies ---------------------------------------------------------------------------------------------------------

if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(SDL3_DIR "lib/SDL3-3.2.24/cmake")
endif()

if (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    # Can't just link the system library anymore due to ABI breakage in libc++, how cute, isn't C++ just so great.
    link_libraries(/opt/homebrew/Cellar/llvm/21.1.3/lib/c++/libc++.dylib)
endif()

# Depencencies ---------------------------------------------------------------------------------------------------------

find_package(SDL3 REQUIRED)

# Targets --------------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE GAME_SOURCES CONFIGURE_DEPENDS src/*.cpp src/*.mm)
file(GLOB_RECURSE GAME_OBJECT_SOURCES CONFIGURE_DEPENDS object/*.cpp)

if(NOT HOT_RELOAD)
    add_executable(bubble ${GAME_SOURCES})
    target_include_directories(bubble PRIVATE include)
    target_link_libraries(bubble PRIVATE SDL3::SDL3)
endif()

foreach(obj_src ${GAME_OBJECT_SOURCES})
    get_filename_component(obj_name ${obj_src} NAME_WE)

    add_library(${obj_name} SHARED ${obj_src} src/rt/selector.cpp)
    target_link_libraries(${obj_name} PRIVATE SDL3::SDL3)
    target_include_directories(${obj_name} PRIVATE include)

    # Set output directory for shared objects, also for Windows dll files.
    set_target_properties(${obj_name} PROPERTIES
        PREFIX ""
        SUFFIX ".object"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/obj"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/obj"
    )
endforeach()

# Resources ------------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE RESOURCES res/*)

foreach(RESOURCE ${RESOURCES})
    file(COPY ${RESOURCE} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/res")
endforeach()

if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    file(COPY "lib/SDL3-3.2.24/lib/x64/SDL3.dll" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()
