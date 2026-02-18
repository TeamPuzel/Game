option(ENABLE_VULKAN_SUPPORT "Enable Vulkan renderer" OFF)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

cmake_minimum_required(VERSION 3.21)
project(Renderer C CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -Wno-unused -Wpedantic -Wno-logical-op-parentheses")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ftrivial-auto-var-init=pattern")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

    if (ENABLE_VULKAN_SUPPORT)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_VULKAN_SUPPORT")
    endif()
endif()

if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(SDL3_DIR "lib/SDL3-3.2.24/cmake")
    file(COPY "lib/SDL3-3.2.24/lib/x64/SDL3.dll" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

# if (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
#     # Can't just link the system library anymore due to ABI breakage in libc++, how cute
#     link_libraries(/opt/homebrew/Cellar/llvm/21.1.3/lib/c++/libc++.dylib)
# endif()

# Depencencies ---------------------------------------------------------------------------------------------------------

find_package(SDL3 REQUIRED)
if (ENABLE_VULKAN_SUPPORT)
    find_package(Vulkan REQUIRED)
endif()

# Targets --------------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE RENDERER_SOURCES CONFIGURE_DEPENDS src/*.cpp)
file(GLOB_RECURSE RENDERER_HEADERS CONFIGURE_DEPENDS src/*.hpp)

add_executable(renderer ${RENDERER_SOURCES} ${RENDERER_HEADERS})
target_include_directories(renderer PRIVATE include)
target_link_libraries(renderer PRIVATE SDL3::SDL3)

if (ENABLE_VULKAN_SUPPORT)
    target_link_libraries(renderer PRIVATE Vulkan::Vulkan)
endif()

# Resources ------------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE RESOURCES res/*)

foreach(RESOURCE ${RESOURCES})
    file(COPY ${RESOURCE} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/res")
endforeach()
