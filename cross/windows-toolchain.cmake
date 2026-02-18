# This file is not used for the Windows build process natively.
# This toolchain is for cross-compilation to Windows from Unix systems.

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang)
set(CMAKE_LINKER lld)
set(CMAKE_RC_COMPILER llvm-rc)
set(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS clang-scan-deps)

# These two are all you need to cross compile to Windows.
# (assuming you even want to use the STL or other Windows APIs).
# I think with clever assumptions and runtime linking of SDL3
# it would actually be possible to cross compile without these.
# And of course dropping the C++ standard library, at which point
# it's easier to just use a less bloated programming language
# and completely avoid this issue.
set(MSVC_DIR "/Users/teampuzel/CXX/WINLIB/msvc" CACHE PATH "Path to the msvc directory")
set(WIN_SDK_DIR "/Users/teampuzel/CXX/WINLIB/sdk" CACHE PATH "Path to the windows sdk directory")

# We need to configure the program to use a normal entry point instead of the default Windows garbage.
# On top of that SDL is being "helpful" and defining garbage macros that no programmer should ever pollute
# the namespace with, defining fucking main is gettext levels of bad practice. Fuck you. Seriously.
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Xlinker /ENTRY:mainCRTStartup")
add_compile_definitions(SDL_MAIN_HANDLED)

include_directories(
    "${MSVC_DIR}/include"
    "${WIN_SDK_DIR}/Include/10.0.26100.0/um"
    "${WIN_SDK_DIR}/Include/10.0.26100.0/ucrt"
    "${WIN_SDK_DIR}/Include/10.0.26100.0/shared"
)

link_directories(
    "${MSVC_DIR}/lib/x64"
    "${WIN_SDK_DIR}/Lib/10.0.26100.0/um/x64"
    "${WIN_SDK_DIR}/Lib/10.0.26100.0/ucrt/x64"
)

# We can now cross-target Windows with the proper Microsoft ABI and without bloat like mingw.
set(CMAKE_C_COMPILER_TARGET x86_64-pc-windows-msvc)
set(CMAKE_CXX_COMPILER_TARGET x86_64-pc-windows-msvc)
