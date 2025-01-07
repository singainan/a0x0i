cmake_minimum_required(VERSION 3.22)

project(Optimlir)
set(CMAKE_CXX_STANDARD 20)
find_package(Threads) # Detect this before enabling strict compilation

find_package(Quick)
if (Quick_FOUND)
    prefer_mold()
    use_common_macros()
    vcpkg_init()
endif()

add_subdirectory(src)
