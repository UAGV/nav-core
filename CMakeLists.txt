cmake_minimum_required(VERSION 3.18)
project(navcore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Warnings-as-errors on all major compilers
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# --------------------------------------------------------------------------- #
# Dependencies                                                                #
# --------------------------------------------------------------------------- #
find_package(pybind11 REQUIRED)

# --------------------------------------------------------------------------- #
# Python extension                                                             #
# --------------------------------------------------------------------------- #
pybind11_add_module(_navcore src/bindings.cpp)
target_include_directories(_navcore PRIVATE include)
install(TARGETS _navcore DESTINATION navcore)

# --------------------------------------------------------------------------- #
# C++ unit tests (optional, built when BUILD_TESTS=ON)                        #
# --------------------------------------------------------------------------- #
option(BUILD_TESTS "Build C++ unit tests" OFF)

if(BUILD_TESTS)
    find_package(Catch2 3 QUIET)
    if(NOT Catch2_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG        v3.4.0
        )
        FetchContent_MakeAvailable(Catch2)
    endif()

    add_executable(navcore_tests
        tests/cpp/test_quaternion.cpp
        tests/cpp/test_frames.cpp
        tests/cpp/test_gnss_error.cpp
        tests/cpp/test_ranging.cpp
        tests/cpp/test_ekf.cpp
        tests/cpp/test_eskf.cpp
    )
    target_include_directories(navcore_tests PRIVATE include)
    target_link_libraries(navcore_tests PRIVATE Catch2::Catch2WithMain)
    include(CTest)
    include(Catch)
    catch_discover_tests(navcore_tests)
endif()
