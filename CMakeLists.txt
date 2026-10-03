cmake_minimum_required(VERSION 3.10)
project(SimpleBitboardChessEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_executable(chessengine main.cpp)
target_compile_options(chessengine PRIVATE -O2 -Wall -Wextra)

if(NOT WIN32)
  # the asynchronous stdin reader uses std::thread on POSIX
  find_package(Threads REQUIRED)
  target_link_libraries(chessengine PRIVATE Threads::Threads)
endif()

enable_testing()
add_test(NAME perft_and_consistency COMMAND chessengine test)