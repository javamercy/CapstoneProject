cmake_minimum_required(VERSION 3.10)
project(AKFSFSimulation)

set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

execute_process(
        COMMAND brew --prefix
        OUTPUT_VARIABLE BREW_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

include_directories(${BREW_PREFIX}/include)

find_package(SDL2 REQUIRED)
find_package(SDL2_ttf REQUIRED)

if (NOT SDL2_LIBRARIES)
    set(SDL2_LIBRARIES SDL2)
    set(SDL2_TTF_LIBRARIES SDL2_ttf)
endif ()

link_directories(${BREW_PREFIX}/lib)

set(SRC_CPP
        src/main.cpp
        src/display.cpp
        src/beacons.cpp
        src/sensors.cpp
        src/simulation.cpp
        src/utils.cpp
        src/kalmanfilter.cpp
)

add_executable(${PROJECT_NAME} ${SRC_CPP})

target_link_libraries(${PROJECT_NAME}
        "-framework OpenGL"
        "-L${BREW_PREFIX}/lib"
        "-lSDL2"
        "-lSDL2_ttf"
        Eigen3::Eigen
)

add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/data/ $<TARGET_FILE_DIR:${PROJECT_NAME}>)
