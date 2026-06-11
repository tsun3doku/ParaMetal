cmake_minimum_required(VERSION 3.21)
project(ParaMetal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt6 requires AUTOMOC for Q_OBJECT macros
set(CMAKE_AUTOMOC ON)

# Find required packages
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)
find_package(Vulkan REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/libs/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/libs/pybind11)

file(GLOB SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/*.cpp")
file(GLOB HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/*.hpp"
                  "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/*.h")

file(GLOB_RECURSE NODEGRAPH_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/nodegraph/*.cpp")
file(GLOB_RECURSE NODEGRAPH_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/nodegraph/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/nodegraph/*.h")

file(GLOB FRAMEGRAPH_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/framegraph/*.cpp")
file(GLOB FRAMEGRAPH_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/framegraph/*.hpp")

file(GLOB VULKAN_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/vulkan/*.cpp")
file(GLOB VULKAN_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/vulkan/*.hpp")

file(GLOB HEAT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/heat/*.cpp")
file(GLOB HEAT_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/heat/*.hpp")

file(GLOB REMESHER_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/mesh/remesher/*.cpp")
file(GLOB REMESHER_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/mesh/remesher/*.hpp")

file(GLOB RENDER_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/render/*.cpp")
file(GLOB RENDER_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/render/*.hpp")
file(GLOB RENDERERS_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/renderers/*.cpp")
file(GLOB RENDERERS_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/renderers/*.hpp")
file(GLOB SCENE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/scene/*.cpp")
file(GLOB SCENE_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/scene/*.hpp")
file(GLOB VORONOI_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/voronoi/*.cpp")
file(GLOB VORONOI_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/voronoi/*.hpp")
file(GLOB SPATIAL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/spatial/*.cpp")
file(GLOB SPATIAL_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/spatial/*.hpp")
file(GLOB UTIL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/util/*.cpp")
file(GLOB UTIL_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/util/*.hpp")
file(GLOB UTIL_HEADERS_H CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/util/*.h")
file(GLOB MESH_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/mesh/*.cpp")
file(GLOB MESH_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/mesh/*.hpp")
file(GLOB APP_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/app/*.cpp")
file(GLOB APP_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/app/*.hpp")
file(GLOB APP_HEADERS_H CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/app/*.h")
file(GLOB RUNTIME_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/runtime/*.cpp")
file(GLOB RUNTIME_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/runtime/*.hpp")
file(GLOB CONTACT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/contact/*.cpp")
file(GLOB CONTACT_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/contact/*.hpp")
file(GLOB DOMAIN_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/domain/*.hpp")

file(GLOB_RECURSE PY_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/py/*.cpp")
file(GLOB_RECURSE PY_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/py/*.hpp")

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS} ${NODEGRAPH_SOURCES} ${NODEGRAPH_HEADERS} ${FRAMEGRAPH_SOURCES} ${FRAMEGRAPH_HEADERS} ${VULKAN_SOURCES} ${VULKAN_HEADERS} ${HEAT_SOURCES} ${HEAT_HEADERS} ${REMESHER_SOURCES} ${REMESHER_HEADERS} ${RENDER_SOURCES} ${RENDER_HEADERS} ${RENDERERS_SOURCES} ${RENDERERS_HEADERS} ${SCENE_SOURCES} ${SCENE_HEADERS} ${VORONOI_SOURCES} ${VORONOI_HEADERS} ${SPATIAL_SOURCES} ${SPATIAL_HEADERS} ${UTIL_SOURCES} ${UTIL_HEADERS} ${UTIL_HEADERS_H} ${MESH_SOURCES} ${MESH_HEADERS} ${APP_SOURCES} ${APP_HEADERS} ${APP_HEADERS_H} ${RUNTIME_SOURCES} ${RUNTIME_HEADERS} ${CONTACT_SOURCES} ${CONTACT_HEADERS} ${DOMAIN_HEADERS} ${PY_SOURCES} ${PY_HEADERS})

# Include dir
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/nodegraph
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/framegraph
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/vulkan
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/heat
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/mesh/remesher
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/render
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/renderers
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/scene
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/voronoi
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/spatial
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/util
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/mesh
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/app
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/contact
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/domain
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/libs
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/libs/glm
    ${Python3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/libs/stb
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/libs/nanoflann/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/libs/tinyobjloader
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/libs/tinyply/source
    ${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/libs/entt/src
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Vulkan::Vulkan
    OpenMP::OpenMP_CXX
    pybind11::embed
    Python3::Python
)

if(WIN32)
    set(RESOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/resources/ParaMetal.rc")
    target_sources(${PROJECT_NAME} PRIVATE ${RESOURCE_FILES})
endif()

# Windows only: use windeployqt to copy Qt runtime files
if(WIN32)
    # Find windeployqt 
    get_target_property(QT_QMAKE_EXECUTABLE Qt6::qmake IMPORTED_LOCATION)
    get_filename_component(QT_BIN_DIR "${QT_QMAKE_EXECUTABLE}" DIRECTORY)
    find_program(QT_WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${QT_BIN_DIR}")
    
    if(QT_WINDEPLOYQT_EXECUTABLE)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND "${QT_WINDEPLOYQT_EXECUTABLE}"
                --no-translations
                --no-system-d3d-compiler
                --no-opengl-sw
                --skip-plugin-types generic,networkinformation,tls
                "$<TARGET_FILE:${PROJECT_NAME}>"
            COMMENT "Running windeployqt to copy Qt dependencies..."
        )
    endif()
endif()


# Copy shaders, models, and textures folders to output dir
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/shaders"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/models"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/models"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal/textures"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/textures"
    COMMENT "Copying additional files to output directory..."
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ParaMetal"
    OUTPUT_NAME "parametal"
)

