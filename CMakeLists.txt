cmake_minimum_required(VERSION 3.16)
project(UrbanRoadDamage LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# Dependencies
# =============================================================================
# 1. OpenCV (Image I/O, drawing bounding boxes, resize)
find_package(OpenCV QUIET)
if (OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(WARNING "OpenCV not found in system path. Please set OpenCV_DIR to build full road_damage binary.")
endif()

# 2. ONNX Runtime (Conditional for high-speed edge C++ inference)
# Set ONNXRUNTIME_ROOT_DIR to point to ONNX Runtime C++ SDK if installed
if (DEFINED ONNXRUNTIME_ROOT_DIR)
    message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_ROOT_DIR}")
    include_directories(${ONNXRUNTIME_ROOT_DIR}/include)
    link_directories(${ONNXRUNTIME_ROOT_DIR}/lib)
    add_compile_definitions(HAVE_ONNXRUNTIME)
    set(ONNXRUNTIME_LIBS onnxruntime)
else()
    message(STATUS "ONNX Runtime directory not specified. C++ CLI will build in stub detection mode.")
    set(ONNXRUNTIME_LIBS "")
endif()

# =============================================================================
# Core Sources
# =============================================================================
set(SEVERITY_SOURCES
    src/severity/severity.h
    src/severity/severity.cpp
)

set(EXPORT_SOURCES
    src/export/json_writer.h
    src/export/json_writer.cpp
    src/export/csv_writer.h
    src/export/csv_writer.cpp
)

set(DETECTION_SOURCES
    src/detection/inference.h
    src/detection/inference.cpp
)

# =============================================================================
# Executables
# =============================================================================
if (OpenCV_FOUND)
    add_executable(road_damage
        src/main.cpp
        ${DETECTION_SOURCES}
        ${SEVERITY_SOURCES}
        ${EXPORT_SOURCES}
    )
    target_link_libraries(road_damage PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIBS})
endif()

# =============================================================================
# Testing Harness (CTest)
# =============================================================================
enable_testing()

if (OpenCV_FOUND)
    add_executable(test_severity_cpp
        tests/test_severity.cpp
        ${SEVERITY_SOURCES}
        ${DETECTION_SOURCES}
    )
    target_link_libraries(test_severity_cpp PRIVATE ${OpenCV_LIBS})
    add_test(NAME SeverityRulesTest COMMAND test_severity_cpp)
endif()
