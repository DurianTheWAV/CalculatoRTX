# =============================================================================
#  CalculatoRTX - calculatrice entièrement ray tracée (OptiX), DLSS et calcul GPU (CUDA)
#  Plateformes : Linux (CachyOS / Arch, Ubuntu...) et Windows 10/11 (MSVC 2022)
# =============================================================================
cmake_minimum_required(VERSION 3.24)

# Architecture cible par défaut : Ada Lovelace (RTX 4070 Ti = AD104, sm_89).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

project(CalculatoRTX VERSION 1.0.0 LANGUAGES C CXX CUDA)

include(FetchContent)

option(CRTX_ENABLE_DLSS "Intégrer NVIDIA DLSS (SDK téléchargé automatiquement si absent)" ON)
option(CRTX_FETCH_DEPS "Télécharger automatiquement les dépendances manquantes (GLFW, en-têtes OptiX, SDK DLSS)" ON)
option(CRTX_BUILD_TESTS "Construire les tests unitaires CPU du moteur de calcul" ON)
option(CRTX_DLSS_DEV_RUNTIME "Utiliser la bibliothèque DLSS de développement (overlay de debug)" OFF)
set(CRTX_PTX_ARCH "89" CACHE STRING "Architecture virtuelle du PTX OptiX (compute_XX)")
set(OptiX_INSTALL_DIR "" CACHE PATH "Racine du SDK OptiX (contient include/optix.h)")
set(CRTX_OPTIX_GIT_TAG "v9.1.0" CACHE STRING "Version des en-têtes OptiX téléchargés (v9.1.0, v9.0.0 pour les pilotes plus anciens)")
set(DLSS_SDK_DIR "" CACHE PATH "Racine du SDK DLSS (contient include/nvsdk_ngx.h)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Type de compilation" FORCE)
endif()

find_package(CUDAToolkit 12.4 REQUIRED)
find_package(Vulkan REQUIRED)
message(STATUS "CUDA ${CUDAToolkit_VERSION} (${CUDAToolkit_BIN_DIR}), architectures : ${CMAKE_CUDA_ARCHITECTURES}")

# ----------------------------------------------------------------------------- GLFW
find_package(glfw3 3.3 CONFIG QUIET)
if(NOT glfw3_FOUND)
    if(NOT CRTX_FETCH_DEPS)
        message(FATAL_ERROR "GLFW introuvable (installez 'glfw' ou activez CRTX_FETCH_DEPS)")
    endif()
    message(STATUS "GLFW : téléchargement de la version 3.4")
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(glfw)
endif()

# ----------------------------------------------------------------------------- OptiX (en-têtes)
set(_optix_hints "${OptiX_INSTALL_DIR}" "$ENV{OptiX_INSTALL_DIR}" "$ENV{OPTIX_ROOT}" "$ENV{OPTIX_ROOT_DIR}"
                 /opt/optix /usr/local/optix /usr/include/optix)
if(WIN32)
    file(GLOB _optix_win LIST_DIRECTORIES true "C:/ProgramData/NVIDIA Corporation/OptiX SDK *")
    list(SORT _optix_win ORDER DESCENDING)
    list(APPEND _optix_hints ${_optix_win})
endif()
find_path(OPTIX_INCLUDE_DIR optix.h HINTS ${_optix_hints} PATH_SUFFIXES include NO_DEFAULT_PATH)
if(OPTIX_INCLUDE_DIR)
    file(STRINGS "${OPTIX_INCLUDE_DIR}/optix.h" _optix_ver REGEX "#define OPTIX_VERSION [0-9]+")
    string(REGEX MATCH "[0-9]+" _optix_ver "${_optix_ver}")
    if(_optix_ver LESS 80000)
        message(WARNING "OptiX ${_optix_ver} trop ancien (8.0+ requis pour SER) : utilisation des en-têtes officiels 9.1")
        unset(OPTIX_INCLUDE_DIR CACHE)
        set(OPTIX_INCLUDE_DIR "")
    endif()
endif()
if(NOT OPTIX_INCLUDE_DIR)
    if(NOT CRTX_FETCH_DEPS)
        message(FATAL_ERROR "En-têtes OptiX introuvables : définissez -DOptiX_INSTALL_DIR=...")
    endif()
    message(STATUS "OptiX : téléchargement des en-têtes officiels NVIDIA/optix-dev ${CRTX_OPTIX_GIT_TAG}")
    FetchContent_Declare(optix_dev
        GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
        GIT_TAG ${CRTX_OPTIX_GIT_TAG}
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR _pas_de_cmake_)
    FetchContent_MakeAvailable(optix_dev)
    set(OPTIX_INCLUDE_DIR "${optix_dev_SOURCE_DIR}/include")
endif()
message(STATUS "OptiX : ${OPTIX_INCLUDE_DIR}")

# ----------------------------------------------------------------------------- DLSS (NGX)
set(CRTX_WITH_DLSS OFF)
if(CRTX_ENABLE_DLSS)
    set(_dlss_root "")
    foreach(_d "${DLSS_SDK_DIR}" "$ENV{DLSS_SDK_DIR}")
        if(_d AND EXISTS "${_d}/include/nvsdk_ngx.h")
            set(_dlss_root "${_d}")
            break()
        endif()
    endforeach()
    if(NOT _dlss_root AND CRTX_FETCH_DEPS)
        message(STATUS "DLSS : téléchargement du SDK officiel NVIDIA/DLSS v310.9.1 (quelques centaines de Mo)")
        FetchContent_Declare(dlss_sdk
            GIT_REPOSITORY https://github.com/NVIDIA/DLSS.git
            GIT_TAG v310.9.1
            GIT_SHALLOW TRUE
            GIT_SUBMODULES ""
            SOURCE_SUBDIR _pas_de_cmake_)
        FetchContent_MakeAvailable(dlss_sdk)
        set(_dlss_root "${dlss_sdk_SOURCE_DIR}")
    endif()
    if(_dlss_root)
        set(_dlss_variant rel)
        if(CRTX_DLSS_DEV_RUNTIME)
            set(_dlss_variant dev)
        endif()
        if(WIN32)
            set(_dlss_libdir "${_dlss_root}/lib/Windows_x86_64/x64")
            set(_crt d)  # /MD (runtime C dynamique, défaut CMake)
            if(CMAKE_MSVC_RUNTIME_LIBRARY MATCHES "^MultiThreaded(Debug)?$")
                set(_crt s)
            endif()
            set(DLSS_LIB_RELEASE "${_dlss_libdir}/nvsdk_ngx_${_crt}.lib")
            set(DLSS_LIB_DEBUG "${_dlss_libdir}/nvsdk_ngx_${_crt}_dbg.lib")
            file(GLOB DLSS_RUNTIME "${_dlss_root}/lib/Windows_x86_64/${_dlss_variant}/nvngx_dlss.dll")
        else()
            set(DLSS_LIB_RELEASE "${_dlss_root}/lib/Linux_x86_64/libnvsdk_ngx.a")
            set(DLSS_LIB_DEBUG "${DLSS_LIB_RELEASE}")
            file(GLOB DLSS_RUNTIME "${_dlss_root}/lib/Linux_x86_64/${_dlss_variant}/libnvidia-ngx-dlss.so.*")
        endif()
        if(EXISTS "${DLSS_LIB_RELEASE}")
            set(CRTX_WITH_DLSS ON)
            set(DLSS_INCLUDE_DIR "${_dlss_root}/include")
            message(STATUS "DLSS : ${_dlss_root}")
            if(NOT DLSS_RUNTIME)
                message(WARNING "Bibliothèque d'exécution DLSS introuvable : DLSS sera indisponible à l'exécution")
            endif()
        else()
            message(WARNING "SDK DLSS incomplet (${DLSS_LIB_RELEASE} absent) : compilation sans DLSS")
        endif()
    else()
        message(WARNING "SDK DLSS introuvable : compilation sans DLSS (-DDLSS_SDK_DIR=...)")
    endif()
endif()

# ----------------------------------------------------------------------------- programmes OptiX -> PTX
add_library(crtx_optix_ptx OBJECT src/render/device/Programs.cu)
set_target_properties(crtx_optix_ptx PROPERTIES CUDA_PTX_COMPILATION ON CUDA_ARCHITECTURES OFF)
target_include_directories(crtx_optix_ptx PRIVATE "${OPTIX_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_options(crtx_optix_ptx PRIVATE
    --gpu-architecture=compute_${CRTX_PTX_ARCH}
    --use_fast_math
    --generate-line-info
    -Wno-deprecated-gpu-targets
    $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler=/utf-8>)

set(CRTX_EMBEDDED_PTX "${CMAKE_CURRENT_BINARY_DIR}/generated/optix_programs_ptx.cpp")
add_custom_command(
    OUTPUT "${CRTX_EMBEDDED_PTX}"
    COMMAND ${CMAKE_COMMAND} "-DINPUT=$<TARGET_OBJECTS:crtx_optix_ptx>" "-DOUTPUT=${CRTX_EMBEDDED_PTX}"
            -DSYMBOL=g_optixProgramsPtx -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedFile.cmake"
    DEPENDS crtx_optix_ptx $<TARGET_OBJECTS:crtx_optix_ptx> "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedFile.cmake"
    COMMENT "Intégration du PTX OptiX dans l'exécutable"
    VERBATIM COMMAND_EXPAND_LISTS)

# ----------------------------------------------------------------------------- exécutable
add_executable(CalculatoRTX
    src/main.cpp
    src/app/App.cpp
    src/app/CalculatorController.cpp
    src/calc/CalcEngine.cu
    src/dlss/DlssUpscaler.cpp
    src/gpu/VulkanContext.cpp
    src/render/Kernels.cu
    src/render/OptixRenderer.cpp
    src/scene/CalculatorScene.cpp
    src/scene/Mesh.cpp
    src/scene/StrokeFont.cpp
    "${CRTX_EMBEDDED_PTX}")

target_include_directories(CalculatoRTX PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${OPTIX_INCLUDE_DIR}")
target_link_libraries(CalculatoRTX PRIVATE CUDA::cudart_static Vulkan::Vulkan glfw ${CMAKE_DL_LIBS})
if(NOT WIN32)
    find_package(Threads REQUIRED)
    target_link_libraries(CalculatoRTX PRIVATE Threads::Threads)
endif()
set_target_properties(CalculatoRTX PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Static
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Le moteur de calcul garde une arithmétique IEEE stricte (pas de --use_fast_math).
set_source_files_properties(src/render/Kernels.cu PROPERTIES COMPILE_OPTIONS "--use_fast_math")

target_compile_options(CalculatoRTX PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W3 /utf-8 /permissive- /MP>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -Wextra -Wno-missing-field-initializers>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets --extended-lambda>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/utf-8>)
target_compile_definitions(CalculatoRTX PRIVATE
    $<$<PLATFORM_ID:Windows>:NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS>)

if(CRTX_WITH_DLSS)
    target_compile_definitions(CalculatoRTX PRIVATE CRTX_WITH_DLSS=1)
    target_include_directories(CalculatoRTX PRIVATE "${DLSS_INCLUDE_DIR}")
    target_link_libraries(CalculatoRTX PRIVATE
        $<$<CONFIG:Debug>:${DLSS_LIB_DEBUG}> $<$<NOT:$<CONFIG:Debug>>:${DLSS_LIB_RELEASE}>)
    foreach(_rt IN LISTS DLSS_RUNTIME)
        add_custom_command(TARGET CalculatoRTX POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_rt}" "$<TARGET_FILE_DIR:CalculatoRTX>"
            COMMENT "Copie de la bibliothèque DLSS à côté de l'exécutable")
    endforeach()
endif()

install(TARGETS CalculatoRTX RUNTIME DESTINATION bin)
if(CRTX_WITH_DLSS AND DLSS_RUNTIME)
    install(FILES ${DLSS_RUNTIME} DESTINATION bin)
endif()

# ----------------------------------------------------------------------------- tests CPU
if(CRTX_BUILD_TESTS)
    enable_testing()
    add_executable(calc_host_test tests/CalcHostTest.cu)
    target_include_directories(calc_host_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    set_target_properties(calc_host_test PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    target_compile_options(calc_host_test PRIVATE
        -Wno-deprecated-gpu-targets $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler=/utf-8>)
    add_test(NAME calc_host_test COMMAND calc_host_test)
endif()

message(STATUS "--------------------------------------------------------------")
message(STATUS " CalculatoRTX ${PROJECT_VERSION}")
message(STATUS "   DLSS    : ${CRTX_WITH_DLSS}")
message(STATUS "   OptiX   : ${OPTIX_INCLUDE_DIR}")
message(STATUS "   CUDA    : ${CUDAToolkit_VERSION}, sm_${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "--------------------------------------------------------------")
