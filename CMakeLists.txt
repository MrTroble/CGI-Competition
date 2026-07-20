cmake_minimum_required(VERSION 3.20)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project("TAA-Main")

find_package(Vulkan 1.4 REQUIRED COMPONENTS glslang shaderc_combined SPIRV-Tools)

#add_compile_options(-fsanitize=address)
#add_link_options(-fsanitize=address)

message(${Vulkan_GLSLC_EXECUTABLE})

include(FetchContent)

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/shader")
file(GLOB files "shader/*.vert" "shader/*.frag" "shader/*.rchit" "shader/*.rmiss" "shader/*.rgen")
list(APPEND PERMUTATION_SHADER "USE_ALBEDO" "USE_ALB_NORM")
foreach(file ${files})
  cmake_path(GET file FILENAME filename)
  add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/shader/${filename}.spv" COMMAND ${Vulkan_GLSLC_EXECUTABLE} $<$<CONFIG:Release>:-O> --target-env=vulkan1.2 -DINT_GLSL -c "${file}" -o "${CMAKE_BINARY_DIR}/shader/${filename}.spv" MAIN_DEPENDENCY ${file} WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/shader)
  list(APPEND SPV_TARGETS "${CMAKE_BINARY_DIR}/shader/${filename}.spv")
  if("${filename}" STREQUAL "geometry.frag")
    foreach(perm ${PERMUTATION_SHADER})
        set(LOCAL_NAME_PERM "${perm}_${filename}")
        add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/shader/${LOCAL_NAME_PERM}.spv" COMMAND ${Vulkan_GLSLC_EXECUTABLE} $<$<CONFIG:Release>:-O> --target-env=vulkan1.2 -DINT_GLSL -D${perm} -c "${file}" -o "${CMAKE_BINARY_DIR}/shader/${LOCAL_NAME_PERM}.spv" MAIN_DEPENDENCY ${file} WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/shader)
        list(APPEND SPV_TARGETS "${CMAKE_BINARY_DIR}/shader/${LOCAL_NAME_PERM}.spv")
    endforeach()
  endif()
endforeach()
add_custom_target(shaderTarget DEPENDS ${SPV_TARGETS})

add_subdirectory("submodules")
add_subdirectory("main")

configure_file("assets/provence.glb" "assets/provence.glb" COPYONLY)
configure_file("assets/cube.glb" "assets/cube.glb" COPYONLY)
configure_file("assets/DamagedHelmet.glb" "assets/DamagedHelmet.glb" COPYONLY)
configure_file("assets/MetalRoughSpheresNoTextures.glb" "assets/MetalRoughSpheresNoTextures.glb" COPYONLY)
configure_file("assets/MetalRoughSpheres.glb" "assets/MetalRoughSpheres.glb" COPYONLY)
configure_file("assets/IORTestGrid.glb" "assets/IORTestGrid.glb" COPYONLY)
