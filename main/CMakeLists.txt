add_executable (taa-main "main.cpp"  "Loader.cpp" "trackball.cpp" "rng.cpp")

set_property(TARGET taa-main PROPERTY CXX_STANDARD 20)
target_include_directories(taa-main PUBLIC ../submodules/glfw ../submodules/imgui ../submodules/glm ../submodules/tinygltf ../submodules/flatbuffers/include)
target_link_libraries(taa-main PUBLIC tinygltf Vulkan::Vulkan glfw imgui Vulkan::glslang Vulkan::shaderc_combined Vulkan::SPIRV-Tools)
add_dependencies(taa-main shaderTarget)
