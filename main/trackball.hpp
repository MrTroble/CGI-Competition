#pragma once

#include <glm/glm.hpp>
#include <chrono>

#define SPEED 4.0

// https://github.com/ishikawash/glfw-example/blob/master/normal_map_demo/trackball.cpp
struct Trackball {
	bool isBeingDragged = false;
	glm::vec3 center = glm::vec3(0.0);
	glm::vec3 startPosition = glm::vec3(0.0);
	float radius = 2.0;
	glm::mat4 view;
	glm::vec3 cameraMovement = glm::vec3(0.0);
	std::chrono::time_point<std::chrono::system_clock> lastPositionUpdate;
	glm::vec3 cameraPosition = glm::vec3(-1.0f, 3.0f, 0.0f);
	// relative to camera position
	glm::vec3 lookAt = glm::vec3(1.0f, 0.0, 0.0);
};

struct MainContext;

glm::vec3 trackball_map_to_sphere(Trackball& trackball, glm::vec2 point);
void trackball_drag_start(MainContext& context);
bool trackball_drag_update(MainContext& context);
void trackball_drag_end(MainContext& context);
bool trackball_move_camera(MainContext& context);
