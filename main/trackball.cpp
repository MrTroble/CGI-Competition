#include "MainContext.hpp"
#include "trackball.hpp"
#include "glm/gtc/epsilon.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include <chrono>

using namespace std::chrono_literals;

glm::vec3 trackball_map_to_sphere(Trackball& trackball, glm::vec2 point) {
	float safe_radius = trackball.radius - 1.0f;
	if (glm::length(point) > safe_radius) {
		float theta = std::atan2(point.y, point.x);
		point.x = safe_radius * cos(theta);
		point.y = safe_radius * sin(theta);
	}

	float length_squared = point.x * point.x + point.y * point.y;
	float z = sqrtf(trackball.radius * trackball.radius - length_squared);
	glm::vec3 q(point.x, point.y, z);
	return normalize(q / trackball.radius);
}

void trackball_drag_start(MainContext& context) {
	if (context.trackball.isBeingDragged) {
		return;
	}
	context.trackball.isBeingDragged = true;

	int width, height;
    glfwGetWindowSize(context.window, &width, &height);

    double xpos, ypos;
    glfwGetCursorPos(context.window, &xpos, &ypos);

    glm::vec2 point((2.0f * xpos - width) / width, (2.0f * ypos - height) / height);
    glm::vec3 current_position = trackball_map_to_sphere(context.trackball, point);
    context.trackball.startPosition = current_position;
}

bool trackball_drag_update(MainContext& context) {
	if (!context.trackball.isBeingDragged) {
		return false;
	}

	int width, height;
    glfwGetWindowSize(context.window, &width, &height);

    double xpos, ypos;
    glfwGetCursorPos(context.window, &xpos, &ypos);

    // Map into range [-1, 1]
    glm::vec2 point((2.0f * xpos - width) / width, (2.0f * ypos - height) / height);
    glm::vec3 current_position = trackball_map_to_sphere(context.trackball, point);

    glm::vec3 axis = glm::cross(context.trackball.startPosition, current_position);
    float angle = glm::acos(glm::dot(context.trackball.startPosition, current_position));


    if (!glm::all(glm::epsilonEqual(axis, glm::vec3(0.0f), 0.0001f))) {
        glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), angle, axis);
        context.trackball.lookAt = glm::vec3(rotation * glm::vec4(context.trackball.lookAt, 0));
        context.trackball.view = glm::lookAt(context.trackball.cameraPosition, context.trackball.cameraPosition + context.trackball.lookAt, glm::vec3(0.0, -1.0, 0.0));
        context.trackball.startPosition = current_position;
        return true;
    }

    return false;
}

void trackball_drag_end(MainContext& context) {
	if (!context.trackball.isBeingDragged) {
		return;
	}
	context.trackball.isBeingDragged = false;
}


bool trackball_move_camera(MainContext& context) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - context.trackball.lastPositionUpdate);

    // Just don't bother to update more than 60 times a second
    if(elapsed < 16ms) {
        return false;
    }
    context.trackball.lastPositionUpdate = std::chrono::system_clock::now();

    if (context.trackball.cameraMovement == glm::vec3(0.0)) {
        return false;
    }

    const float scalingFactor = (float)elapsed.count() / 1000.0;

    glm::vec3 moveBy = scalingFactor * context.trackball.cameraMovement;
    glm::vec3 rotated = glm::vec3(glm::inverse(glm::lookAt(glm::vec3(0.0), context.trackball.lookAt, glm::vec3(0.0, -1.0, 0.0))) * glm::vec4(moveBy, 1.0));
    rotated.y = moveBy.y;
    context.trackball.cameraPosition += rotated;

    context.trackball.view = glm::lookAt(context.trackball.cameraPosition, context.trackball.cameraPosition + context.trackball.lookAt, glm::vec3(0.0, -1.0, 0.0));
    return true;
}
