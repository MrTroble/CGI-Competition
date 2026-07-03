#pragma once

#include "glm/glm.hpp"

struct Rng {
    glm::uvec2 seed;
};

Rng rngInit(uint32_t i, uint32_t j, uint32_t offset, uint32_t seed);

// returns random float between (0,1]
float rngRndFloat(Rng& rng);

// returns random vec2 between (0,1]^2
glm::vec2 rngRndVec2(Rng& rng);

// returns random vec3 between (0,1]^3
glm::vec3 rngRndVec3(Rng& rng);
