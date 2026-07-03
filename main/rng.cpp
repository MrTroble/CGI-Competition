#include "rng.hpp"
#include <bit>


float uintToFloat(uint32_t x) {
    uint32_t a = 0x3f800000;
    uint32_t u = a | (x >> 9);

    return std::bit_cast<float>(u) - 1.f;
}

Rng rngInit(uint32_t i, uint32_t j, uint32_t offset, uint32_t seed) {
    glm::uvec2 state = glm::uvec2(i, j) ^ glm::uvec2(offset << 16, (seed + offset) << 16);
    return Rng{ state };
}

// returns random float between (0,1]
float rngRndFloat(Rng& rng) {
    // PCG2D, as described here: https://jcgt.org/published/0009/03/02/
    rng.seed = 1664525u * rng.seed + 1013904223u;
    rng.seed.x += 1664525u * rng.seed.y;
    rng.seed.y += 1664525u * rng.seed.x;
    rng.seed ^= (rng.seed >> 16u);
    rng.seed.x += 1664525u * rng.seed.y;
    rng.seed.y += 1664525u * rng.seed.x;
    rng.seed ^= (rng.seed >> 16u);
    // return float from seed
    return 1.f - uintToFloat(rng.seed.x);
}

glm::vec2 rngRndVec2(Rng& rng) {
    // PCG2D, as described here: https://jcgt.org/published/0009/03/02/
    rng.seed = 1664525u * rng.seed + 1013904223u;
    rng.seed.x += 1664525u * rng.seed.y;
    rng.seed.y += 1664525u * rng.seed.x;
    rng.seed ^= (rng.seed >> 16u);
    rng.seed.x += 1664525u * rng.seed.y;
    rng.seed.y += 1664525u * rng.seed.x;
    rng.seed ^= (rng.seed >> 16u);
    // return float from seed
    return 1.f - glm::vec2(uintToFloat(rng.seed.x), uintToFloat(rng.seed.y));
}

// returns random vec3 between (0,1]^3
glm::vec3 rngRndVec3(Rng& rng) {
    return glm::vec3(rngRndFloat(rng), rngRndVec2(rng));
}

