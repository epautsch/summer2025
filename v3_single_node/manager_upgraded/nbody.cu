#include <cuda_runtime.h>
#include <math.h>

struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float mass;
};

__global__ void nbody_kernel(Particle* particles, int num_particles, float G, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_particles) {
        float fx = 0.0f;
        float fy = 0.0f;
        float fz = 0.0f;

        for (int j = 0; j < num_particles; ++j) {
            if (i != j) {
                float dx = particles[j].x - particles[i].x;
                float dy = particles[j].y - particles[i].y;
                float dz = particles[j].z - particles[i].z;

                float dist_sq = dx * dx + dy * dy + dz * dz;
                float dist = sqrtf(dist_sq);

                float force = G * particles[i].mass * particles[j].mass / (dist_sq + 1e-6f); // Add a small constant to avoid division by zero

                fx += force * dx / dist;
                fy += force * dy / dist;
                fz += force * dz / dist;
            }
        }

        particles[i].vx += (fx / particles[i].mass) * dt;
        particles[i].vy += (fy / particles[i].mass) * dt;
        particles[i].vz += (fz / particles[i].mass) * dt;

        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
    }
}
