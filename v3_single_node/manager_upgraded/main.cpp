#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Structure to represent a particle
struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float mass;
};

// CUDA kernel to update particle positions
__global__ void nbody_kernel(Particle* particles, int num_particles, float G, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles) {
        // Calculate force on particle i
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

        // Update particle velocity and position
        particles[i].vx += (fx / particles[i].mass) * dt;
        particles[i].vy += (fy / particles[i].mass) * dt;
        particles[i].vz += (fz / particles[i].mass) * dt;

        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
    }
}

int main() {
    int num_particles = 1024;
    float G = 6.674e-11f;
    float dt = 0.01f;

    // Allocate host memory
    std::vector<Particle> host_particles(num_particles);

    // Initialize particles (example)
    for (int i = 0; i < num_particles; ++i) {
        host_particles[i].x = (float)rand() / RAND_MAX;
        host_particles[i].y = (float)rand() / RAND_MAX;
        host_particles[i].z = (float)rand() / RAND_MAX;
        host_particles[i].vx = 0.0f;
        host_particles[i].vy = 0.0f;
        host_particles[i].vz = 0.0f;
        host_particles[i].mass = 1.0f;
    }

    // Allocate device memory
    Particle* device_particles;
    cudaMalloc((void**)&device_particles, num_particles * sizeof(Particle));

    // Copy data from host to device
    cudaMemcpy(device_particles, host_particles.data(), num_particles * sizeof(Particle), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int block_size = 256;
    int grid_size = (num_particles + block_size - 1) / block_size;

    // Launch the kernel
    nbody_kernel<<<grid_size, block_size>>>(device_particles, num_particles, G, dt);

    // Copy data from device to host
    cudaMemcpy(host_particles.data(), device_particles, num_particles * sizeof(Particle), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_particles);

    // Print some results (example)
    std::cout << "Particle 0 x: " << host_particles[0].x << std::endl;

    return 0;
}
