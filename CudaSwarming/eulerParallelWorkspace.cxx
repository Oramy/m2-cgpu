#include <cmath>
#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>
#include <random>
#include <vector>

#include "agent.hxx"
#include "vector.hxx"
#include "eulerParallelWorkspace.hxx"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 256

__global__ void compute_force_device(Agent* agents, Real na, Real rCohesion, Real wCohesion,
    Real wAlignment, Real wSeparation) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    // We check that our index is in the right range.
    if (k < na) {
        agents[k].compute_force(agents, na, k, rCohesion);

        agents[k].direction = agents[k].cohesion * wCohesion
            + agents[k].alignment * wAlignment
            + agents[k].separation * wSeparation;
    }
}
__global__ void euler_method(Agent * agents, Real na, Real dt, Real max_speed,
                        Real lx, Real ly, Real lz) {

    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(k < na) {
      agents[k].velocity += dt*agents[k].direction;

      double speed = agents[k].velocity.norm()/max_speed;
      if (speed > 1. && speed>0.) {
        agents[k].velocity /= speed;
      }
      agents[k].position += dt*agents[k].velocity;

      if(agents[k].position.x <40)
        agents[k].position.x = lx - 40;
      if(agents[k].position.x >lx - 40)
        agents[k].position.x = 40;
      if(agents[k].position.y <40)
        agents[k].position.y = ly - 40;
      if(agents[k].position.y >ly - 40)
        agents[k].position.y = 40;
      if(agents[k].position.z <40)
        agents[k].position.z = lz - 40;
      if(agents[k].position.z >lz - 40)
        agents[k].position.z = 40;
    }
}
EulerParallelWorkspace::EulerParallelWorkspace(ArgumentParser &parser): Workspace(parser){
    savepath = "boidsParallel.xyz";
    size_t arrayMemSize = sizeof(Agent) * na;
    cudaMalloc(&agentsDP, arrayMemSize);
}
EulerParallelWorkspace::~EulerParallelWorkspace() {
    cudaFree(agentsDP);
}

std::vector<time_t> EulerParallelWorkspace::move()
{
    std::vector<time_t> times;

    std::chrono::steady_clock::time_point start, end, start2, end2, start3, end3;

    start = std::chrono::steady_clock::now();
    
    // Compute forces applied on specific agent
    for(size_t k = 0; k< na; k++){
      agents[k].compute_force(agents, na, k, rCohesion);

      agents[k].direction = agents[k].cohesion*wCohesion
        + agents[k].alignment*wAlignment
        + agents[k].separation*wSeparation;
    }
    end = std::chrono::steady_clock::now();
    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    start3 = std::chrono::steady_clock::now();

    size_t arrayMemSize = sizeof(Agent) * na;
    cudaMemcpy(agentsDP, (void*) agents, arrayMemSize, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    end3 = std::chrono::steady_clock::now();
    dim3 block(BLOCK_SIZE);
    dim3 grid(int(ceil((float(na)/BLOCK_SIZE))));
    euler_method<<<grid, block>>>((Agent *) agentsDP, na, dt, max_speed, lx, ly, lz);
    cudaDeviceSynchronize();
    start2 = std::chrono::steady_clock::now();
    cudaMemcpy(agents, agentsDP, 3*arrayMemSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    end2 = std::chrono::steady_clock::now();
    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - start3).count()
                        + std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count());
    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(start2 - end3).count());
    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start).count());
    return times;
}

std::vector<std::string> EulerParallelWorkspace::getTimeDescriptions() {
    std::vector<std::string> timeDescriptions;
    timeDescriptions.push_back("EulerParallelWorkspace.forceComputationDuration");
    timeDescriptions.push_back("EulerParallelWorkspace.memoryAllocationAndTransferDuration");
    timeDescriptions.push_back("EulerParallelWorkspace.eulerIntegrationDuration");
    timeDescriptions.push_back("EulerParallelWorkspace.totalTime");
    return timeDescriptions;
}
