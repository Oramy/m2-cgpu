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

#define BLOCK_SIZE 2

__global__ void euler_method(Real * direction, Real * velocity, Real * position,
        Real * norm, Real na, Real dt, Real max_speed,
                        Real lx, Real ly, Real lz) {

    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    // We get the length of the space in dimension k%3.
    int l = k % 3 == 0 ? lx : (k % 3 == 1 ? ly : lz);

    // We check that our index is in the right range.
    if (k < 3 * na) {
      velocity[k] += dt * direction[k];

      double speed = norm[k/3]/max_speed;
      if (speed > 1. && speed>0.) {
        velocity[k] /= speed;
      }
      position[k] += dt * velocity[k];

      if(position[k] < 40)
        position[k] = l - 40;
      if(position[k] > l - 40)
        position[k] = 40;
    }
}
EulerParallelWorkspace::EulerParallelWorkspace(ArgumentParser &parser): Workspace(parser){
    savepath = "boidsParallel.xyz";
}

std::vector<time_t> EulerParallelWorkspace::move()
{
    std::vector<time_t> times;

    std::chrono::steady_clock::time_point start, end, start2, end2;

    start = std::chrono::steady_clock::now();
    // Compute forces applied on specific agent
    for(size_t k = 0; k< na; k++){
      agents[k].compute_force(agents, k, rCohesion);

      agents[k].direction = agents[k].cohesion*wCohesion
        + agents[k].alignment*wAlignment
        + agents[k].separation*wSeparation;
    }
    end = std::chrono::steady_clock::now();
    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    start = std::chrono::steady_clock::now();
    Real* directionD, *positionD, *velocityD, *normD;
    Real* directionH = new Real[na * 3];
    Real* positionH = new Real[na * 3];
    Real* velocityH = new Real[na * 3];
    Real* normH = new Real[na];
    

    for (size_t k = 0; k < na; k++) {
        directionH[3*k] = agents[k].direction.x;
        directionH[3*k+1] = agents[k].direction.y;
        directionH[3*k+2] = agents[k].direction.z;
        positionH[3*k] = agents[k].position.x;
        positionH[3*k+1] = agents[k].position.y;
        positionH[3*k+2] = agents[k].position.z;
        velocityH[3*k] = agents[k].velocity.x;
        velocityH[3*k+1] = agents[k].velocity.y;
        velocityH[3*k+2] = agents[k].velocity.z;
        normH[k] = agents[k].velocity.norm();
    }

    size_t arrayMemSize = sizeof(Real) * na;

    cudaMalloc(&directionD, 3*arrayMemSize);
    cudaMalloc(&positionD, 3*arrayMemSize);
    cudaMalloc(&velocityD, 3*arrayMemSize);
    cudaMalloc(&normD, 3*arrayMemSize);

    cudaMemcpy(directionD, directionH, 3*arrayMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(positionD, positionH, 3*arrayMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(velocityD, velocityH, 3*arrayMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(normD, normH, arrayMemSize, cudaMemcpyHostToDevice);

    end = std::chrono::steady_clock::now();
    dim3 block(BLOCK_SIZE);
    dim3 grid(int(ceil((float(na)/BLOCK_SIZE))));
    euler_method<<<grid, block>>>(directionD, velocityD, positionD, normD, na, dt, max_speed, lx, ly, lz);
    cudaDeviceSynchronize();
    start2 = std::chrono::steady_clock::now();
    cudaMemcpy(positionH, positionD, 3*arrayMemSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(velocityH, velocityD, 3*arrayMemSize, cudaMemcpyDeviceToHost);
    for (size_t k = 0; k < na; k++) {
        agents[k].position.x=  positionH[3 * k];
        agents[k].position.y= positionH[3 * k + 1];
        agents[k].position.z= positionH[3 * k + 2];
        agents[k].velocity.x=  velocityH[3 * k];
        agents[k].velocity.y= velocityH[3 * k + 1];
        agents[k].velocity.z= velocityH[3 * k + 2];
    }
    end2 = std::chrono::steady_clock::now();
    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start + end2 - start2).count());
    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count());
    return times;
}

std::vector<std::string> EulerParallelWorkspace::getTimeDescriptions() {
    std::vector<std::string> timeDescriptions;
    timeDescriptions.push_back("EulerParallelWorkspace.forceComputingDuration");
    timeDescriptions.push_back("EulerParallelWorkspace.memoryAllocationAndTransferDuration");
    timeDescriptions.push_back("EulerParallelWorkspace.eulerIntegrationDuration");
    return timeDescriptions;
}
