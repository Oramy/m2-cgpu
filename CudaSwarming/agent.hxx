#ifndef AGENT_HXX
#define AGENT_HXX

#include "types.hxx"
#include "vector.hxx"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_HOST
#define CUDA_DEVICE
#endif 

typedef enum {
  prey,
  predator,
  active,
  wall
} AgentType;

class Agent{
  public :
    Vector position;
    Vector velocity;
    Vector direction;

    Vector cohesion;
    Vector separation;
    Vector alignment;

    double max_speed;
    double max_force;

    // Distance of influence
    double rc, rs, ra;

    CUDA_CALLABLE_MEMBER Agent(const Vector &pos, const Vector &vel, const Vector &dir);
    CUDA_CALLABLE_MEMBER Agent();

    CUDA_CALLABLE_MEMBER void compute_force(Agent* agents, size_t na, size_t index, double dist);
    CUDA_CALLABLE_MEMBER size_t find_closest(Agent* agents, size_t na, size_t index);
    CUDA_CALLABLE_MEMBER Agent& operator=(const Agent& agent);
};

#endif
