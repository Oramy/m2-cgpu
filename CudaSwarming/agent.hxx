#ifndef AGENT_HXX
#define AGENT_HXX

#include "types.hxx"
#include "vector.hxx"

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

    Agent(const Vector &pos, const Vector &vel, const Vector &dir);

    void compute_force(Container &agent_list, size_t index, double dist);

    size_t find_closest(Container &agent_list, size_t index);
};

#endif
