#include "agent.hxx"

CUDA_CALLABLE_MEMBER Agent::Agent(const Vector &pos, const Vector &vel, const Vector &dir){
  position = pos;
  velocity = vel;
  direction = dir;
  max_speed = 80.0;
  max_force = 20.0;
}
CUDA_CALLABLE_MEMBER Agent::Agent() {}
CUDA_CALLABLE_MEMBER Agent& Agent::operator=(const Agent& agent) {
    position = agent.position;
    velocity = agent.velocity;
    direction = agent.direction;
    max_speed = agent.max_speed;
    max_force = agent.max_force;
    cohesion = agent.cohesion;
    alignment = agent.alignment;
    separation = agent.separation;
    return *this;
}


CUDA_CALLABLE_MEMBER void Agent::compute_force(Agent * agent_list, size_t na, size_t index, double rad) {
  cohesion.x = 0;
  cohesion.y = 0;
  cohesion.z = 0;
  alignment.x = alignment.y = alignment.z = 0;
  separation.x = separation.y = separation.z = 0;

  int count_c = 0 , count_s = 0 , count_a = 0 ;
  for(size_t i = 0; i < na; i++) {
    Real dist = (this->position - agent_list[i].position).norm();
    if (i != index && dist < rs && dist > 0.){
      separation += (this->position - agent_list[i].position).normalized()/dist;
      ++count_s;
    }
    if (i != index && dist < ra){
      alignment += agent_list[i].velocity;
      ++count_a;
    }
    if (i != index && dist < rc){
      cohesion += agent_list[i].position;
      ++count_c;
    }
  }

  // Compute separation contribution
  if (count_s > 0){
    separation.normalize();
    separation *= max_force;
    separation -= velocity;
    if (separation.norm() > max_force){
      separation.normalize();
      separation *= max_force;
    }
  }
  // Compute alignment contribution
  if (count_a > 0){
    if (alignment.norm() > 0.0){
      alignment.normalize();
      alignment *= max_force;
    }
    alignment -= velocity;
    if (alignment.norm() > max_force){
      alignment.normalize();
      alignment *= max_force;
    }
  }
  // Compute cohesion contribution
  if (count_c > 0){
    // Compute center of gravity
    cohesion /= count_c;

    // Direction of displacement
    cohesion -= position;

    cohesion.normalize();
    cohesion *= max_speed;
    cohesion -= velocity;
    if (cohesion.norm() > max_force){
      cohesion.normalize();
      cohesion *= max_force;
    }
  }
}

CUDA_DEVICE size_t Agent::find_closest(Agent* agents, size_t na, size_t index) {
  size_t closest_agent = index;
  double min_dist = 1000;

  double dist;

  for (size_t i = 0; i < na; i++) {
    if (i != index) {
      dist= (this->position - agents[i].position).norm();

      if(dist < min_dist) {
        min_dist = dist;
        closest_agent = i;
      }
    }
  }
  return closest_agent;
}
