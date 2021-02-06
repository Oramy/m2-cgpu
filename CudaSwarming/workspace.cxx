#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>

#include "agent.hxx"
#include "vector.hxx"
#include "workspace.hxx"

Workspace::Workspace(ArgumentParser &parser)
{

    savepath = "boids.xyz";
  na = parser("agents").asInt();

  wCohesion = parser("wc").asDouble();
  wAlignment = parser("wa").asDouble();
  wSeparation = parser("ws").asDouble();

  rCohesion = parser("rc").asDouble();
  rAlignment = parser("ra").asDouble();
  rSeparation = parser("rs").asDouble();
  dt= 0.01;
  max_speed = 20.0;
  max_force = 80.0;
  time = 0.,

  this->init();}

Workspace::Workspace(size_t nAgents,
             Real wc, Real wa, Real ws,
             Real rc, Real ra, Real rs) :
             na(nAgents), dt(.05), time(0),
             wCohesion(wc), wAlignment(wa), wSeparation(ws),
             rCohesion(rc), rAlignment(ra), rSeparation(rs),
             max_speed(20.), max_force(80.)
{ this->init(); }

void Workspace::init(){
  lx = 800.0;
  ly = 800.0;
  lz = 800.0;

  padding = 0.02 * lx;
  // Random generator seed
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  std::default_random_engine rng;
  rng.seed(0);

  // Initialize agents
  agents = new Agent[na];
  // This loop may be quite expensive due to random number generation
  for(size_t j = 0; j < na; j++){
    // Create random position
    //Vector position(lx*(0.02 + drand48()), ly*(0.02 + drand48()), lz*(0.02 + drand48()));
    Vector position(lx*(0.02 + unif(rng)), ly*(0.02 + unif(rng)), lz*(0.02 + unif(rng)));
    Vector velocity(160 * (unif(rng) - 0.5), 160*(unif(rng) - 0.5), 160*(unif(rng) - 0.5));

    // Create random velocity
    agents[j] =  Agent(position, velocity, Zeros());
    agents[j].max_force = max_force;
    agents[j].max_speed = max_speed;
    agents[j].ra = rAlignment;
    agents[j].rc = rCohesion;
    agents[j].rs = rSeparation;
  }
}

std::vector<time_t> Workspace::move()
{
    std::vector<time_t> times;
    std::chrono::steady_clock::time_point start, end, start2, end2;

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

    start2 = std::chrono::steady_clock::now();
    // Time integration using euler method
    for(size_t k = 0; k< na; k++){
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
    end2 = std::chrono::steady_clock::now();
    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count());
    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start).count());
    return times;
}

std::vector<time_t> Workspace::simulate(int nsteps) {
  // store initial positions
    save(0);

    // perform nsteps time steps of the simulation
    int step = 0;
    long long totalTime = 0;

    std::vector<time_t> sum;
    while (step++ < nsteps) {
      std::vector<time_t> times = this->move();
      for (int i = 0; i < times.size(); i++) {
          if (i >= sum.size())
              sum.push_back(times[i]);
          else
              sum[i] += times[i];
      }
      // store every 20 steps
      if (step%20 == 0) save(step);
    }


    return sum;
}

void Workspace::save(int stepid) {
  std::ofstream myfile;

  myfile.open(savepath, stepid==0 ? std::ios::out : std::ios::app);

    myfile << std::endl;
    myfile << na << std::endl;
    for (size_t p=0; p<na; p++)
        myfile << "B " << agents[p].position;

    myfile.close();
}
std::vector<std::string> Workspace::getTimeDescriptions() {
    std::vector<std::string> timeDescriptions;
    timeDescriptions.push_back("Workspace.forceComputingDuration");
    timeDescriptions.push_back("Workspace.eulerIntegrationDuration");
    timeDescriptions.push_back("Workspace.totalTime");
    return timeDescriptions;
}

void Workspace::simulateAndPrintTimeRecords(int nSteps) {
  std::vector<time_t> elapsed_times = simulate(nSteps);
  std::vector<std::string> time_descs(getTimeDescriptions());

  for (int i = 0; i < elapsed_times.size(); i++) {
	  std::cout << time_descs[i] << " " << long double (elapsed_times[i]) / 1'000'000'000.0 << std::endl;
  }
}
