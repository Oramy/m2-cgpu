/*
*/
#ifndef  WORKSPACE
#define  WORKSPACE

#include "parser.hxx"
#include "types.hxx"
#include <vector>


class Workspace
{
protected:
  Agent* agents;
  unsigned int na;

  Real dt;
  int time;

  Real wCohesion, wAlignment, wSeparation;
  Real rCohesion, rAlignment, rSeparation;
  Real maxU;

  Real max_speed;
  Real max_force;

  Real tUpload, tDownload, tCohesion, tAlignment, tSeparation;

  // Size of the domain
  Real lx, ly, lz;

  // Lower bound of the domain
  Real xmin, ymin, zmin;

  // Padding around the domain
  Real padding;


  Real domainsize;

  std::string savepath;
  void init();

public:
  Workspace(ArgumentParser &parser);

  Workspace(size_t nAgents,
  Real wc, Real wa, Real ws,
  Real rc, Real ra, Real rs);

  virtual std::vector<std::string> getTimeDescriptions();
  virtual std::vector<time_t> move();
  virtual std::vector<time_t> simulate(int nsteps);
  void simulateAndPrintTimeRecords(int nSteps);
  void save(int stepid);
};

#endif
