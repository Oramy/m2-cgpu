/*
*/
#ifndef  EULER_PARALLEL_WORKSPACE
#define  EULER_PARALLEL_WORKSPACE

#include "parser.hxx"
#include "types.hxx"
#include "workspace.hxx"
#include <vector>


class EulerParallelWorkspace: public Workspace
{
public:
  virtual std::vector<time_t> move();
  EulerParallelWorkspace(ArgumentParser &parser);
  virtual std::vector<std::string> getTimeDescriptions();
  ~EulerParallelWorkspace();
protected:
	void* agentsDP;
	Agent* agentsH;
};

#endif
