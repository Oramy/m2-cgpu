
// Agent (particle model)
#include "agent.hxx"

#include "types.hxx"
#include "parser.hxx"

#include "workspace.hxx"
#include "eulerParallelWorkspace.hxx"
#include <vector>
#include <iomanip>

// Main class for running the parallel flocking sim
int main(int argc, char **argv) {
  // Create parser
  ArgumentParser parser;

  // Add options to parser
  parser.addOption("agents", 100);
  parser.addOption("steps", 5000);
  parser.addOption("wc", 1.0);//7.0);
  parser.addOption("wa", 1.0);// 12.0);
  parser.addOption("ws", 1.5);// 55.0);

  parser.addOption("rc", 90);
  parser.addOption("ra", 90);
  parser.addOption("rs", 25);

  // Parse command line arguments
  parser.setOptions(argc, argv);

  // Create workspace
  EulerParallelWorkspace workspace(parser);
  Workspace workspaceSeq(parser);
  // Launch simulation
  int nSteps = parser("steps").asInt();
  std::cout << std::fixed  << std::setprecision(2)
      << std::setfill('0');
  workspace.simulateAndPrintTimeRecords(nSteps);
  workspaceSeq.simulateAndPrintTimeRecords(nSteps);
  return 0;
}
