#include "flamegpu/flamegpu.h"

FLAMEGPU_AGENT_FUNCTION(test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    return flamegpu::ALIVE;
}

int main(int argc, const char ** argv) {
    flamegpu::ModelDescription model("TestModel");
    flamegpu::AgentDescription agent = model.newAgent("electron");
    agent.newVariable<float>("x");
    agent.newFunction("test", test_func);

    flamegpu::CUDASimulation sim(model);
    sim.initialise(argc, argv);
    sim.step();
    return 0;
}