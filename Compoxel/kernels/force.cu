#ifndef FLAMEGPU_AGENT_FUNCTION
#define FLAMEGPU_AGENT_FUNCTION(name, in, out) void name()
#endif

FLAMEGPU_AGENT_FUNCTION(force_kernel, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    return flamegpu::ALIVE;
}