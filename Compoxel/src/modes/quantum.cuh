#pragma once
#include "flamegpu/flamegpu.h"
#include "../kernels/quantum/rejection_sample.cu" // Include the device kernel

// A function to generate and return the configured Phase A model
flamegpu::ModelDescription buildQuantumMode() {
    flamegpu::ModelDescription model("Quantum_Rejection_Sampler");

    // 1. Define Nucleus (Environment Property for Phase A)
    model.Environment().add<float>("nucleus_x", 0.0f);
    model.Environment().add<float>("nucleus_y", 0.0f);
    model.Environment().add<float>("nucleus_z", 0.0f);
    model.Environment().add<float>("nucleus_charge", 1.0f);

    // 2. Define Electron Agent
    flamegpu::AgentDescription electron = model.newAgent("electron");
    electron.newVariable<float>("x");
    electron.newVariable<float>("y");
    electron.newVariable<float>("z");
    electron.newVariable<float>("alpha", 1.0f); // 1 = visible, 0 = rejected

    // 3. Attach the Kernel
    electron.newFunction("sample", sample_wavefunction); 

    // 4. Set Execution Order
    model.newLayer().addAgentFunction(sample_wavefunction);

    return model;
}