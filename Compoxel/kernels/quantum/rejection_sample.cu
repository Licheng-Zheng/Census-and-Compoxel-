FLAMEGPU_AGENT_FUNCTION(rejection_sampling, flamegpu::MessageNone, flamegpu::MessageNone) {
    // 1. Read position
    float x = FLAMEGPU->getVariable<float>("x");
    float y = FLAMEGPU->getVariable<float>("y");
    float z = FLAMEGPU->getVariable<float>("z");
    
    // 2. Math (Hydrogen 1s density: rho = exp(-r))
    float r = sqrt(x*x + y*y + z*z);
    float p = expf(-r);
    
    // 3. Rejection logic
    float dice_roll = FLAMEGPU->random.uniform<float>();
    
    if (dice_roll < p) {
        FLAMEGPU->setVariable<float>("alpha", 1.0f); // Keep
    } else {
        FLAMEGPU->setVariable<float>("alpha", 0.0f); // Reject
    }
    
    return flamegpu::ALIVE;
}