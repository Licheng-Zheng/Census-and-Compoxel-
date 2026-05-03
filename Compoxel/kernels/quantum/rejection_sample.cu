#include "/usr/local/lib/python3.12/site-packages/nvidia/curand/include/curand_kernel.h"

FLAMEGPU_AGENT_FUNCTION(rejection_sampling, flamegpu::MessageNone, flamegpu::MessageNone) {
    float x = FLAMEGPU->getVariable<float>("x");
    float y = FLAMEGPU->getVariable<float>("y");
    float z = FLAMEGPU->getVariable<float>("z");
    int orbital_id = FLAMEGPU->getVariable<int>("orbital_id");
    
    float r = sqrt(x*x + y*y + z*z);
    float p = 0.0f;
    
    switch(orbital_id) {
        case 0: 
            // 1s Orbital
            // Max is at r=0. Peak = 1.0
            p = expf(-2.0f * r); 
            break;
            
        case 1: 
            // 2p_z Orbital 
            // Proportional to z^2 * e^{-r}
            // Max occurs at z=2, r=2. Peak = 4 * e^-2 ≈ 0.5413
            p = (z * z * expf(-r)) / 0.5413f; 
            break;
            
        case 2:
            // 3d_z^2 Orbital
            // Proportional to (3z^2 - r^2)^2 * e^{-2r/3}
            // Max occurs at z=3, r=3. Peak ≈ 43.85
            float z_sq = z * z;
            float r_sq = r * r;
            float angular_part = (3.0f * z_sq - r_sq);
            p = (angular_part * angular_part * expf(-2.0f * r / 3.0f)) / 43.85f;
            break;
    }
    
    float dice_roll = FLAMEGPU->random.uniform<float>();
    
    if (dice_roll < p) {
        FLAMEGPU->setVariable<float>("alpha", 1.0f);
    } else {
        FLAMEGPU->setVariable<float>("alpha", 0.0f);
    }
    
    return flamegpu::ALIVE;
}