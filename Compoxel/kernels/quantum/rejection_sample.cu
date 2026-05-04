FLAMEGPU_AGENT_FUNCTION(rejection_sampling, flamegpu::MessageNone, flamegpu::MessageNone) {
    // 1. Get Agent Variables
    float x = FLAMEGPU->getVariable<float>("x");
    float y = FLAMEGPU->getVariable<float>("y");
    float z = FLAMEGPU->getVariable<float>("z");
    int orbital_id = FLAMEGPU->getVariable<int>("orbital_id");
    
    // 2. Pre-calculate common terms
    float r_sq = x*x + y*y + z*z;
    float r = sqrtf(r_sq);
    float p = 0.0f;
    
    // 3. Orbital Probability Density Logic (normalized to peak = 1.0)
    switch(orbital_id) {
        // --- n=1 ---
        case 0: // 1s
            p = expf(-2.0f * r); 
            break;

        // --- n=2 ---
        case 1: // 2s
            {
                float poly = (2.0f - r);
                p = (poly * poly * expf(-r)) / 0.191f;
            }
            break;
        case 2: // 2pz
            p = (z * z * expf(-r)) / 0.5413f;
            break;
        case 3: // 2px
            p = (x * x * expf(-r)) / 0.5413f;
            break;
        case 4: // 2py
            p = (y * y * expf(-r)) / 0.5413f;
            break;

        // --- n=3 ---
        case 5: // 3s
            {
                float poly = 6.0f - 6.0f * r + r_sq;
                p = (poly * poly * expf(-2.0f * r / 3.0f)) / 3.73f;
            }
            break;
        case 6: // 3pz
            {
                float poly = (4.0f - r);
                p = (z * z * poly * poly * expf(-2.0f * r / 3.0f)) / 1.51f;
            }
            break;
        case 7: // 3px
            {
                float poly = (4.0f - r);
                p = (x * x * poly * poly * expf(-2.0f * r / 3.0f)) / 1.51f;
            }
            break;
        case 8: // 3py
            {
                float poly = (4.0f - r);
                p = (y * y * poly * poly * expf(-2.0f * r / 3.0f)) / 1.51f;
            }
            break;
        case 9: // 3dz2
            {
                float angular = (3.0f * z * z - r_sq);
                p = (angular * angular * expf(-2.0f * r / 3.0f)) / 43.85f;
            }
            break;
        case 10: // 3dxy
            p = (x * x * y * y * expf(-2.0f * r / 3.0f)) / 6.58f;
            break;
        case 11: // 3dyz
            p = (y * y * z * z * expf(-2.0f * r / 3.0f)) / 6.58f;
            break;
        case 12: // 3dxz
            p = (x * x * z * z * expf(-2.0f * r / 3.0f)) / 6.58f;
            break;
        case 13: // 3dx2-y2
            {
                float diff = (x * x - y * y);
                p = (diff * diff * expf(-2.0f * r / 3.0f)) / 26.33f;
            }
            break;

        default:
            p = 0.0f;
            break;
    }
    
    // 4. Rejection Sampling Decision
    float dice_roll = FLAMEGPU->random.uniform<float>();
    
    if (dice_roll < p) {
        FLAMEGPU->setVariable<float>("alpha", 1.0f);
    } else {
        // We set alpha to 0 for visualization or potential removal
        FLAMEGPU->setVariable<float>("alpha", 0.0f);
    }
    
    return flamegpu::ALIVE;
}