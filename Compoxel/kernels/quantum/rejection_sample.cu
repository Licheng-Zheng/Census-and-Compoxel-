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
               

case 3: // 2s Orbital
{
float radial_2s = (2.0f - r);
p = (radial_2s * radial_2s * expf(-r)) / 0.191f;
}
break;
}

case 6: // 3s Orbital
{
float poly = 6.0f - 6.0f * r + (r * r);
p = (poly * poly * expf(-2.0f * r / 3.0f)) / 3.73f;
}
break;
case 7: // 3p_z Orbital
p = (z * z * (4.0f - r) * (4.0f - r) * expf(-2.0f * r / 3.0f)) / 1.51f;
break;

case 8: // 3p_x Orbital
p = (x * x * (4.0f - r) * (4.0f - r) * expf(-2.0f * r / 3.0f)) / 1.51f;
break;

case 9: // 3p_y Orbital
p = (y * y * (4.0f - r) * (4.0f - r) * expf(-2.0f * r / 3.0f)) / 1.51f;
break;
case 10: // 3d_xy
p = (x * x * y * y * expf(-2.0f * r / 3.0f)) / 6.58f;
break;

case 11: // 3d_yz
p = (y * y * z * z * expf(-2.0f * r / 3.0f)) / 6.58f;
break;

case 12: // 3d_xz
p = (x * x * z * z * expf(-2.0f * r / 3.0f)) / 6.58f;
break;

case 13: // 3d_x2-y2
{
float diff = (x * x - y * y);
p = (diff * diff * expf(-2.0f * r / 3.0f)) / 26.33f;
}
break;

float dice_roll = FLAMEGPU->random.uniform<float>();

if (dice_roll < p) {
FLAMEGPU->setVariable<float>("alpha", 1.0f);
} else {
FLAMEGPU->setVariable<float>("alpha", 0.0f);
}
    
    return flamegpu::ALIVE;
}