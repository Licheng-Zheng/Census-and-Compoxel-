FLAMEGPU_AGENT_FUNCTION(integration, flamegpu::MessageNone, flamegpu::MessageNone) {
    float vx = FLAMEGPU->getVariable<float>("vx") + FLAMEGPU->getVariable<float>("fx");
    float vy = FLAMEGPU->getVariable<float>("vy") + FLAMEGPU->getVariable<float>("fy");
    float vz = FLAMEGPU->getVariable<float>("vz") + FLAMEGPU->getVariable<float>("fz");

    const float max_speed = FLAMEGPU->environment.getProperty<float>("max_speed");
    float speed = sqrtf(vx*vx + vy*vy + vz*vz);
    if (speed > max_speed) {
        vx = (vx / speed) * max_speed;
        vy = (vy / speed) * max_speed;
        vz = (vz / speed) * max_speed;
    }

    float nx = FLAMEGPU->getVariable<float>("x") + vx;
    float ny = FLAMEGPU->getVariable<float>("y") + vy;
    float nz = FLAMEGPU->getVariable<float>("z") + vz;

    // Correct Environment Access for Arrays
    const float min_x = FLAMEGPU->environment.getProperty<float>("bounds_min", 0);
    const float max_x = FLAMEGPU->environment.getProperty<float>("bounds_max", 0);
    const float min_y = FLAMEGPU->environment.getProperty<float>("bounds_min", 1);
    const float max_y = FLAMEGPU->environment.getProperty<float>("bounds_max", 1);
    const float min_z = FLAMEGPU->environment.getProperty<float>("bounds_min", 2);
    const float max_z = FLAMEGPU->environment.getProperty<float>("bounds_max", 2);
    
    if (nx < min_x) nx = max_x; if (nx > max_x) nx = min_x;
    if (ny < min_y) ny = max_y; if (ny > max_y) ny = min_y;
    if (nz < min_z) nz = max_z; if (nz > max_z) nz = min_z;

    FLAMEGPU->setVariable<float>("x", nx);
    FLAMEGPU->setVariable<float>("y", ny);
    FLAMEGPU->setVariable<float>("z", nz);
    FLAMEGPU->setVariable<float>("vx", vx);
    FLAMEGPU->setVariable<float>("vy", vy);
    FLAMEGPU->setVariable<float>("vz", vz);
    return flamegpu::ALIVE;
}