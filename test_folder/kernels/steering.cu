FLAMEGPU_AGENT_FUNCTION(steering, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const float3 my_pos = { FLAMEGPU->getVariable<float>("x"), 
                            FLAMEGPU->getVariable<float>("y"), 
                            FLAMEGPU->getVariable<float>("z") };

    const float perception_radius = FLAMEGPU->environment.getProperty<float>("perception_radius");
    const float sep_w = FLAMEGPU->environment.getProperty<float>("separation_weight");
    const float ali_w = FLAMEGPU->environment.getProperty<float>("alignment_weight");
    const float coh_w = FLAMEGPU->environment.getProperty<float>("cohesion_weight");

    float3 sep_steer = {0,0,0};
    float3 ali_vel = {0,0,0};
    float3 coh_pos = {0,0,0};
    int count = 0;

    for (const auto &msg : FLAMEGPU->message_in(my_pos.x, my_pos.y, my_pos.z)) {
        // Corrected coordinate access for FLAME GPU 2
        float3 other_pos = { msg.getVariable<float>("x"), 
                             msg.getVariable<float>("y"), 
                             msg.getVariable<float>("z") };
        
        float dx = my_pos.x - other_pos.x;
        float dy = my_pos.y - other_pos.y;
        float dz = my_pos.z - other_pos.z;
        float dist_sq = dx*dx + dy*dy + dz*dz;

        if (dist_sq > 0 && dist_sq < (perception_radius * perception_radius)) {
            float dist = sqrtf(dist_sq);
            sep_steer.x += dx / dist;
            sep_steer.y += dy / dist;
            sep_steer.z += dz / dist;
            
            ali_vel.x += msg.getVariable<float>("vx");
            ali_vel.y += msg.getVariable<float>("vy");
            ali_vel.z += msg.getVariable<float>("vz");

            coh_pos.x += other_pos.x;
            coh_pos.y += other_pos.y;
            coh_pos.z += other_pos.z;
            count++;
        }
    }

    float3 final_f = {0,0,0};
    if (count > 0) {
        // Average Alignment Velocity
        ali_vel.x /= count; ali_vel.y /= count; ali_vel.z /= count;
        // Average Cohesion Position (Vector from me to center of mass)
        coh_pos.x = (coh_pos.x / count) - my_pos.x;
        coh_pos.y = (coh_pos.y / count) - my_pos.y;
        coh_pos.z = (coh_pos.z / count) - my_pos.z;

        final_f.x = (sep_steer.x * sep_w) + (ali_vel.x * ali_w) + (coh_pos.x * coh_w);
        final_f.y = (sep_steer.y * sep_w) + (ali_vel.y * ali_w) + (coh_pos.y * coh_w);
        final_f.z = (sep_steer.z * sep_w) + (ali_vel.z * ali_w) + (coh_pos.z * coh_w);
    }

    FLAMEGPU->setVariable<float>("fx", final_f.x);
    FLAMEGPU->setVariable<float>("fy", final_f.y);
    FLAMEGPU->setVariable<float>("fz", final_f.z);
    
    return flamegpu::ALIVE;