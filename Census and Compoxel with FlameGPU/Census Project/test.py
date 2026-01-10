import pyflamegpu
import random
import os

# 1. Environment Setup
os.environ['CUDA_PATH'] = '/usr/local/cuda'

# 2. Model Definition
model = pyflamegpu.ModelDescription("Boids_Model")

# 3. Agent Definition
bird_agent = model.newAgent("bird_agent")
bird_agent.newVariableFloat("x")
bird_agent.newVariableFloat("y")
bird_agent.newVariableFloat("z")
bird_agent.newVariableFloat("vx")
bird_agent.newVariableFloat("vy")
bird_agent.newVariableFloat("vz")
bird_agent.newVariableFloat("fx")
bird_agent.newVariableFloat("fy")
bird_agent.newVariableFloat("fz")

# 4. Environment Properties
env = model.Environment()
env.newPropertyFloat("separation_weight", 1.5)
env.newPropertyFloat("alignment_weight", 1.0)
env.newPropertyFloat("cohesion_weight", 1.0)
env.newPropertyFloat("perception_radius", 10.0)
env.newPropertyFloat("max_speed", 2.0)
env.newPropertyArrayFloat("bounds_min", [-100.0, -100.0, -100.0])
env.newPropertyArrayFloat("bounds_max", [100.0, 100.0, 100.0])

# 5. Message Definition (Spatial 3D)
message = model.newMessageSpatial3D("LocationMessage")
message.newVariableFloat("vx")
message.newVariableFloat("vy")
message.newVariableFloat("vz")
message.setMin(-100.0, -100.0, -100.0)
message.setMax(100.0, 100.0, 100.0)
message.setRadius(10.0)

# 6. Agent Functions (CUDA C++)
output_code = r'''
FLAMEGPU_AGENT_FUNCTION(output_message, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
    FLAMEGPU->message_out.setVariable<float>("vx", FLAMEGPU->getVariable<float>("vx"));
    FLAMEGPU->message_out.setVariable<float>("vy", FLAMEGPU->getVariable<float>("vy"));
    FLAMEGPU->message_out.setVariable<float>("vz", FLAMEGPU->getVariable<float>("vz"));
    return flamegpu::ALIVE;
}
'''

steering_code = r'''
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
}
'''

integration_code = r'''
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
'''

# 7. Registration and Layers
out_func = bird_agent.newRTCFunction("output_message", output_code)
out_func.setMessageOutput("LocationMessage")

steer_func = bird_agent.newRTCFunction("steering", steering_code)
steer_func.setMessageInput("LocationMessage")

move_func = bird_agent.newRTCFunction("integration", integration_code)

model.newLayer().addAgentFunction(out_func)
model.newLayer().addAgentFunction(steer_func)
model.newLayer().addAgentFunction(move_func)

# 8. Initialization
sim = pyflamegpu.CUDASimulation(model)
pop = pyflamegpu.AgentVector(bird_agent, 1000)

for i in range(1000):
    instance = pop[i]
    instance.setVariableFloat("x", random.uniform(-100, 100))
    instance.setVariableFloat("y", random.uniform(-100, 100))
    instance.setVariableFloat("z", random.uniform(-100, 100))
    instance.setVariableFloat("vx", random.uniform(-1, 1))
    instance.setVariableFloat("vy", random.uniform(-1, 1))
    instance.setVariableFloat("vz", random.uniform(-1, 1))

sim.setPopulationData(pop)

# 9. Run
sim.step()
print("Simulation step successful.")