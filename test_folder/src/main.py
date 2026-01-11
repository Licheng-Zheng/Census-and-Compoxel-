import pyflamegpu
import random
import os

import matplotlib as mpl

mpl.rcParams['animation.embed_limit'] = 100000000 

# 1. Environment Setup
os.environ['CUDA_PATH'] = '/usr/local/cuda'

def load_cuda_code(filepath):
    with open(filepath, 'r') as f:
        return f.read()

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
message.setMin(-20.0, -20.0, -20.0)
message.setMax(20.0, 20.0, 20.0)
message.setRadius(10.0) 

# 6. Agent Functions (CUDA C++)

# 7. Registration and Layers
output_code = load_cuda_code("../kernels/output.cu")        # note output.cu filename (not output_message.cu)
steering_code = load_cuda_code("../kernels/steering.cu")
integration_code = load_cuda_code("../kernels/integration.cu")

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
pop = pyflamegpu.AgentVector(bird_agent, 7000)

for i in range(7000):
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

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# --- [Insert the previous Model, Agent, and Sim setup code here] ---

# Prepare the Matplotlib Figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Initial population retrieval to setup plot limits
final_pop = pyflamegpu.AgentVector(bird_agent)
sim.getPopulationData(final_pop)

# Setup the scatter plot
# We extract x, y, z coordinates from the AgentVector
x_data = [a.getVariableFloat("x") for a in final_pop]
y_data = [a.getVariableFloat("y") for a in final_pop]
z_data = [a.getVariableFloat("z") for a in final_pop]
scatter = ax.scatter(x_data, y_data, z_data, s=5, c='blue', alpha=0.6)

# Set plot boundaries based on your environment bounds
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_zlim(-100, 100)
ax.set_title("me sim")

# Updated update function for your visualization block
def update(frame):
    # Progress the simulation 3 steps per animation frame
    for _ in range(3):
        sim.step()
        
    # Get current step count from simulation if needed
    # current_step = sim.getStepCounter() 
    
    sim.getPopulationData(final_pop)
    
    # Calculate and print stats every few steps to the console
    if frame % 10 == 0:
        avg_x = sum([a.getVariableFloat("x") for a in final_pop]) / 1000
        print(f"Frame {frame}: Avg Position X = {avg_x:.2f}")

    new_x = [a.getVariableFloat("x") for a in final_pop]
    new_y = [a.getVariableFloat("y") for a in final_pop]
    new_z = [a.getVariableFloat("z") for a in final_pop]
    
    scatter._offsets3d = (new_x, new_y, new_z)
    return scatter,

from IPython.display import HTML

# 1. Assign the animation to a variable (prevents the 'deleted without rendering' warning)
anim = FuncAnimation(fig, update, frames=300, interval=50, blit=False)

# 2. Use the HTML display tool to render the animation as a video
# Note: This might take a minute to process as it runs the simulation frames
plt.close() # Prevents a duplicate static plot from appearing
HTML(anim.to_html5_video())

# ... existing code ...

from IPython.display import HTML, display
import shutil

# Save animation to a file
output_filename = 'boids_simulation.mp4'
anim.save(output_filename, writer='ffmpeg', dpi=200)

# Provide a download link in the notebook
def create_download_link(filename, link_text):
    from IPython.display import FileLink
    return FileLink(filename, result_html_prefix=f'<a href="{filename}" download>{link_text}</a>')

display(create_download_link(output_filename, "Download Simulation Video"))