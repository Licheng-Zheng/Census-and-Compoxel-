import pyflamegpu
from src.model import create_model
from test import timmy 

timmy(1) 

# 1. Create Model
model = create_model()

# 2. Create Simulation
sim = pyflamegpu.CUDASimulation(model)

# 3. Run
print("Starting simulation...")
sim.step()
print("Simulation finished successfully!")