import pyflamegpu
from src.utils import load_kernel

def create_model():
    model = pyflamegpu.ModelDescription("MD_Sim")
    atom = model.newAgent("Atom")
    
    # REQUIRED for Spatial3D
    atom.newVariableFloat("x")
    atom.newVariableFloat("y")
    atom.newVariableFloat("z")
    
    # Load our C++ kernel
    force_code = load_kernel("force.cu")
    force_func = atom.newRTCFunction("force_kernel", force_code)
    
    # Just a placeholder layer for now
    model.newLayer().addAgentFunction(force_func)
    
    return model