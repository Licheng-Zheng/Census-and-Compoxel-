def create_model():
    model = pyflamegpu.ModelDescription("Compoxel_Sim")
    atom = model.newAgent("Atom")
    
    # 1. Setup Variables
    atom.newVariableFloat("x")
    atom.newVariableFloat("y")
    atom.newVariableFloat("z")
    
    # 2. REQUIRED: Create the Message
    # If your C++ expects MessageSpatial3D, you MUST define it here
    message = model.newMessageSpatial3D("location_message")
    message.setMin(-100, -100, -100) # Must match your domain size
    message.setMax(100, 100, 100)
    message.setRadius(5.0)           # Interaction radius
    
    # 3. Setup the Function
    force_code = load_kernel("force.cu")
    force_func = atom.newRTCFunction("force_kernel", force_code)
    
    # 4. CRITICAL FIX: Connect the Message to the Function
    force_func.setMessageInput("location_message") 
    
    # 5. Add to Layer
    model.newLayer().addAgentFunction(force_func)
    
    return model