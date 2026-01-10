import os

def load_kernel(filename):
    """Reads a C++ file from the kernels/ directory."""
    # This path logic works on both Windows and Linux/Colab
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kernel_path = os.path.join(base_path, "kernels", filename)
    
    with open(kernel_path, "r") as f:
        return f.read()