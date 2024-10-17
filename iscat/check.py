import torch

# Check if PyTorch is installed and print the version
print("PyTorch version:", torch.__version__)

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")

# Create a tensor and move it to the selected device
x = torch.rand(3, 3, device=device)
y = torch.rand(3, 3, device=device)

# Perform a simple operation
z = x + y

# Print the result
print("Tensor x:\n", x)
print("Tensor y:\n", y)
print("Sum of x and y (z):\n", z)
