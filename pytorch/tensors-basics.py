import torch
import numpy as np

#Tensors are a specialized data structure that are similar to arrays and matrices. 

#Tensor Initialization
# Directly from data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# From a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(f"Data Tensor: \n{x_data}\n")
print(f"Numpy Tensor: \n{x_np}\n")
#Output: Data Tensor: 
#tensor([[1, 2],
#        [3, 4]])   
#Numpy Tensor: 
#tensor([[1, 2],
#        [3, 4]])


# From another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n{x_ones}\n")
#Output: tensor([[1, 1],
#                [1, 1]])

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n{x_rand}\n")
#Output: tensor([[0.3744, 0.9507],
#                [0.7320, 0.5987]]) 

# With random or constant values
shape = (2,3,) # 2 rows, 3 columns

rand_tensor = torch.rand(shape)
print(f"Random Tensor: \n{rand_tensor}\n")
#Output: tensor([[0.3744, 0.9507, 0.7320],
#                [0.5987, 0.1560, 0.1560]])

ones_tensor = torch.ones(shape)
print(f"Ones Tensor: \n{ones_tensor}\n")
#Output: tensor([[1., 1., 1.],
#                [1., 1., 1.]])

zeros_tensor = torch.zeros(shape)
print(f"Zeros Tensor: \n{zeros_tensor}\n")
#Output: tensor([[0., 0., 0.],
#                [0., 0., 0.]])

#Tensor Attributes
#defines the shape, datatype and the device of the tensor
tensor = torch.rand(3,4) #3 rows, 4 columns
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
#Output: Shape of tensor: torch.Size([3, 4])
#        Datatype of tensor: torch.float32
#        Device tensor is stored on: cpu



#Tensor Operations
#Move tensor to GPU if available
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print("Tensor moved to GPU")

#Standard numpy-like indexing and slicing
tensor = torch.ones(4,4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:,0]}")
print(f"Last column: {tensor[:,-1]}")
tensor[:,1] = 0
print(f"Modified Tensor: \n{tensor}")
#Output: First row: tensor([1., 1., 1., 1.])
#        First column: tensor([1., 1., 1., 1.])
#        Last column: tensor([1., 1., 1., 1.])
#        Modified Tensor: 
#        tensor([[1., 0., 1., 1.],
#                [1., 0., 1., 1.],
#                [1., 0., 1., 1.],
#                [1., 0., 1., 1.]]) 

#Joining tensors along a given dimension
#using torch.cat
t1 = torch.cat([tensor, tensor, tensor], dim=1) #dim=0 for rows, dim=1 for columns
print(f"Concatenated Tensor along columns: \n{t1}")
#Output: tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
#                [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
#                [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
#                [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])

#using torch.stack
t2 = torch.stack([tensor, tensor, tensor], dim=0) #dim=0 will create a new dimension
print(f"Stacked Tensor along new dimension: \n{t2}")
#Output: tensor([[[1., 0., 1., 1.],
#                 [1., 0., 1., 1.],
#                 [1., 0., 1., 1.],
#                 [1., 0., 1., 1.]],
#                [[1., 0., 1., 1.],
#                 [1., 0., 1., 1.],
#                 [1., 0., 1., 1.],
#                 [1., 0., 1., 1.]],
#                [[1., 0., 1., 1.],
#                 [1., 0., 1., 1.],
#                 [1., 0., 1., 1.],
#                 [1., 0., 1., 1.]]])


#Multiplying tensors (element-wise multiplication)
y1 = tensor * tensor  # element-wise product
y2 = tensor.mul(tensor) # element-wise product
print(f"Element-wise multiplication using '*': \n{y1}")
print(f"Element-wise multiplication using 'mul': \n{y2}")
#Output: tensor([[1., 0., 1., 1.],
#                [1., 0., 1., 1.],
#                [1., 0., 1., 1.],
#                [1., 0., 1., 1.]])
#        tensor([[1., 0., 1., 1.],
#                [1., 0., 1., 1.],
#                [1., 0., 1., 1.],
#                [1., 0., 1., 1.]]) 

#Matrix multiplication
y3 = torch.matmul(tensor, tensor.T) # matrix product
y4 = tensor @ tensor.T # matrix product
print(f"Matrix multiplication using 'matmul': \n{y3}")
print(f"Matrix multiplication using '@': \n{y4}")
#Output: tensor([[3., 0., 3., 3.],
#                [3., 0., 3., 3.],
#                [3., 0., 3., 3.],
#                [3., 0., 3., 3.]])
#        tensor([[3., 0., 3., 3.],
#                [3., 0., 3., 3.],
#                [3., 0., 3., 3.],
#                [3., 0., 3., 3.]]) 

#In place operations. These operations modify the tensor in place and do not create a new tensor. They are denoted by a trailing underscore.
print(f"Original tensor before in-place operation: \n{tensor}")
tensor.add_(5)
print(f"Tensor after in-place addition: \n{tensor}")
#Output: Original tensor before in-place operation: 
#        tensor([[1., 0., 1., 1.],
#                [1., 0., 1., 1.],
#                [1., 0., 1., 1.],
#                [1., 0., 1., 1.]])
#        Tensor after in-place addition: 
#        tensor([[6., 5., 6., 6.],
#                [6., 5., 6., 6.],
#                [6., 5., 6., 6.],
#                [6., 5., 6., 6.]])
# Since in place tensor involve loss of information, use them with caution.


#Bridging with NumPy
#Converting a Torch Tensor to a NumPy array and vice versa is a breeze. They share the underlying memory locations, and changing one will change the other.
t = torch.ones(3)
print(f"Original Torch Tensor: {t}")
n = t.numpy()
print(f"Converted NumPy Array: {n}")
t.add_(1)
print(f"Torch Tensor after in-place addition: {t}")
print(f"NumPy Array after Torch Tensor modification: {n}")
#Output: Original Torch Tensor: tensor([1., 1., 1.])
#        Converted NumPy Array: [1. 1. 1.]
#        Torch Tensor after in-place addition: tensor([2., 2., 2.])
#        NumPy Array after Torch Tensor modification: [2. 2. 2.]

#Converting NumPy Array to Torch Tensor
n = np.ones(3)
print(f"Original NumPy Array: {n}")
#Output: Original NumPy Array: [1. 1. 1.]
t = torch.from_numpy(n)
print(f"Converted Torch Tensor: {t}")
#Output: Converted Torch Tensor: tensor([1., 1., 1.], dtype=torch.float64)
np.add(n, 1, out=n)
print(f"NumPy Array after in-place addition: {n}")
print(f"Torch Tensor after NumPy Array modification: {t}")
#Output: NumPy Array after in-place addition: [2. 2. 2.]
#        Torch Tensor after NumPy Array modification: tensor([2., 2., 2 ], dtype=torch.float64)
#The Torch Tensor reflects the changes made to the NumPy array. 
# 
# This is because they share the same memory location.
# Note: The default datatype for NumPy arrays is float64, while for Torch tensors it is float32. This can lead to unexpected behavior if not handled properly.
# Always ensure that the datatypes are compatible when converting between the two.
# 
# This concludes the basic overview of PyTorch tensors, their operations, and interoperability with NumPy.
# They are the fundamental building blocks for creating and training neural networks in PyTorch.






