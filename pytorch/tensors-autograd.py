"""
touch.autograd is a PyTorch package that provides automatic differentiation for all operations on Tensors. 
This engine is what powers neural network training.
It is a define-by-run framework, which means that your backprop is defined by how your code is run, and that every single iteration can be different.
---
NN functions have parameters (weights and biases) that are stored in tensors.
--
In this tutorial, we load a pretrained model 'resnet18' from torchvision.models, and use it to perform inference.
We create a random data tensor to represent a single image with 3 channels, and height & width of 64, and its corresponding label initialized to some random values. 
Label in pretrained models has shape (1,1000)

"""

import torch
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.randn(1,3,64,64)
labels = torch.randn(1,1000)

#Input data through the model
prediction = model(data) #forward pass
print(prediction)
#Output
#tensor([[ 0.0811,  0.0811,  0.0811,  ...,  0.0811,  0.0811,  0.0811]],
#       grad_fn=<AddmmBackward0>)  

#Define loss function - using the prediction and the labels
loss = (prediction - labels).sum()

#Next step is backpropagate the loss 
#Autograd calculates and stores the gradients for each parameter in the parameter's .grad attribute.
loss.backward() #backward pass

# Load optimizer with learning rate = 0.01 and momentum = 0.9
optim = torch.optim.SDG(model.parameters(), lr=1e-2, momentum=0.9)

#Call .step() to initiate the gradiaent descent. Optimizer iterates through all the parameters and updates their values based on the gradients stored in .grad attribute.
optim.step() #gradient descent



#Differntiation in AutoGrad
# Using 2 tensors a and b with autograd enabled = TRUE to track all operatiions

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
Q = 3*a**3 - b**2
#Assuming a and b are parameters of NN, Q is the error
#We want gradients of error (Q) with respect to parameters a and b

external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

#Equivalenty we can also aggregate Q into scalar and then call backward
# Q.sum().backward()
# Gradients are stored in a.grad and b.grad
# Check if collected gradients are correct
print(a.grad == 9*a**2) #True
print(b.grad == -2*b) #True


#generally, torch.autograd is Jacobian vector product
#If we have a vector valued function y = f(x) where x is a vector of size n and y is a vector of size m, then the Jacobian J is an m x n matrix where J[i][j] = dy[i]/dx[j]

#Computational Graph
# Autograd stores in a DAG with Functions objects
# Leaves are the inputs to the model and roots are the outputs



