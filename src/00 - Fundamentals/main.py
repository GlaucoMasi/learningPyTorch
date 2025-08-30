import torch
import numpy as np

TENSOR = torch.tensor([[[1, 2, 3], [3, 6, 9], [2, 4, 5]]])
TENSOR, TENSOR.ndim, TENSOR.shape

random_tensor = torch.rand(size = (224, 224, 3))

ones = torch.ones(size = (3, 4))
zeros = torch.zeros(size = (3, 4))

range_tensor = torch.arange(start = 0, end = 100, step = 10)

# To create a similiar (same shape) tensor of zeros/ones
similar_tensor = torch.zeros_like(input = range_tensor)

tensor = torch.rand(3, 4)
print(tensor)
print(tensor.shape)
print(tensor.dtype)
print(tensor.device)

tensor = torch.tensor([1, 2, 3])
print(tensor + 10)
print(tensor + tensor)
print(tensor * 10)
print(tensor * tensor)

# Dimensions have to match:
# (2, 3) @ (3, 4) -> (2, 4), (2, 3) @ (2, 4) won't work
print(torch.matmul(tensor, tensor))

tensorA = torch.rand(2, 3)
tensorB = torch.rand(5, 3)
# torch.matmul(tensorA, tensorB) won't work
torch.mm(tensorA, tensorB.T)
torch.mm(tensorA, torch.transpose(tensorB, 0, 1))

torch.manual_seed(42)
linear = torch.nn.Linear(in_features = 3, out_features = 6)
x = torch.rand(2, 3)
print(linear(x))

x = torch.arange(0, 100, 10)
x.min(), torch.min(x)
x.max(), torch.max(x)
x.sum(), torch.sum(x)
x.type(torch.float32).mean(), torch.mean(x.type(torch.float32))

x.argmax(), torch.argmax(x)
x.argmin(), torch.argmin(x)

x = torch.arange(1, 8)
z = x.reshape(1, 7), x.reshape(1, 1, -1)
z = x.view(1, 7)
z = torch.stack([x, x, x, x], dim = 0)
z = x.squeeze()
z = x.unsqueeze(dim = 1)

x = torch.rand(3, 4, 2)
z = x.permute(2, 1, 0)

print(x)
print(x[1])
print(x[1][3])
print(x[0][0][1])

# Slicing: start:end
# : -> all, :10 -> first ten, 5:10 -> middle 5
print(x[:, 0])

array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
array = tensor.numpy()

# Reproducibility
torch.manual_seed(42)
tensorA = torch.rand(3, 4)

torch.manual_seed(42)
tensorB = torch.rand(3, 4)

print(tensorA == tensorB)

#GPUtensor = torch.rand(3, 4).to("cuda") Not actually working on this machine
#GPUtensor.numpy() won't work
#GPUtensor.cpu().numpy()