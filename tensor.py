import torch

# scalar is just a single number
# vector is just a 1D list of number
# matrix is a 2D list of number
# vector is a ND list of number
# ndim is the faces the data possess, like x, y, dept
scalar = torch.tensor(10)
vector = torch.tensor([1, 2, 4])
matrix = torch.tensor([[2, 4, 6], [1, 3, 5]])
tensor = torch.tensor([[[2, 2, 2],
                        [4, 4, 4],
                        [5, 5, 44]]])
tensor1 = torch.tensor([[[2, 2, 2],
                         [4, 4, 4],
                         [5, 5, 44]], [[2, 2, 2],
                                       [4, 4, 4],
                                       [5, 5, 44]]])

# random tensors
ranMatrix = torch.rand(6, 4)
ranTensor = torch.rand(2,  1, 1, 4)
imageTensor = torch.rand(size=(255, 244, 3))

# zeros tensors
zeroMatrix = torch.zeros(5, 6)
# ones tensors
oneMatrix = torch.ones(2, 4)

# range tensors
rangeMatrix = torch.arange(1, 10)
rangeMatrix1 = torch.arange(start=20, end= 200, step=50)

# likes tensors
zerolikes = torch.zeros_like(rangeMatrix1)

if __name__ == '__main__':
    print(zeroMatrix * 2)
    print(oneMatrix * 0)
    print(rangeMatrix)
    print(rangeMatrix1)
    print(zerolikes)
    print('scalar shape', scalar.shape, 'scalar ndim', scalar.ndim)
    print('vector shape', vector.shape, 'vector ndim', vector.ndim)
    print('matrix shape', matrix.shape, 'matrix ndim', matrix.ndim)
    print('tensor shape', tensor.shape, 'tensor ndim', tensor.ndim)
    print('tensor1 shape', tensor1.shape, 'tensor1 ndim', tensor1.ndim)
    print('ranMatrix shape', ranMatrix.shape, 'ranMatrix ndim', ranMatrix.ndim, ranMatrix.size())
    print('imageTensor shape', imageTensor.shape, 'imageTensor ndim', imageTensor.ndim)
    print('ranTensor shape', ranTensor.shape, 'ranTensor ndim', ranTensor.ndim)