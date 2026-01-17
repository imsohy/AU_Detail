import torch
x = torch.Tensor([1,2,3,4,5,6,7,8,9,10]).view(2,1,5)
m = torch.Tensor([[3]])[:,None,...]
print(x)
print(torch.mean(x, dim=0))
print(x.shape)
print(m.shape)
print(x[1:].shape)
print(torch.cat((x[1:],m),dim=2))
# print(torch.mean(x, dim=0))
# print(torch.mean(x, dim=0).shape)
#
# print(torch.mean(x, dim=1))
# print(torch.mean(x, dim=1).shape)
# print(torch.mean(x, dim=2))
# print(torch.mean(x, dim=2).shape)
#
# print(x[:,None,...])
# print(x[:,None,...].shape)
#
# print(torch.mean(x[:,:,1:4], dim=0))
# print(torch.mean(x[:,:,1:4], dim=0).shape)
