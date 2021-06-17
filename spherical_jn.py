import torch
from torch.autograd import Function
from scipy.special import spherical_jn


class SphericalJN(Function):
    @staticmethod
    def forward(ctx, n, input, derivative=False):
        ctx.save_for_backward(input, n)
        output = spherical_jn(n.detach().numpy(),
                              input.detach().numpy())
        return torch.tensor(output)

    @staticmethod
    def backward(ctx, grad_output):
        input, n = ctx.saved_tensors
        input_d = spherical_jn(n.detach().numpy(),
                               input.detach().numpy(),
                               derivative=True)
        input_d = torch.tensor(input_d)
        return None, input_d*grad_output


class SphericalJND(Function):
    @staticmethod
    def forward(ctx, n, input):
        output = spherical_jn(n.detach().numpy(),
                              input.detach().numpy(),
                              derivative=True)
        output = torch.tensor(output)
        ctx.save_for_backward(input, n, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, n, output = ctx.saved_tensors
        input_d = -2*output/input + \
            ((n+n**2)/input**2-1) * \
            spherical_jn(n, input)
        return None, input_d*grad_output
