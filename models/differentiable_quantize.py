import torch
import torch.nn as nn


class differentiable_quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input


class DifferentiableQuantize(nn.Module):
    def __init__(self):
        super(DifferentiableQuantize, self).__init__()

    def forward(self, input):
        dq=differentiable_quantize.apply
        output = dq(input)
        return output


if __name__ == "__main__":
    a = torch.tensor([3.0,4.2,5,6,7,8,9,0])
    a.requires_grad_()
    dq=DifferentiableQuantize()
    b=dq(a)
    # b=torch.round(a)
    b=torch.sum(b**2)
    b.backward()
    print(a,a.grad)