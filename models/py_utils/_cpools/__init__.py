import torch

from torch import nn
from torch.autograd import Function

import top_pool, bottom_pool, left_pool, right_pool

class TopPoolFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = top_pool.forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input  = ctx.saved_variables[0]
        output = top_pool.backward(input, grad_output)[0]
        return output

class BottomPoolFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = bottom_pool.forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input  = ctx.saved_variables[0]
        output = bottom_pool.backward(input, grad_output)[0]
        return output

class LeftPoolFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = left_pool.forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input  = ctx.saved_variables[0]
        output = left_pool.backward(input, grad_output)[0]
        return output

class RightPoolFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = right_pool.forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input  = ctx.saved_variables[0]
        output = right_pool.backward(input, grad_output)[0]
        return output

class TopPool(nn.Module):
    def forward(self, x):
        return TopPoolFunction.apply(x)

class BottomPool(nn.Module):
    def forward(self, x):
        return BottomPoolFunction.apply(x)

class LeftPool(nn.Module):
    def forward(self, x):
        return LeftPoolFunction.apply(x)

class RightPool(nn.Module):
    def forward(self, x):
        return RightPoolFunction.apply(x)
