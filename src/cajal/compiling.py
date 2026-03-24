import torch
import torch.nn as nn
from cajal.typing import *

'''
This file implements the interpreter for Cajal.
'''

type Env = dict[str, torch.Tensor]

# --- Compiling --- 

# WARNING: Not batch-sensitive. But worry about that later.
# 1. Use torch.vmap
# 2. Make intrinsically batch-sensitive
def compile(tm: Tm) -> nn.Module:
    match tm:

        case TmVar(x):

            class NnVar(nn.Module):
                def forward(self, env: Env):
                    return env[x]

            return NnVar()

        case TmUnit():

            class NnUnit(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.value = nn.Parameter(torch.tensor([1.0]))
                def forward(self, _: Env) -> torch.Tensor:
                    return self.value

            return NnUnit()

        case TmProd(tms):

            class NnProd(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.tms_compiled = nn.ModuleList([compile(tm) for tm in tms])
                def forward(self, env: Env) -> torch.Tensor:
                    return torch.cat([tm_compiled(env) for tm_compiled in self.tms_compiled])

            return NnProd()
        
        case TmInj(m, tm, TySum(tys)):

            class NnInj(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.tm_compiled = compile(tm)
                    self.zero_left: torch.Tensor
                    self.zero_right: torch.Tensor
                    self.register_buffer('zero_left', zero(TySum(tys[:m])))
                    self.register_buffer('zero_right', zero(TySum(tys[m+1:])))
                def forward(self, env: Env) -> torch.Tensor:
                    return torch.cat([self.zero_left, self.tm_compiled(env), self.zero_right])
            
            return NnInj()
        
        case TmLet(x, tm1, tm2):

            class NnLet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.tm1_compiled = compile(tm1)
                    self.tm2_compiled = compile(tm2)
                def forward(self, env: Env):
                    v1 = self.tm1_compiled(env)
                    return self.tm2_compiled(env | {x: v1})
                
            return NnLet()

        case TmCase(tm_sum, xs, tms):

            class NnCase(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.tm_sum_compiled = compile(tm_sum)
                    self.tms_compiled = nn.ModuleList([compile(tm) for tm in tms])
                def forward(self, env: Env):
                    v_sum = self.tm_sum_compiled(env)
                    return ...
                
            return NnCase()
                    
        case _:
            raise NotImplementedError()

# --- Helpers ---


def dim(ty: Ty) -> int:
    match ty:
        case TyUnit():
            return 1
        case TyProd(tys):
            return sum([dim(ty) for ty in tys])
        case TySum(tys):
            return sum([dim(ty) for ty in tys])
        case TyDict(ty1, ty2):
            return dim(ty1) * dim(ty2)


def zero(ty: Ty) -> torch.Tensor:
    return torch.zeros(dim(ty))
