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
            
            match tm_sum.ty_checked:
                case TySum(tys):

                    ptr = 0
                    slices = []
                    for i in range(len(tys)):
                        offset = dim(tys[i])
                        slices.append((ptr, ptr + offset))
                        ptr += offset

                    class NnCase(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.tm_sum_compiled = compile(tm_sum)
                            self.tms_compiled = nn.ModuleList([compile(tm) for tm in tms])
                        def forward(self, env: Env):
                            v_sum = self.tm_sum_compiled(env)
                            vs = []
                            for i in range(len(self.tms_compiled)):
                                v_i = v_sum[slices[i][0]: slices[i][1]]
                                vs.append(self.tms_compiled[i](env | {xs[i]: v_i}))
                            return torch.stack(vs).sum(dim=0)
                    
                    return NnCase()
                        
                case _:
                    raise TypeError(f"{tm_sum=} is not a sum type: {tm_sum.ty_checked=}.")
                
                
        case TmSeq(tm1, tm2):

            class NnSeq(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.tm1_compiled = compile(tm1)
                    self.tm2_compiled = compile(tm2)
                def forward(self, env: Env):
                    return self.tm1_compiled(env) * self.tm2_compiled(env)

            return NnSeq()
        
        case TmProj(n, tm_prod):
            match tm_prod.ty_checked:
                case TyProd(tys):

                    ptr = 0
                    slices = []
                    for i in range(len(tys)):
                        offset = dim(tys[i])
                        slices.append((ptr, ptr + offset))
                        ptr += offset

                    class NnProj(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.tm_prod_compiled = compile(tm_prod)
                        def forward(self, env: Env):
                            v_prod = self.tm_prod_compiled(env)
                            return v_prod[slices[n][0]: slices[n][1]]
                    
                    return NnProj()

                case _:
                    raise TypeError(f"{tm_prod=} is not a prod type: {tm_prod.ty_checked=}.")

        case TmChoice(tm1, tm2):

            class NnChoice(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.tm1_compiled = compile(tm1)
                    self.tm2_compiled = compile(tm2)
                def forward(self, env: Env):
                    return self.tm1_compiled(env) + self.tm2_compiled(env)

            return NnChoice()

        case TmDict(tm_ks, tm_vs):
            match (tm_ks.ty_checked, tm_vs.ty_checked):
                case (TyProd(key_tys), TyProd(val_tys)):
                    n = len(key_tys)
                    d_k = dim(key_tys[0])
                    d_v = dim(val_tys[0])

                    class NnDict(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.tm_ks_compiled = compile(tm_ks)
                            self.tm_vs_compiled = compile(tm_vs)
                        def forward(self, env: Env):
                            K = self.tm_ks_compiled(env).reshape(n, d_k)
                            V = self.tm_vs_compiled(env).reshape(n, d_v)
                            return K.T @ V

                    return NnDict()

                case _:
                    raise TypeError(f"TmDict: expected TyProd types, got {tm_ks.ty_checked=}, {tm_vs.ty_checked=}.")
        
        case TmLookup(tm1, tm2, rel):
            # We know the type of queries
            # We know the type of keys
            # So we can sample the behavior of rel on the basis for
            # queries to compute the matrix?
            raise NotImplementedError()

        case _:
            raise NotImplementedError()



def compile_val(v: Val) -> nn.Module:
    match v:
        case VUnit():

            class NnUnit(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.value = nn.Parameter(torch.tensor([1.0]))
                def forward(self, _: Env) -> torch.Tensor:
                    return self.value

            return NnUnit()

        case VError():

            class NnError(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.value = nn.Parameter(torch.tensor([0.0]))
                def forward(self, _: Env) -> torch.Tensor:
                    return self.value

            return NnError()

        case VInj(m, v_sum, TySum(tys)):

            class NnInj(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.v_compiled = compile_val(v_sum)
                    self.zero_left: torch.Tensor
                    self.zero_right: torch.Tensor
                    self.register_buffer('zero_left', zero(TySum(tys[:m])))
                    self.register_buffer('zero_right', zero(TySum(tys[m+1:])))
                def forward(self, env: Env) -> torch.Tensor:
                    return torch.cat([self.zero_left, self.v_compiled(env), self.zero_right])
            
            return NnInj()


        case VProd(tms, stored_env):
            stored_compiled = {x: compile_val(v) for x, v in stored_env.items()}

            class NnProd(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.tms_compiled = nn.ModuleList([compile(tm) for tm in tms])
                    self.stored_compiled = nn.ModuleDict(stored_compiled)
                def forward(self, env: Env) -> torch.Tensor:
                    extra = {x: m(env) for x, m in self.stored_compiled.items()}
                    full_env = env | extra
                    return torch.cat([tm_compiled(full_env) for tm_compiled in self.tms_compiled])

            return NnProd()

        case VDict(v1, v2):
            raise NotImplementedError()

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
