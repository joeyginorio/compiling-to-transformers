import torch
import torch.nn as nn
from cajal.typing import *

'''
This file implements the interpreter for Cajal.
'''

type Env = dict[str, torch.Tensor]

# --- Compiling --- 

def compile(tm: Tm) -> nn.Module:
    match tm:

        case TmVar(x):

            class NnVar(nn.Module):
                def forward(self, env: Env):
                    return env[x]

            return NnVar()

        case TmUnit():

            class NnUnit(nn.Module):
                def forward(self, _: Env) -> torch.Tensor:
                    return torch.tensor([1.0])

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
                            return (K.T @ V).flatten()

                    return NnDict()

                case _:
                    raise TypeError(f"TmDict: expected TyProd types, got {tm_ks.ty_checked=}, {tm_vs.ty_checked=}.")
        
        case TmLookup(tm1, tm2, rel):
            match (tm1.ty_checked, tm2.ty_checked):
                case (TyDict(ty_k, ty_v), ty_q) if ty_q is not None:
                    d_k = dim(ty_k)
                    d_v = dim(ty_v)
                    d_q = dim(ty_q)

                    R_mat = torch.zeros(d_k, d_q)
                    for ki, kv in enumerate(enum_basis(ty_k)):
                        for qi, qv in enumerate(enum_basis(ty_q)):
                            if rel(kv, qv):
                                R_mat[ki, qi] = 1.0

                    class NnLookup(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.tm1_compiled = compile(tm1)
                            self.tm2_compiled = compile(tm2)
                            self.R: torch.Tensor
                            self.register_buffer('R', R_mat)
                        def forward(self, env: Env):
                            M = self.tm1_compiled(env).reshape(d_k, d_v).T
                            q = self.tm2_compiled(env)
                            return M @ (self.R @ q)

                    return NnLookup()

                case _:
                    raise TypeError(f"TmLookup: expected TyDict and enum query, got {tm1.ty_checked=}, {tm2.ty_checked=}.")

        case _:
            raise NotImplementedError()

def compile_val(v: Val) -> nn.Module:
    match v:
        case VUnit():

            class NnUnit(nn.Module):
                def forward(self, _: Env) -> torch.Tensor:
                    return torch.tensor([1.0])

            return NnUnit()

        case VError(ty=ty):
            zero_tensor = zero(ty) if ty is not None else torch.tensor([0.0])

            class NnError(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.value: torch.Tensor
                    self.register_buffer('value', zero_tensor)
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
            check_val(VProd(tms, stored_env))
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

        case VDict(v_ks, v_vs):
            match (check_val(v_ks), check_val(v_vs)):
                case (TyProd(key_tys), TyProd(val_tys)):
                    n = len(key_tys)
                    d_k = dim(key_tys[0])
                    d_v = dim(val_tys[0])

                    class NnDictVal(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.v_ks_compiled = compile_val(v_ks)
                            self.v_vs_compiled = compile_val(v_vs)
                        def forward(self, env: Env) -> torch.Tensor:
                            K = self.v_ks_compiled(env).reshape(n, d_k)
                            V = self.v_vs_compiled(env).reshape(n, d_v)
                            return (K.T @ V).flatten()

                    return NnDictVal()
                
                case _:
                    raise TypeError(f"VDict requires product types, got {check_val(v_ks)=} and {check_val(v_vs)=}.")

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

def to_multilinear(module: nn.Module, rand: bool = False) -> nn.Module:

    class NnMultilinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.lmap: nn.Linear | None = None
            self.var_names: list[str] | None = None

        def build_lmap(self, env: Env) -> nn.Linear:
            # Fix a canonical ordering of free variables and their dimensions.
            var_names = list(env.keys())
            dims = [env[x].shape[0] for x in var_names]

            # Standard basis for each variable's vector space.
            bases = [torch.eye(d) for d in dims]

            # Dimension of the tensor product space (d_x1 * d_x2 * ... * d_xk).
            d_in = 1
            for d in dims:
                d_in *= d

            # Enumerate all combinations of basis vectors across variables.
            # Starting from [{}], each step expands by all basis vectors of the next variable,
            # giving all combinations: {x1: e_i, x2: e_j, ...} for all i, j, ...
            env_samples: list[Env] = [{}]
            for x, base in zip(var_names, bases):
                env_samples = [s | {x: e} for s in env_samples for e in base]

            # Sample the module on each basis combination.
            # Each output is one column of the weight matrix W.
            columns = []
            for env_sample in env_samples:
                with torch.no_grad():
                    columns.append(module(env_sample))

            # Stack columns into W of shape (d_out, d_in).
            # W represents the linear map in the tensor product space such that
            # W @ kron(x1, x2, ..., xk) = module({x1: x1, x2: x2, ...}).
            d_out = columns[0].shape[0]
            W = torch.stack(columns, dim=1)

            self.var_names = var_names
            lmap = nn.Linear(d_in, d_out, bias=False)
            
            if not rand:
                with torch.no_grad():
                    lmap.weight.copy_(W)

            return lmap

        def forward(self, env: Env) -> torch.Tensor:
            # Build the linear map on the first call, then cache it.
            if self.lmap is None:
                self.lmap = self.build_lmap(env)
            assert self.lmap is not None and self.var_names is not None
            # Map the environment to the tensor product space via successive Kronecker products.
            tensors = [env[x] for x in self.var_names]
            kron = tensors[0] if tensors else torch.tensor([1.0])
            for t in tensors[1:]:
                kron = torch.kron(kron, t)
            return self.lmap(kron)

    return NnMultilinear()


def enum_basis(ty: Ty) -> list[Val]:
    match ty:
        case TyUnit():
            return [VUnit()]
        case TySum(tys):
            return [VInj(i, VUnit(), ty) for i in range(len(tys))]
        case _:
            raise TypeError(f"enum_basis: expected enum type, got {ty=}")