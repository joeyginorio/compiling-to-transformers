from cajal.syntax import *

'''
This file implements the typechecker for Cajal.
'''

# TODO: Need an easy way to take a list of types and package them into a prod.
# Write a helper function to do this!
# TODO: CONSIDER CHANGING THE PAIR TO BE N-ARY. 
# TODO: Projections must take an integer argument. 
# The above changes probably will make things easier down the line.
# TODO: Consider changing sum to be n-ary as well!
# TODO: Have to enforce enum

# --- Typing ---

def _check(tm: Tm, ctx: Ctx) -> tuple[Ty, Ctx]:
    match tm:

        case TmVar(x):
            if x not in ctx:
                raise TypeError(f"TmVar: {x=} not in {ctx=}.")
            
            ctx_remain = {y: ty for (y, ty) in ctx.items() if y != x}
            return ctx[x], ctx_remain
        
        case TmUnit():
            return TyUnit(), ctx
        
        case TmInj1(tm, ty_sum):
            ty1, ctx_remain = _check(tm, ctx)

            match ty_sum:
                case TySum(ty_sum1, _):
                    if ty1 != ty_sum1:
                        raise TypeError(f"{ty_sum=} mismatches first injection {ty1}.")
                    return ty_sum, ctx_remain
                case _:
                    raise TypeError(f"{ty_sum=} is not a sum type.")
        
        case TmInj2(tm, ty_sum):
            ty2, ctx_remain = _check(tm, ctx)

            match ty_sum:
                case TySum(_, ty_sum2):
                    if ty2 != ty_sum2:
                        raise TypeError(f"{ty_sum=} mismatches first injection {ty2}.")
                    return ty_sum, ctx_remain
                case _:
                    raise TypeError(f"{ty_sum} is not a sum type.")
        
        case TmPair(tm1, tm2):
            ty1, ctx_remain1 = _check(tm1, ctx.copy())
            ty2, ctx_remain2 = _check(tm2, ctx.copy())
            return TyProd(ty1, ty2), ctx_remain1 | ctx_remain2
        
        case TmDict(tm1s, tm2s):
            if len(tm1s) != len(tm2s):
                raise TypeError(f"{tm1s=} and {tm2s=} are not same length.")

            ty1s, ctx1s_remain = zip(*[_check(tm, ctx.copy()) for tm in tm1s])

            if not all(ty == ty1s[0] for ty in ty1s):
                raise TypeError(f"{tm1s=} is not a homogeneous list.")
            
            if not all(ctx1 == ctx1s_remain[0] for ctx1 in ctx1s_remain):
                raise TypeError(f"{tm1s=} does not uniformly use context at each index.")

            ty2s, ctx2s_remain = zip(*[_check(tm, ctx1s_remain[0]) for tm in tm2s])

            if not all(ty == ty2s[0] for ty in ty2s):
                raise TypeError(f"{tm2s=} is not a homogeneous list.")
            
            if not all(ctx2 == ctx2s_remain[0] for ctx2 in ctx2s_remain):
                raise TypeError(f"{tm2s=} does not uniformly use context at each index.")
            
            return ...


        # STUB: Change
        case _:
            return TyUnit()
        


def check_val(val: Val) -> Ty:
    ...