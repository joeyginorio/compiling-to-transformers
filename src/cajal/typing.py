from cajal.syntax import *

'''
This file implements the typechecker for Cajal.
'''


# --- Typing ---

def _check(tm: Tm, ctx: Ctx) -> tuple[Ty, Ctx]:
    match tm:

        case TmVar(x):
            if x not in ctx:
                raise TypeError(f"TmVar: {x=} not in {ctx=}.")

            ctx_remain = {y: ty for (y, ty) in ctx.items() if y != x}
            tm.ty_checked = ctx[x]
            return tm.ty_checked, ctx_remain

        case TmUnit():
            tm.ty_checked = TyUnit()
            return tm.ty_checked, ctx

        case TmInj(n, tm, ty_sum):
            ty, ctx_remain = _check(tm, ctx)

            match ty_sum:
                case TySum(tys):
                    if ty != tys[n]:
                        raise TypeError(f"{ty_sum=} mismatches {n}-injection: {tys[n]}.")
                    tm.ty_checked = ty_sum
                    return tm.ty_checked, ctx_remain
                case _:
                    raise TypeError(f"{ty_sum=} is not a sum type.")

        case TmProd(tms):
            tys, ctxs_remain = zip(*[_check(tm, ctx) for tm in tms])

            if not all(ctx == ctxs_remain[0] for ctx in ctxs_remain):
                raise TypeError(f"{tms=} does not uniformly use context at each index.")

            tm.ty_checked = TyProd(list(tys))
            return tm.ty_checked, ctxs_remain[0]

        case TmDict(tm1, tm2):
            ty1, ctx_remain1 = _check(tm1, ctx)
            ty2, ctx_remain2 = _check(tm2, ctx_remain1)

            match (ty1, ty2):
                case (TyProd(tys_k), TyProd(tys_v)):
                    if len(tys_k) != len(tys_v):
                        raise TypeError(f"Dict keys and values have different lengths.")
                    if not all(ty == tys_k[0] for ty in tys_k):
                        raise TypeError(f"Dict keys are not homogeneous.")
                    if not all(ty == tys_v[0] for ty in tys_v):
                        raise TypeError(f"Dict values are not homogeneous.")
                    match tys_k[0]:
                        case TySum(tys) if all(ty == TyUnit() for ty in tys):
                            pass
                        case _:
                            raise TypeError(f"Key type must be of enum type, got {tys_k[0]=}.")
                    tm.ty_checked = TyDict(tys_k[0], tys_v[0])
                    return tm.ty_checked, ctx_remain2
                case _:
                    raise TypeError(f"Dict requires product types, got {ty1=} and {ty2=}.")

        case TmSeq(tm1, tm2):
            ty1, ctx_remain1 = _check(tm1, ctx)
            ty2, ctx_remain2 = _check(tm2, ctx_remain1)

            match ty1:
                case TyUnit():
                    tm.ty_checked = ty2
                    return tm.ty_checked, ctx_remain2
                case _:
                    raise TypeError(f"{tm1=} is not of Unit type: {ty2=}.")

        case TmLet(x, tm1, tm2):
            ty1, ctx_remain1 = _check(tm1, ctx)
            ty2, ctx_remain2 = _check(tm2, ctx_remain1 | {x: ty1})
            tm.ty_checked = ty2
            return tm.ty_checked, ctx_remain2

        case TmCase(tm_sum, xs, tms):
            ty, ctx_remain = _check(tm_sum, ctx)

            match ty:
                case TySum(tys):
                    ty_cases, ctxs_remain = zip(*[_check(tms[i], ctx_remain | {xs[i]: tys[i]}) for i in range(len(tms))])

                    if not all(ty_case == ty_cases[0] for ty_case in ty_cases):
                        raise TypeError(f"TmCase branches have different return types: {ty_cases=}.")

                    if not all(ctx_remain == ctxs_remain[0] for ctx_remain in ctxs_remain):
                        raise TypeError(f"TmCase branches do not uniformly use context.")

                    tm.ty_checked = ty_cases[0]
                    return ty_cases[0], ctxs_remain[0]

                case _:
                    raise TypeError(f"{tm=} is not of sum type, {ty=}.")

        case TmProj(n, tm_prod):
            ty, ctx_remain = _check(tm_prod, ctx)

            match ty:
                case TyProd(tys):
                    tm.ty_checked = tys[n]
                    return tm.ty_checked, ctx_remain
                case _:
                    raise TypeError(f"{tm=} is not of product type.")

        case TmChoice(tm1, tm2):
            ty1, ctx_remain1 = _check(tm1, ctx)
            ty2, ctx_remain2 = _check(tm2, ctx)

            if ty1 != ty2:
                raise TypeError(f"Types of choices must match, got {ty1=} and {ty2=}.")
            if ctx_remain1 != ctx_remain2:
                raise TypeError(f"TmChoice branches do not uniformly use context.")

            tm.ty_checked = ty1
            return tm.ty_checked, ctx_remain1

        case TmLookup(tm1, tm2, rel):
            ty1, ctx_remain1 = _check(tm1, ctx)
            ty2, ctx_remain2 = _check(tm2, ctx_remain1)

            match ty2:
                case TySum(tys) if all(ty == TyUnit() for ty in tys):
                    pass
                case _:
                    raise TypeError(f"Query type is not of enum type, got {ty2=}.")

            match ty1:
                case TyDict(ty_k, ty_v):
                    tm.ty_checked = ty_v
                    return tm.ty_checked, ctx_remain2
                case _:
                    raise TypeError(f"Dictionary is not of dictionary type, got {ty1=}.")


def check_val(v: Val) -> Ty:
    match v:

        case VUnit():
            return TyUnit()

        case VError():
            return TyUnit()

        case VInj(n, v_inner, ty):
            match ty:
                case TySum(tys):
                    ty_inner = check_val(v_inner)
                    if ty_inner != tys[n]:
                        raise TypeError(f"VInj: inner value type {ty_inner} does not match {tys[n]}.")
                    return ty
                case _:
                    raise TypeError(f"VInj: annotation {ty} is not a sum type.")

        case VProd(tms):
            return TyProd([check(tm, {}) for tm in tms])

        case VDict(v1, v2):
            ty1 = check_val(v1)
            ty2 = check_val(v2)

            match (ty1, ty2):
                case (TyProd(tys_k), TyProd(tys_v)):
                    if len(tys_k) != len(tys_v):
                        raise TypeError(f"Dict keys and values have different lengths.")
                    if not all(ty == tys_k[0] for ty in tys_k):
                        raise TypeError(f"Dict keys are not homogeneous.")
                    if not all(ty == tys_v[0] for ty in tys_v):
                        raise TypeError(f"Dict values are not homogeneous.")
                    match tys_k[0]:
                        case TySum(tys) if all(ty == TyUnit() for ty in tys):
                            pass
                        case _:
                            raise TypeError(f"Key type must be of enum type, got {tys_k[0]=}.")
                    return TyDict(tys_k[0], tys_v[0])
                case _:
                    raise TypeError(f"Dict requires product types, got {ty1=} and {ty2=}.")


def check(tm: Tm, ctx: Ctx) -> Ty:
    ty, ctx = _check(tm, ctx)
    if ctx:
        raise TypeError(f"Unused context during typechecking: {ctx=}")
    else:
        return ty