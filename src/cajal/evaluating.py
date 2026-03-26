from random import random
from random import choice
from cajal.syntax import *

'''
This file implements the interpreter for Cajal.
'''

# --- Evaluation ---

type Env = dict[str, Val]

def evaluate(tm: Tm, env: Env) -> list[Val]:
    match tm:

        case TmVar(x):
            return [env[x]]

        case TmUnit():
            return [VUnit()]

        case TmProd(tms):
            return [VProd(tms, env)]

        case TmInj(n, tm, ty):
            results = []
            for v in evaluate(tm, env):
                match v:
                    case VError():
                        results.append(VError())
                    case _:
                        results.append(VInj(n, v, ty))
            return results

        case TmDict(tm1, tm2):
            results = []
            for v1 in evaluate(tm1, env):
                for v2 in evaluate(tm2, env):
                    match (v1, v2):
                        case (VError(), _) | (_, VError()):
                            results.append(VError())
                        case _:
                            results.append(VDict(v1, v2))
            return results

        case TmSeq(tm1, tm2):
            results = []
            for v1 in evaluate(tm1, env):
                match v1:
                    case VError():
                        results.append(VError())
                    case VUnit():
                        results += evaluate(tm2, env)
                    case _:
                        raise ValueError(f"Unexpected value: {v1}")
            return results

        case TmCase(tm, xs, tms):
            results = []
            for v in evaluate(tm, env):
                match v:
                    case VError():
                        results.append(VError())
                    case VInj(n, v1, ty):
                        results += evaluate(tms[n], env | {xs[n]: v1})
                    case _:
                        raise ValueError(f"Unexpected value: {v}")
            return results

        case TmProj(n, tm):
            results = []
            for v in evaluate(tm, env):
                match v:
                    case VError():
                        results.append(VError())
                    case VProd(tms, stored_env):
                        if 0 <= n < len(tms):
                            results += evaluate(tms[n], stored_env)
                        else:
                            raise ValueError(f"{n=} is projecting out-of-bounds.")
                    case _:
                        raise ValueError(f"Unexpected value: {v}")
            return results

        case TmChoice(tm1, tm2):
            return evaluate(tm1, env) + evaluate(tm2, env)

        case TmLet(x, tm1, tm2):
            results = []
            for v1 in evaluate(tm1, env):
                results += evaluate(tm2, env | {x: v1})
            return results

        case TmLookup(tm1, tm2, rel):
            results = []
            for v1 in evaluate(tm1, env):
                for v2 in evaluate(tm2, env):
                    match (v1, v2):
                        case (VError(), _) | (_, VError()):
                            results.append(VError())
                        case (VDict(VProd(ks, ks_env), VProd(vs, vs_env)), q):
                            matched = False
                            for k, v in zip(ks, vs):
                                v_vals = evaluate(v, vs_env)
                                for kv in evaluate(k, ks_env):
                                    if rel(kv, q):
                                        results += v_vals
                                        matched = True
                            if not matched:
                                results.append(VError())
                        case _:
                            raise ValueError(f"Unexpected values: {v1}, {v2}")
            return results