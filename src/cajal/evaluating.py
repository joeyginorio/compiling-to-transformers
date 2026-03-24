from random import random
from random import choice
from cajal.syntax import *

'''
This file implements the interpreter for Cajal.
'''

# --- Evaluation ---

type Env = dict[str, Val]

def evaluate(tm: Tm, env: Env) -> Val:
    match tm:

        case TmVar(x):
            return env[x]
        
        case TmUnit():
            return VUnit()
        
        case TmProd(tms):
            return VProd(tms)
        
        case TmInj(n, tm, ty):
            v = evaluate(tm, env)
            match v:
                case VError():
                    return VError()
                case _:
                    return VInj(n, v, ty)

        case TmDict(tm1, tm2):
            v1 = evaluate(tm1, env)
            v2 = evaluate(tm2, env)
            match (v1, v2):
                case (VError(), _) | (_, VError()):
                    return VError()
                case _:
                    return VDict(v1, v2)
                
        case TmSeq(tm1, tm2):
            v1 = evaluate(tm1, env)
            v2 = evaluate(tm2, env)
            match v1:
                case VError():
                    return VError()
                case VUnit():
                    return v2
                case _:
                    raise ValueError(f"Unexpected value: {v1}")

        case TmCase(tm, xs, tms):
            v = evaluate(tm, env)
            match v:
                case VError():
                    return VError()
                case VInj(n, v1, ty):
                    return evaluate(tms[n], env | {xs[n]: v1})
                case _:
                    raise ValueError(f"Unexpected value: {v}")

        case TmProj(n, tm):
            v = evaluate(tm, env)
            match v:
                case VError():
                    return VError()
                case VProd(tms):
                    if 0 <= n < len(tms):
                        return evaluate(tms[n], env)
                    else:
                        raise ValueError(f"{n=} is projecting out-of-bounds.")
                case _:
                    raise ValueError(f"Unexpected value: {v}")

        case TmChoice(tm1, tm2):
            if random() < .5:
                return evaluate(tm1, env)
            else:
                return evaluate(tm2, env)
            
        case TmLet(x, tm1, tm2):
            v1 = evaluate(tm1, env)
            return evaluate(tm2, env | {x: v1})
        
        case TmLookup(tm1, tm2, rel):
            v1 = evaluate(tm1, env)
            v2 = evaluate(tm2, env)
            match (v1, v2):
                case (VError(), _):
                    return VError()
                case (_, VError()):
                    return VError()
                case (VDict(VProd(ks), VProd(vs)), q):
                    V = [evaluate(v, env) for (k, v) in zip(ks, vs) if rel(evaluate(k, env), q)]
                    if V:
                        return choice(V)
                    else:
                        return VError()
                case _:
                    raise ValueError(f"Unexpected values: {v1}, {v2}")

        case _:
            raise ValueError(f"Unexpected term: {tm}")