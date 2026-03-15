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
        
        case TmError():
            return VError()
        
        case TmPair(tm1, tm2):
            return VPair(tm1, tm2)
        
        case TmInj1(tm, ty):
            v = evaluate(tm, env)
            match v:
                case VError():
                    return VError()
                case _:
                    return VInj1(v, ty)
                
        case TmInj2(tm, ty):
            v = evaluate(tm, env)
            match v:
                case VError():
                    return VError()
                case _:
                    return VInj2(v, ty)

        case TmDict(tm1s, tm2s):
            v1s = [evaluate(tm1, env) for tm1 in tm1s]
            v2s = [evaluate(tm2, env) for tm2 in tm2s]
            if (VError() in v1s) or (VError() in v2s):
                return VError()
            else:
                return VDict(v1s, v2s)
                
        case TmCase(tm, x1, tm1, x2, tm2):
            v = evaluate(tm, env)
            match v:
                case VError():
                    return VError()
                case VInj1(v1, ty):
                    return evaluate(tm1, env | {x1: v1})
                case VInj2(v2, ty):
                    return evaluate(tm2, env | {x2: v2})
                case _:
                    raise ValueError(f"Unexpected value: {v}")

        case TmProj1(tm):
            v = evaluate(tm, env)
            match v:
                case VError():
                    return VError()
                case VPair(tm1, tm2):
                    return evaluate(tm1, env)
                case _:
                    raise ValueError(f"Unexpected value: {v}")

        case TmProj2(tm):
            v = evaluate(tm, env)
            match v:
                case VError():
                    return VError()
                case VPair(tm1, tm2):
                    return evaluate(tm2, env)
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
                case (VDict(ks, vs), q):
                    V = [v for (k,v) in zip(ks, vs) if rel(k,q)]
                    if V:
                        return choice(V)
                    else:
                        return VError()
                case _:
                    raise ValueError(f"Unexpected values: {v1}, {v2}")