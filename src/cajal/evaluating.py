from random import random
from random import choice
from collections.abc import Mapping
from cajal.syntax import *

type Env = Mapping[str, tuple[Val, Ty]]

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

        case TmDict(tm1, tm2):
            v1 = evaluate(tm1, env)
            v2 = evaluate(tm2, env)
            match (v1, v2):
                case (VError(), _):
                    return VError()
                case (_, VError()):
                    return VError()
                case _:
                    return VDict(v1, v2)
                
        case TmCase(tm, x1, tm1, x2, tm2):
            v = evaluate(tm, env)
            match v:
                case VError():
                    return VError()
                case VInj1(v1, ty):
                    return evaluate(tm1, env | {x1: v1})
                case VInj2(v2, ty):
                    return evaluate(tm2, env | {x2: v2})
                
        case TmProj1(tm):
            v = evaluate(tm, env)
            match v:
                case VError():
                    return VError()
                case VPair(tm1, tm2):
                    return evaluate(tm1, env)

        case TmProj2(tm):
            v = evaluate(tm, env)
            match v:
                case VError():
                    return VError()
                case VPair(tm1, tm2):
                    return evaluate(tm2, env)
                
        case TmChoice(tm1, tm2):
            if random() < .5:
                return evaluate(tm1, env)
            else:
                return evaluate(tm2, env)
            
        case TmLet(x, tm1, tm2):
            v1 = evaluate(tm1)
            return evaluate(tm2, env | {x: v1})
        
        case TmLookup(tm1, tm2, rel):
            v1 = evaluate(tm1)
            v2 = evaluate(tm2)
            match (v1, v2):
                case (VError(), _):
                    return VError()
                case (_, VError()):
                    return VError()
                case (VDict(ks, vs), q):
                    V = [v for (k,v) in zip(ks, vs) if rel(k,q)]
                    return choice(V)

        case _:
            raise ValueError(f"Unexpected term: {tm}")

