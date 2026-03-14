from collections.abc import Mapping
from cajal.syntax import *

type Env = Mapping[str, tuple[Val, Ty]]

def evaluate(tm: Tm, env: Env) -> Val:
    match tm:
        case TmVar(x):
            v = env[x]
            return v
        case TmUnit():
            return VUnit()
        case TmError():
            return VError()
        case TmPair(tm1, tm2):
            return VPair(tm1, tm2)
        case TmInj1(tm):
            v = evaluate(tm, env)
            return VInj1(v)
        case TmInj2(tm):
            v = evaluate(tm, env)
            return VInj2(v)
        case TmDict(tm1, tm2):
            v1 = evaluate(tm1, env)
            v2 = evaluate(tm2, env)