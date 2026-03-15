from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

# --- Types ---

type Ty = (  TyUnit
           | TySum
           | TyProd
           | TyDict
           )

@dataclass
class TyUnit: ...

@dataclass
class TySum:
    ty1: Ty
    ty2: Ty

@dataclass
class TyProd:
    ty1: Ty
    ty2: Ty

@dataclass
class TyDict:
    ty1: Ty
    ty2: Ty


# --- Terms ---

type Tm = (  TmVar
           | TmUnit 
           | TmError
           | TmPair
           | TmInj1
           | TmInj2
           | TmDict
           | TmProj1
           | TmProj2
           | TmCase
           | TmChoice
           | TmLookup
           | TmLet
           )

@dataclass
class TmVar:
    name: str

@dataclass
class TmUnit: ...

@dataclass
class TmError: ...

@dataclass
class TmPair:
    tm1: Tm
    tm2: Tm

@dataclass
class TmInj1:
    tm: Tm
    ty: Ty

@dataclass
class TmInj2:
    tm: Tm
    ty: Ty

@dataclass
class TmDict:
    tm1: list[Tm]
    tm2: list[Tm]

@dataclass
class TmProj1:
    tm: Tm

@dataclass
class TmProj2:
    tm: Tm

@dataclass
class TmCase:
    tm: Tm
    name1: str
    tm1: Tm
    name2: str
    tm2: Tm

@dataclass
class TmChoice:
    tm1: Tm
    tm2: Tm

@dataclass
class TmLookup:
    tm1: Tm
    tm2: Tm
    rel: Callable[[Any, Any], bool]

@dataclass
class TmLet:
    name: str
    tm1: Tm
    tm2: Tm


# --- Values ---

type Val = (  VUnit 
            | VInj1
            | VInj2
            | VPair
            | VDict
            | VError
            )

@dataclass
class VUnit: ...

@dataclass
class VError: ...

@dataclass
class VInj1:
    v: Val
    ty: Ty

@dataclass
class VInj2:
    v: Val
    ty: Ty

@dataclass
class VPair:
    tm1: Tm
    tm2: Tm

@dataclass
class VDict:
    v1: list[Val]
    v2: list[Val]

