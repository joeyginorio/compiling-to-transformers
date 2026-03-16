from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

'''
This file implements the syntax for Cajal: its terms, values, and contexts.
'''

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
    tys: list[Ty]

@dataclass
class TyProd:
    tys: list[Ty]

@dataclass
class TyDict:
    ty1: Ty
    ty2: Ty


# --- Terms ---

type Tm = (  TmVar
           | TmUnit 
           | TmProd
           | TmInj
           | TmDict
           | TmProj
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
class TmProd:
    tms: list[Tm]

@dataclass
class TmInj:
    n: int
    tm: Tm
    ty: Ty

@dataclass
class TmDict:
    tm1: Tm
    tm2: Tm

@dataclass
class TmProj:
    n: int
    tm: Tm

@dataclass
class TmCase:
    tm: Tm
    xs: list[str]
    tms: list[Tm]

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
            | VInj
            | VProd
            | VDict
            | VError
            )

@dataclass
class VUnit: ...

@dataclass
class VError: ...

@dataclass
class VInj:
    n: int
    v: Val
    ty: Ty

@dataclass
class VProd:
    tm: list[Tm]

@dataclass
class VDict:
    v1: Val
    v2: Val


# --- Context ---

type Ctx = dict[str, Ty]