from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any, Optional

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
           | TmSeq
           | TmProj
           | TmCase
           | TmChoice
           | TmLookup
           | TmLet
           )

@dataclass
class Term:
    ty_checked: Optional[Ty] = field(default=None, init=False)

@dataclass
class TmVar(Term):
    name: str

@dataclass
class TmUnit(Term): ...

@dataclass
class TmProd(Term):
    tms: list[Tm]

@dataclass
class TmInj(Term):
    n: int
    tm: Tm
    ty: Ty

@dataclass
class TmDict(Term):
    tm1: Tm
    tm2: Tm

@dataclass
class TmSeq(Term):
    tm1: Tm
    tm2: Tm

@dataclass
class TmProj(Term):
    n: int
    tm: Tm

@dataclass
class TmCase(Term):
    tm: Tm
    xs: list[str]
    tms: list[Tm]

@dataclass
class TmChoice(Term):
    tm1: Tm
    tm2: Tm

@dataclass
class TmLookup(Term):
    tm1: Tm
    tm2: Tm
    rel: Callable[[Any, Any], bool]

@dataclass
class TmLet(Term):
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
class VError:
    ty: Optional[Ty] = field(default=None, compare=False)

@dataclass
class VInj:
    n: int
    v: Val
    ty: Ty

@dataclass
class VProd:
    tm: list[Tm]
    env: dict = field(default_factory=dict)

@dataclass
class VDict:
    v1: Val
    v2: Val


# --- Context ---

type Ctx = dict[str, Ty]