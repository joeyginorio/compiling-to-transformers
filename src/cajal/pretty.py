from cajal.syntax import *

'''
This file implements a pretty printer for Cajal terms, types, and values.
'''


# --- Types ---

def pretty_ty(ty: Ty) -> str:
    match ty:
        case TyUnit():
            return "1"
        case TySum(tys):
            return " + ".join(_ty_atom(t) for t in tys)
        case TyProd(tys):
            return " * ".join(_ty_atom(t) for t in tys)
        case TyDict(ty1, ty2):
            return f"{_ty_atom(ty1)} -> {_ty_atom(ty2)}"


def _ty_atom(ty: Ty) -> str:
    """Wrap compound types in parens when used as subterms."""
    match ty:
        case TySum() | TyProd() | TyDict():
            return f"({pretty_ty(ty)})"
        case _:
            return pretty_ty(ty)


# --- Terms ---

def pretty(tm: Tm, indent: int | None = None) -> str:
    if indent is None:
        return _flat(tm)
    return _indented(tm, indent)


def _flat(tm: Tm) -> str:
    match tm:
        case TmUnit():
            return "()"
        case TmVar(name):
            return name
        case TmProd(tms):
            return "(" + ", ".join(_flat(t) for t in tms) + ")"
        case TmInj(n, t, ty):
            # TyDict uses '->' which conflicts with the TmDict separator; wrap it.
            ann = f"({pretty_ty(ty)})" if isinstance(ty, TyDict) else pretty_ty(ty)
            return f"inj_{n}({_flat(t)}) : {ann}"
        case TmDict(tm1, tm2):
            return "{" + _flat(tm1) + " -> " + _flat(tm2) + "}"
        case TmSeq(tm1, tm2):
            return f"{_wrap(tm1)}; {_wrap(tm2)}"
        case TmProj(n, t):
            return f"proj_{n}({_flat(t)})"
        case TmCase(t, xs, tms):
            branches = " | ".join(f"{x} => {_flat(b)}" for x, b in zip(xs, tms))
            return f"case {_wrap(t)} of {{ {branches} }}"
        case TmChoice(tm1, tm2):
            return f"{_wrap(tm1)} | {_wrap(tm2)}"
        case TmLookup(tm1, tm2, _):
            return f"{_wrap_tight(tm1)}({_flat(tm2)})"
        case TmLet(name, tm1, tm2):
            return f"let {name} = {_flat(tm1)} in {_flat(tm2)}"


def _indented(tm: Tm, level: int) -> str:
    ind = "  " * level
    ind1 = "  " * (level + 1)
    match tm:
        case TmLet(name, tm1, tm2):
            return f"let {name} = {_flat(tm1)} in\n{ind}{_indented(tm2, level)}"
        case TmCase(t, xs, tms):
            first_x, *rest_xs = xs
            first_b, *rest_bs = tms
            branches = f"{first_x} => {_indented(first_b, level + 1)}"
            for x, b in zip(rest_xs, rest_bs):
                branches += f"\n{ind}| {x} => {_indented(b, level + 1)}"
            return f"case {_wrap(t)} of {{\n{ind1}{branches}\n{ind}}}"
        case TmSeq(tm1, tm2):
            return f"{_wrap(tm1)};\n{ind}{_indented(tm2, level)}"
        case TmChoice(tm1, tm2):
            return f"{_wrap(tm1)}\n{ind}| {_indented(tm2, level)}"
        case _:
            return _flat(tm)


def _is_compound(tm: Tm) -> bool:
    """True for terms without their own bracketing delimiters."""
    return isinstance(tm, (TmSeq, TmChoice, TmLet, TmCase))


def _wrap(tm: Tm) -> str:
    """Parenthesize compound terms when used as binary operator subterms."""
    s = _flat(tm)
    return f"({s})" if _is_compound(tm) else s


def _wrap_tight(tm: Tm) -> str:
    """Parenthesize anything non-atomic in function position of a lookup."""
    match tm:
        case TmVar() | TmUnit():
            return _flat(tm)
        case _:
            return f"({_flat(tm)})"


# --- Values ---

def pretty_val(v: Val) -> str:
    match v:
        case VUnit():
            return "()"
        case VError():
            return "error"
        case VInj(n, inner, ty):
            return f"inj_{n}({pretty_val(inner)}) : {pretty_ty(ty)}"
        case VProd(tms, env):
            return "(" + ", ".join(_flat(t) for t in tms) + ")"
        case VDict(v1, v2):
            return "{" + pretty_val(v1) + " -> " + pretty_val(v2) + "}"
