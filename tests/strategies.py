import hypothesis.strategies as st
from cajal.typing import *
from cajal.evaluating import *

'''
This file implements strategies for sampling various data necessary to write
property-based tests for Cajal. These samplers let us to "prove" theorems about
programs by sampling data and checking the theorems against each sample.
'''

type Gen[T] = st.SearchStrategy[T]

# --- Generating programs ---

@st.composite
def gen_prog(draw) -> tuple[Tm, Ctx, Ty]:
    ctx = draw(gen_ctx())
    ty  = draw(gen_ty())
    _, tm, ty = draw(gen_tm(ctx, ty))
    return (tm, ctx, ty)


# --- Generating types ---

def gen_ty(max_leaves: int = 4) -> Gen[Ty]:
    return st.recursive(
        gen_ty_unit(),
        lambda subty: st.one_of(
            gen_ty_sum(subty),
            # gen_ty_prod(subty),
            # gen_ty_dict(subty)
        ),
        max_leaves = max_leaves
    )

def gen_ty_unit() -> Gen[Ty]:
    return st.just(TyUnit())

def gen_ty_sum(gen_subty: Gen[Ty], min_size: int = 1, max_size: int = 2) -> Gen[Ty]:
    return st.builds(TySum, st.lists(gen_subty, min_size=min_size, max_size=max_size))

def gen_ty_prod(gen_subty: Gen[Ty], min_size: int = 1, max_size: int = 2) -> Gen[Ty]:
    return st.builds(TyProd, st.lists(gen_subty, min_size=min_size, max_size=max_size))

def gen_ty_enum(min_size: int = 1, max_size: int = 2) -> Gen[Ty]:
    return st.builds(TySum, st.lists(st.just(TyUnit()), min_size=min_size, max_size=max_size))

def gen_ty_dict(gen_subty: Gen[Ty]) -> Gen[Ty]:
    return st.builds(TyDict, gen_ty_enum(), gen_subty)


# --- Generating contexts ---

def gen_ctx(min_size: int = 0, max_size: int = 3) -> Gen[Ctx]:
    c = st.sampled_from("abcdefghijklmnopqrstuvwxyz")
    n = st.integers(min_value=0, max_value=100).map(str)
    name = st.builds(lambda c, n: c + n, *(c, n))
    return st.dictionaries(name, gen_ty(), min_size=min_size, max_size=max_size)


# --- Generating terms ---

@st.composite
def gen_tm(draw, ctx: Ctx, ty: Ty) -> Tm:
    match ty:

        case TyUnit():
            if len(ctx) == 1 and next(iter(ctx.values())) == ty:
                return draw(gen_tm_var(ctx))
            elif not ctx:
                return draw(gen_tm_unit())
            else:
                return draw(st.one_of([gen_tm_case(ctx, ty)]))
            
        case TySum(tys):
            if len(ctx) == 1 and next(iter(ctx.values())) == ty:
                return draw(gen_tm_var(ctx))
            else:
                return draw(st.one_of([gen_tm_inj(ctx, ty),
                                       gen_tm_case(ctx, ty)]))
                
        case _:
            raise TypeError(f"Can't generate a term of type {ty=} in {ctx=}.")

@st.composite
def gen_tm_unit(draw) -> Tm:
    return TmUnit()

@st.composite
def gen_tm_var(draw, ctx: Ctx) -> Tm:
    x, ty = next(iter(ctx.items()))
    return TmVar(x)

@st.composite
def gen_tm_inj(draw, ctx: Ctx, ty: Ty) -> Tm:
    match ty:
        case TySum(tys):
            n = draw(st.integers(min_value=0, max_value=len(tys)-1))
            tm_sub = draw(gen_tm(ctx, tys[n]))
            return TmInj(n, tm_sub, ty)
        case _:
            raise TypeError(f"Can't generate an injection of non-sum type, {ty=}.")

@st.composite
def gen_tm_case(draw, ctx: Ctx, ty: Ty) -> Tm:
    ty_sum = draw(gen_ty_sum(gen_ty()))
    ctx1, ctx2 = draw(split_ctx(ctx))
    tm_sum = draw(gen_tm(ctx1, ty_sum))

    xs = []
    tms = []
    c = draw(st.sampled_from("abcdefghijklmnopqrstuvwxyz"))
    for i in range(len(ty_sum.tys)):
        xs.append(c + str(i))
        tms.append(draw(gen_tm(ctx2 | {xs[i]: ty_sum.tys[i]}, ty)))

    return TmCase(tm_sum, xs, tms)

# --- Helpers ---
@st.composite
def split_ctx(draw, ctx: Ctx) -> tuple[Ctx, Ctx]:
    keys = list(ctx.keys())
    match len(keys):
        case 0:
            return {}, {}
        case 1:
            i = draw(st.integers(min_value=0, max_value=1))
            return (ctx, {}) if i == 0 else ({}, ctx)
        case _:
            i = draw(st.integers(min_value=1, max_value=len(keys)-1))
            return {k: ctx[k] for k in keys[:i]}, {k: ctx[k] for k in keys[i:]}