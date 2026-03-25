import hypothesis.strategies as st
from collections.abc import Mapping
from cajal.typing import *
from cajal.evaluating import *

'''
This file implements strategies for sampling various data necessary to write
property-based tests for Cajal. These samplers let us to "prove" theorems about
programs by sampling data and checking the theorems against each sample.
'''

type Gen[T] = st.SearchStrategy[T]

@dataclass
class GCtx():
    pos: Ctx
    neg: Ctx

    def flat(self) -> Ctx:
        return self.pos | self.neg

    def __len__(self) -> int:
        return len(self.pos) + len(self.neg)

    def __bool__(self) -> bool:
        return bool(self.pos) or bool(self.neg)

    def __or__(self, other: 'Mapping[str, Ty]') -> 'GCtx':
        pos = dict(self.pos)
        neg = dict(self.neg)
        for k, ty in other.items():
            if positive(ty):
                pos[k] = ty
            else:
                neg[k] = ty
        return GCtx(pos=pos, neg=neg)


# --- Generating programs ---

@st.composite
def gen_closed_prog(draw) -> tuple[Tm, Ty]:
    ty = draw(gen_ty())
    ctx = GCtx(pos={}, neg={})
    tm = draw(gen_tm(ctx, ty, set()))
    return (tm, ty)

@st.composite
def gen_prog(draw) -> tuple[GCtx, Tm, Ty]:
    ctx = draw(gen_ctx())
    ty  = draw(gen_ty())
    used: set[str] = set(ctx.flat().keys())
    tm = draw(gen_tm(ctx, ty, used))
    return (ctx, tm, ty)


# --- Generating types ---


def gen_ty(max_leaves: int = 3) -> Gen[Ty]:
    return st.recursive(
        gen_ty_unit(),
        lambda subty: st.one_of(
            gen_ty_sum(subty),
            gen_ty_prod(subty),
            # gen_ty_dict(subty)
        ),
        max_leaves = max_leaves
    )

def gen_ty_pos(max_leaves: int = 3) -> Gen[Ty]:
    return st.recursive(
        gen_ty_unit(),
        lambda subty: st.one_of(
            gen_ty_sum(subty),
            # gen_ty_dict(subty)
        ),
        max_leaves = max_leaves
    )

def gen_ty_neg(max_leaves: int = 3) -> Gen[Ty]:
    return st.recursive(
        gen_ty(),
        lambda subty: st.one_of(
            gen_ty_prod(subty)
        ),
        max_leaves = max_leaves,
        min_leaves = 2
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

@st.composite
def gen_ctx(draw, min_size: int = 0, max_size: int = 4) -> GCtx:
    c = st.sampled_from("abcdefghijklmnopqrstuvwxyz")
    n = st.integers(min_value=0, max_value=100).map(str)
    name = st.builds(lambda c, n: '_' + c + n, *(c, n))

    names = draw(st.lists(name, min_size=min_size, max_size=max_size, unique=True))
    split = draw(st.integers(min_value=0, max_value=len(names)))
    pos_names, neg_names = names[:split], names[split:]

    pos = {x: draw(gen_ty_pos()) for x in pos_names}
    neg = {x: draw(gen_ty_neg()) for x in neg_names}
    return GCtx(pos=pos, neg=neg)

# --- Generating terms ---

@st.composite
def gen_tm(draw, ctx: GCtx, ty: Ty, used: set[str]) -> Tm:

    # 1. Can you use the variable rule?
    if len(ctx) == 1 and next(iter(ctx.flat().values())) == ty:
        return draw(gen_tm_var(ctx))

    # 2. Can you do any closed introductions?
    elif not ctx:
        match ty:
            case TyUnit():
                return TmUnit()
            case TySum(tys):
                n = draw(st.integers(min_value=0, max_value=len(tys)-1))
                tm = draw(gen_tm(ctx, tys[n], used))
                return TmInj(n, tm, ty)
            case TyProd(tys):
                tms = []
                for i in range(len(tys)):
                    tms.append(draw(gen_tm(ctx, tys[i], used)))
                return TmProd(tms)
            case TyDict(ty1, ty2):
                n = draw(st.integers(min_value=1, max_value=3))
                ty1_list = TyProd([ty1]*n)
                ty2_list = TyProd([ty2]*n)
                tm1 = draw(gen_tm(ctx, ty1_list, used))
                tm2 = draw(gen_tm(ctx, ty2_list, used))
                return TmDict(tm1, tm2)

    # 3. Are there any invertible introductions?
    elif not positive(ty):
        match ty:
            case TyProd(tys):
                tms = []
                for i in range(len(tys)):
                    tms.append(draw(gen_tm(ctx, tys[i], used)))
                return TmProd(tms)
            case _:
                raise AssertionError("Impossible code path hit.")

    # 4. Are there any invertible eliminations?
    elif ctx.pos:
        pos = ctx.pos.copy()
        x, ty_pos = pos.popitem()
        ctx = GCtx(pos=pos, neg=ctx.neg)
        match ty_pos:
            case TyUnit():
                tm1 = TmVar(x)
                tm2 = draw(gen_tm(ctx, ty, used))
                return TmSeq(tm1, tm2)
            case TySum(tys):
                tm1 = TmVar(x)
                xs = []
                tms = []
                for i in range(len(tys)):
                    name = draw(fresh(used))
                    xs.append(name)
                    tms.append(draw(gen_tm(ctx | {xs[i]: tys[i]}, ty, used)))

                return TmCase(tm1, xs, tms)
            case TyDict(ty1, ty2):
                name = draw(fresh(used))

                tm11 = TmVar(x)
                ctx1, ctx2 = draw(split_ctx(ctx))
                tm12 = draw(gen_tm(ctx1, ty1, used))
                tm1 = TmLookup(tm11, tm12, lambda x,y: x == y)
                tm2 = draw(gen_tm(ctx2 | {name: ty2}, ty, used))

                return TmLet(name, tm1, tm2)
            case _:
                raise AssertionError("Impossible code path hit.")

    # 5. Only non-invertible choices, begin focusing...
    elif ctx.neg:
        neg = ctx.neg.copy()
        x, ty_neg = neg.popitem()
        ctx = GCtx(pos=ctx.pos, neg=neg)

        # NOTE: Goal type must be positive!
        match ty_neg, ty:
            # D, y: AxB |- Unit
            case TyProd(tys), TyUnit():

                n = draw(st.integers(min_value=0, max_value=len(tys)-1))
                name = draw(fresh(used))

                tm1 = TmProj(n, TmVar(x))
                tm2 = draw(gen_tm(ctx | {name: tys[n]}, TyUnit(), used))

                return TmLet(name, tm1, tm2)

            # D, y: AxB |- C + D
            case TyProd(tys1), TySum(tys2):
                choice = draw(st.sampled_from(['elim', 'intro']))
                match choice:
                    case 'intro':
                        n = draw(st.integers(min_value=0, max_value=len(tys2)-1))
                        tm = draw(gen_tm(ctx | {x: ty_neg}, tys2[n], used))
                        return TmInj(n, tm, ty)
                    case 'elim':
                        n = draw(st.integers(min_value=0, max_value=len(tys1)-1))
                        name = draw(fresh(used))

                        tm1 = TmProj(n, TmVar(x))
                        tm2 = draw(gen_tm(ctx | {name: tys1[n]}, ty, used))

                        return TmLet(name, tm1, tm2)
                    case _:
                        raise AssertionError("Impossible code path hit.")

            # D, y: AxB |- C -> D (Dictionary arrow)
            case TyProd(tys), TyDict(ty1, ty2):
                choice = draw(st.sampled_from(['elim', 'intro']))
                match choice:
                    case 'intro':
                        n = draw(st.integers(min_value=1, max_value=3))
                        ctx1, ctx2 = draw(split_ctx(ctx | {x: ty_neg}))
                        ty1_list = TyProd([ty1]*n)
                        ty2_list = TyProd([ty2]*n)
                        tm1 = draw(gen_tm(ctx1, ty1_list, used))
                        tm2 = draw(gen_tm(ctx2, ty2_list, used))
                        return TmDict(tm1, tm2)

                    case 'elim':
                        n = draw(st.integers(min_value=0, max_value=len(tys)-1))
                        name = draw(fresh(used))

                        tm1 = TmProj(n, TmVar(x))
                        tm2 = draw(gen_tm(ctx | {name: tys[n]}, ty, used))

                        return TmLet(name, tm1, tm2)
                    case _:
                        raise AssertionError("Impossible code path hit.")

            case _:
                raise AssertionError("Impossible code path hit.")

    else:
        raise AssertionError("Impossible code path hit.")

# --- Generating closed values ---

@st.composite
def gen_val(draw, ty: Ty, used: set[str]) -> Val:
    match ty:
        case TyUnit():
            return VUnit()
        case TySum(tys):
            n = draw(st.integers(min_value=0, max_value=len(tys)-1))
            v = draw(gen_val(tys[n], used))
            return VInj(n, v, ty)
        case TyProd(tys):
            tms = []
            for i in range(len(tys)):
                tms.append(draw(gen_tm(GCtx({},{}), tys[i], used)))
            return VProd(tms)
        case TyDict(ty1, ty2):
            n = draw(st.integers(min_value=1, max_value=3))
            ty1_list = TyProd([ty1]*n)
            ty2_list = TyProd([ty2]*n)
            v1 = draw(gen_val(ty1_list, used))
            v2 = draw(gen_val(ty2_list, used))
            return VDict(v1, v2)


# --- Helpers ---

@st.composite
def fresh(draw, used: set[str]) -> str:
    c = draw(st.sampled_from("abcdefghijklmnopqrstuvwxyz"))
    name = c + str(draw(st.integers(min_value=0, max_value=100000)))
    while name in used:
        name = c + str(draw(st.integers(min_value=0, max_value=100000)))
    used.add(name)
    return name

def gen_tm_var(ctx: GCtx) -> Gen[Tm]:
    x, _ = next(iter(ctx.flat().items()))
    return st.just(TmVar(x))

@st.composite
def split_ctx(draw, ctx: GCtx) -> tuple[GCtx, GCtx]:
   ctx_pos1, ctx_pos2 = draw(_split_ctx(ctx.pos))
   ctx_neg1, ctx_neg2 = draw(_split_ctx(ctx.neg))
   ctx1 = GCtx(pos=ctx_pos1, neg=ctx_neg1)
   ctx2 = GCtx(pos=ctx_pos2, neg=ctx_neg2)
   return ctx1, ctx2

@st.composite
def _split_ctx(draw, ctx: Ctx) -> tuple[Ctx, Ctx]:
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

def positive(ty: Ty) -> bool:
    match ty:
        case TyUnit():
            return True
        case TySum(_):
            return True
        case TyProd(_):
            return False
        case TyDict(_):
            return True
