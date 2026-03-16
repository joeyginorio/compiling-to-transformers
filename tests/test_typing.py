import pytest
from cajal.syntax import *
from cajal.typing import _check, Ctx


# --- Atoms ---

def test_unit():
    assert _check(TmUnit(), {}) == (TyUnit(), {})

def test_unit_preserves_ctx():
    assert _check(TmUnit(), {'x': TyUnit()}) == (TyUnit(), {'x': TyUnit()})

def test_var():
    assert _check(TmVar('x'), {'x': TyUnit()}) == (TyUnit(), {})

def test_var_consumes_binding():
    ty, ctx_remain = _check(TmVar('x'), {'x': TyUnit(), 'y': TyUnit()})
    assert ty == TyUnit()
    assert ctx_remain == {'y': TyUnit()}

def test_var_not_in_ctx():
    with pytest.raises(TypeError):
        _check(TmVar('x'), {})

# --- Injections ---

def test_inj0():
    ty = TySum([TyUnit(), TyUnit()])
    assert _check(TmInj(0, TmUnit(), ty), {}) == (ty, {})

def test_inj1():
    ty = TySum([TyUnit(), TyUnit()])
    assert _check(TmInj(1, TmUnit(), ty), {}) == (ty, {})

def test_inj2():
    ty = TySum([TyUnit(), TyUnit(), TyUnit()])
    assert _check(TmInj(2, TmUnit(), ty), {}) == (ty, {})

def test_inj_consumes_binding():
    ty = TySum([TyUnit(), TyUnit()])
    ctx: Ctx = {'x': TyUnit()}
    assert _check(TmInj(0, TmVar('x'), ty), ctx) == (ty, {})

def test_inj_type_mismatch():
    ty = TySum([TyProd([TyUnit()]), TyUnit()])
    with pytest.raises(TypeError):
        _check(TmInj(0, TmUnit(), ty), {})

def test_inj_not_sum_type():
    with pytest.raises(TypeError):
        _check(TmInj(0, TmUnit(), TyUnit()), {})

# --- Products ---

def test_prod():
    assert _check(TmProd([TmUnit(), TmUnit()]), {}) == (TyProd([TyUnit(), TyUnit()]), {})

def test_prod_single():
    assert _check(TmProd([TmUnit()]), {}) == (TyProd([TyUnit()]), {})

def test_prod_consumes_binding():
    ctx: Ctx = {'x': TyUnit()}
    assert _check(TmProd([TmVar('x'), TmVar('x')]), ctx) == (TyProd([TyUnit(), TyUnit()]), {})

def test_prod_non_uniform_ctx():
    ctx: Ctx = {'x': TyUnit(), 'y': TyUnit()}
    with pytest.raises(TypeError):
        _check(TmProd([TmVar('x'), TmVar('y')]), ctx)

# --- Dictionaries ---

_enum2 = TySum([TyUnit(), TyUnit()])

def test_dict():
    keys = TmProd([TmInj(0, TmUnit(), _enum2), TmInj(1, TmUnit(), _enum2)])
    vals = TmProd([TmUnit(), TmUnit()])
    assert _check(TmDict(keys, vals), {}) == (TyDict(_enum2, TyUnit()), {})

def test_dict_consumes_ctx():
    ctx: Ctx = {'x': TyUnit()}
    keys = TmProd([TmInj(0, TmVar('x'), _enum2), TmInj(0, TmVar('x'), _enum2)])
    vals = TmProd([TmUnit(), TmUnit()])
    ty, ctx_remain = _check(TmDict(keys, vals), ctx)
    assert ty == TyDict(_enum2, TyUnit())
    assert ctx_remain == {}

def test_dict_keys_not_product():
    with pytest.raises(TypeError):
        _check(TmDict(TmUnit(), TmProd([TmUnit()])), {})

def test_dict_vals_not_product():
    with pytest.raises(TypeError):
        _check(TmDict(TmProd([TmUnit()]), TmUnit()), {})

def test_dict_length_mismatch():
    keys = TmProd([TmInj(0, TmUnit(), _enum2)])
    vals = TmProd([TmUnit(), TmUnit()])
    with pytest.raises(TypeError):
        _check(TmDict(keys, vals), {})

def test_dict_keys_not_enum():
    keys = TmProd([TmUnit(), TmUnit()])
    vals = TmProd([TmUnit(), TmUnit()])
    with pytest.raises(TypeError):
        _check(TmDict(keys, vals), {})

# --- Case ---

_bool = TySum([TyUnit(), TyUnit()])

def test_case():
    tm = TmCase(TmInj(0, TmUnit(), _bool), ['x', 'y'], [TmVar('x'), TmVar('y')])
    assert _check(tm, {}) == (TyUnit(), {})

def test_case_consumes_scrutinee_ctx():
    ctx: Ctx = {'z': _bool}
    tm = TmCase(TmVar('z'), ['x', 'y'], [TmVar('x'), TmVar('y')])
    ty, ctx_remain = _check(tm, ctx)
    assert ty == TyUnit()
    assert ctx_remain == {}

def test_case_preserves_unused_ctx():
    ctx: Ctx = {'w': TyUnit()}
    tm = TmCase(TmInj(0, TmUnit(), _bool), ['x', 'y'], [TmVar('x'), TmVar('y')])
    ty, ctx_remain = _check(tm, ctx)
    assert ty == TyUnit()
    assert ctx_remain == {'w': TyUnit()}

def test_case_not_sum_type():
    with pytest.raises(TypeError):
        _check(TmCase(TmUnit(), ['x'], [TmUnit()]), {})

def test_case_branches_different_types():
    ty3 = TySum([TyUnit(), TyUnit(), TyUnit()])
    tm = TmCase(
        TmInj(0, TmUnit(), ty3),
        ['x', 'y', 'z'],
        [TmUnit(), TmUnit(), TmProd([TmUnit()])]
    )
    with pytest.raises(TypeError):
        _check(tm, {})

def test_case_branches_non_uniform_ctx():
    ctx: Ctx = {'a': TyUnit()}
    tm = TmCase(
        TmInj(0, TmUnit(), _bool),
        ['x', 'y'],
        [TmVar('a'), TmUnit()]
    )
    with pytest.raises(TypeError):
        _check(tm, ctx)

# --- Projections ---

def test_proj_fst():
    tm = TmProj(0, TmProd([TmUnit(), TmProd([TmUnit()])]))
    assert _check(tm, {}) == (TyUnit(), {})

def test_proj_snd():
    tm = TmProj(1, TmProd([TmUnit(), TmProd([TmUnit()])]))
    assert _check(tm, {}) == (TyProd([TyUnit()]), {})

def test_proj_consumes_ctx():
    ctx: Ctx = {'x': TyProd([TyUnit(), TyUnit()])}
    tm = TmProj(0, TmVar('x'))
    ty, ctx_remain = _check(tm, ctx)
    assert ty == TyUnit()
    assert ctx_remain == {}

def test_proj_preserves_unused_ctx():
    ctx: Ctx = {'x': TyProd([TyUnit(), TyUnit()]), 'y': TyUnit()}
    tm = TmProj(0, TmVar('x'))
    ty, ctx_remain = _check(tm, ctx)
    assert ty == TyUnit()
    assert ctx_remain == {'y': TyUnit()}

def test_proj_not_product_type():
    with pytest.raises(TypeError):
        _check(TmProj(0, TmUnit()), {})

# --- Choice ---

def test_choice():
    tm = TmChoice(TmUnit(), TmUnit())
    assert _check(tm, {}) == (TyUnit(), {})

def test_choice_consumes_ctx():
    ctx: Ctx = {'x': TyUnit()}
    tm = TmChoice(TmVar('x'), TmVar('x'))
    ty, ctx_remain = _check(tm, ctx)
    assert ty == TyUnit()
    assert ctx_remain == {}

def test_choice_preserves_unused_ctx():
    ctx: Ctx = {'x': TyUnit()}
    tm = TmChoice(TmUnit(), TmUnit())
    assert _check(tm, ctx) == (TyUnit(), {'x': TyUnit()})

def test_choice_type_mismatch():
    with pytest.raises(TypeError):
        _check(TmChoice(TmUnit(), TmProd([TmUnit()])), {})

def test_choice_non_uniform_ctx():
    ctx: Ctx = {'x': TyUnit()}
    with pytest.raises(TypeError):
        _check(TmChoice(TmVar('x'), TmUnit()), ctx)

# --- Lookup ---

def _make_dict(key_ty, val_tm_pairs):
    """Helper: build a TmDict from a key type and list of value terms."""
    n = len(val_tm_pairs)
    keys = TmProd([TmInj(i, TmUnit(), key_ty) for i in range(n)])
    vals = TmProd(val_tm_pairs)
    return TmDict(keys, vals)

def test_lookup():
    d = _make_dict(_enum2, [TmUnit(), TmUnit()])
    q = TmInj(0, TmUnit(), _enum2)
    tm = TmLookup(d, q, lambda a, b: a == b)
    assert _check(tm, {}) == (TyUnit(), {})

def test_lookup_consumes_ctx():
    ctx: Ctx = {'x': TyUnit()}
    d = _make_dict(_enum2, [TmVar('x'), TmVar('x')])
    q = TmInj(0, TmUnit(), _enum2)
    tm = TmLookup(d, q, lambda a, b: a == b)
    ty, ctx_remain = _check(tm, ctx)
    assert ty == TyUnit()
    assert ctx_remain == {}

def test_lookup_not_dict_type():
    q = TmInj(0, TmUnit(), _enum2)
    with pytest.raises(TypeError):
        _check(TmLookup(TmUnit(), q, lambda a, b: a == b), {})

def test_lookup_query_not_enum():
    d = _make_dict(_enum2, [TmUnit(), TmUnit()])
    with pytest.raises(TypeError):
        _check(TmLookup(d, TmUnit(), lambda a, b: a == b), {})
