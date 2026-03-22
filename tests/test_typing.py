import pytest
from cajal.syntax import *
from cajal.typing import check, Ctx


# ============= `check`: Property-based Testing

# TODO

# ============= `check`: Unit Testing

# --- Atoms ---

def test_unit():
    assert check(TmUnit(), {}) == TyUnit()

def test_unit_unused_ctx():
    with pytest.raises(TypeError):
        check(TmUnit(), {'x': TyUnit()})

def test_var():
    assert check(TmVar('x'), {'x': TyUnit()}) == TyUnit()

def test_var_not_in_ctx():
    with pytest.raises(TypeError):
        check(TmVar('x'), {})

def test_var_unused_ctx():
    with pytest.raises(TypeError):
        check(TmVar('x'), {'x': TyUnit(), 'y': TyUnit()})

# --- Injections ---

_enum2 = TySum([TyUnit(), TyUnit()])

def test_inj():
    assert check(TmInj(0, TmUnit(), _enum2), {}) == _enum2

def test_inj_different_indices():
    ty3 = TySum([TyUnit(), TyUnit(), TyUnit()])
    assert check(TmInj(1, TmUnit(), _enum2), {}) == _enum2
    assert check(TmInj(2, TmUnit(), ty3), {}) == ty3

def test_inj_consumes_ctx():
    assert check(TmInj(0, TmVar('x'), _enum2), {'x': TyUnit()}) == _enum2

def test_inj_unused_ctx():
    with pytest.raises(TypeError):
        check(TmInj(0, TmUnit(), _enum2), {'x': TyUnit()})

def test_inj_type_mismatch():
    ty = TySum([TyProd([TyUnit()]), TyUnit()])
    with pytest.raises(TypeError):
        check(TmInj(0, TmUnit(), ty), {})

def test_inj_not_sum_type():
    with pytest.raises(TypeError):
        check(TmInj(0, TmUnit(), TyUnit()), {})

# --- Products ---

def test_prod():
    assert check(TmProd([TmUnit(), TmUnit()]), {}) == TyProd([TyUnit(), TyUnit()])

def test_prod_single():
    assert check(TmProd([TmUnit()]), {}) == TyProd([TyUnit()])

def test_prod_consumes_ctx():
    assert check(TmProd([TmVar('x'), TmVar('x')]), {'x': TyUnit()}) == TyProd([TyUnit(), TyUnit()])

def test_prod_unused_ctx():
    with pytest.raises(TypeError):
        check(TmProd([TmUnit(), TmUnit()]), {'x': TyUnit()})

def test_prod_non_uniform_ctx():
    with pytest.raises(TypeError):
        check(TmProd([TmVar('x'), TmVar('y')]), {'x': TyUnit(), 'y': TyUnit()})

# --- Dictionaries ---

def test_dict():
    keys = TmProd([TmInj(0, TmUnit(), _enum2), TmInj(1, TmUnit(), _enum2)])
    vals = TmProd([TmUnit(), TmUnit()])
    assert check(TmDict(keys, vals), {}) == TyDict(_enum2, TyUnit())

def test_dict_consumes_ctx():
    keys = TmProd([TmInj(0, TmVar('x'), _enum2), TmInj(0, TmVar('x'), _enum2)])
    vals = TmProd([TmUnit(), TmUnit()])
    assert check(TmDict(keys, vals), {'x': TyUnit()}) == TyDict(_enum2, TyUnit())

def test_dict_unused_ctx():
    keys = TmProd([TmInj(0, TmUnit(), _enum2), TmInj(1, TmUnit(), _enum2)])
    vals = TmProd([TmUnit(), TmUnit()])
    with pytest.raises(TypeError):
        check(TmDict(keys, vals), {'x': TyUnit()})

def test_dict_keys_not_product():
    with pytest.raises(TypeError):
        check(TmDict(TmUnit(), TmProd([TmUnit()])), {})

def test_dict_vals_not_product():
    with pytest.raises(TypeError):
        check(TmDict(TmProd([TmUnit()]), TmUnit()), {})

def test_dict_length_mismatch():
    keys = TmProd([TmInj(0, TmUnit(), _enum2)])
    vals = TmProd([TmUnit(), TmUnit()])
    with pytest.raises(TypeError):
        check(TmDict(keys, vals), {})

def test_dict_keys_not_enum():
    keys = TmProd([TmUnit(), TmUnit()])
    vals = TmProd([TmUnit(), TmUnit()])
    with pytest.raises(TypeError):
        check(TmDict(keys, vals), {})

# --- Sequencing ---

def test_seq():
    assert check(TmSeq(TmUnit(), TmUnit()), {}) == TyUnit()

def test_seq_returns_second_type():
    assert check(TmSeq(TmUnit(), TmProd([TmUnit()])), {}) == TyProd([TyUnit()])

def test_seq_consumes_ctx():
    assert check(TmSeq(TmVar('x'), TmVar('y')), {'x': TyUnit(), 'y': TyUnit()}) == TyUnit()

def test_seq_unused_ctx():
    with pytest.raises(TypeError):
        check(TmSeq(TmUnit(), TmUnit()), {'x': TyUnit()})

def test_seq_first_not_unit():
    with pytest.raises(TypeError):
        check(TmSeq(TmProd([TmUnit()]), TmUnit()), {})

# --- Let ---

def test_let():
    assert check(TmLet('x', TmUnit(), TmVar('x')), {}) == TyUnit()

def test_let_consumes_ctx():
    assert check(TmLet('x', TmVar('y'), TmVar('x')), {'y': TyUnit()}) == TyUnit()

def test_let_unused_ctx():
    with pytest.raises(TypeError):
        check(TmLet('x', TmUnit(), TmVar('x')), {'z': TyUnit()})

# --- Case ---

_bool = TySum([TyUnit(), TyUnit()])

def test_case():
    tm = TmCase(TmInj(0, TmUnit(), _bool), ['x', 'y'], [TmVar('x'), TmVar('y')])
    assert check(tm, {}) == TyUnit()

def test_case_consumes_ctx():
    tm = TmCase(TmVar('z'), ['x', 'y'], [TmVar('x'), TmVar('y')])
    assert check(tm, {'z': _bool}) == TyUnit()

def test_case_unused_ctx():
    tm = TmCase(TmInj(0, TmUnit(), _bool), ['x', 'y'], [TmVar('x'), TmVar('y')])
    with pytest.raises(TypeError):
        check(tm, {'w': TyUnit()})

def test_case_not_sum_type():
    with pytest.raises(TypeError):
        check(TmCase(TmUnit(), ['x'], [TmUnit()]), {})

def test_case_branches_different_types():
    ty3 = TySum([TyUnit(), TyUnit(), TyUnit()])
    tm = TmCase(
        TmInj(0, TmUnit(), ty3),
        ['x', 'y', 'z'],
        [TmUnit(), TmUnit(), TmProd([TmUnit()])]
    )
    with pytest.raises(TypeError):
        check(tm, {})

def test_case_branches_non_uniform_ctx():
    ctx: Ctx = {'a': TyUnit()}
    tm = TmCase(
        TmInj(0, TmUnit(), _bool),
        ['x', 'y'],
        [TmVar('a'), TmUnit()]
    )
    with pytest.raises(TypeError):
        check(tm, ctx)

# --- Projections ---

def test_proj_fst():
    tm = TmProj(0, TmProd([TmUnit(), TmProd([TmUnit()])]))
    assert check(tm, {}) == TyUnit()

def test_proj_snd():
    tm = TmProj(1, TmProd([TmUnit(), TmProd([TmUnit()])]))
    assert check(tm, {}) == TyProd([TyUnit()])

def test_proj_consumes_ctx():
    tm = TmProj(0, TmVar('x'))
    assert check(tm, {'x': TyProd([TyUnit(), TyUnit()])}) == TyUnit()

def test_proj_unused_ctx():
    tm = TmProj(0, TmProd([TmUnit(), TmUnit()]))
    with pytest.raises(TypeError):
        check(tm, {'y': TyUnit()})

def test_proj_not_product_type():
    with pytest.raises(TypeError):
        check(TmProj(0, TmUnit()), {})

# --- Choice ---

def test_choice():
    assert check(TmChoice(TmUnit(), TmUnit()), {}) == TyUnit()

def test_choice_consumes_ctx():
    assert check(TmChoice(TmVar('x'), TmVar('x')), {'x': TyUnit()}) == TyUnit()

def test_choice_unused_ctx():
    with pytest.raises(TypeError):
        check(TmChoice(TmUnit(), TmUnit()), {'x': TyUnit()})

def test_choice_type_mismatch():
    with pytest.raises(TypeError):
        check(TmChoice(TmUnit(), TmProd([TmUnit()])), {})

def test_choice_non_uniform_ctx():
    with pytest.raises(TypeError):
        check(TmChoice(TmVar('x'), TmUnit()), {'x': TyUnit()})

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
    assert check(TmLookup(d, q, lambda a, b: a == b), {}) == TyUnit()

def test_lookup_consumes_ctx():
    d = _make_dict(_enum2, [TmVar('x'), TmVar('x')])
    q = TmInj(0, TmUnit(), _enum2)
    assert check(TmLookup(d, q, lambda a, b: a == b), {'x': TyUnit()}) == TyUnit()

def test_lookup_unused_ctx():
    d = _make_dict(_enum2, [TmUnit(), TmUnit()])
    q = TmInj(0, TmUnit(), _enum2)
    with pytest.raises(TypeError):
        check(TmLookup(d, q, lambda a, b: a == b), {'x': TyUnit()})

def test_lookup_not_dict_type():
    q = TmInj(0, TmUnit(), _enum2)
    with pytest.raises(TypeError):
        check(TmLookup(TmUnit(), q, lambda a, b: a == b), {})

def test_lookup_query_not_enum():
    d = _make_dict(_enum2, [TmUnit(), TmUnit()])
    with pytest.raises(TypeError):
        check(TmLookup(d, TmUnit(), lambda a, b: a == b), {})
