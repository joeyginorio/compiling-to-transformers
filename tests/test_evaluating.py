from cajal.syntax import *
from cajal.evaluating import evaluate

# ============= `evaluate`: Unit Testing

# --- Atoms ---

def test_unit():
    assert evaluate(TmUnit(), {}) == [VUnit()]

def test_var():
    assert evaluate(TmVar('x'), {'x': VUnit()}) == [VUnit()]

# --- Pairs and Projections ---

def test_prod():
    assert evaluate(TmProd([TmUnit(), TmUnit()]), {}) == [VProd([TmUnit(), TmUnit()])]

def test_proj0():
    assert evaluate(TmProj(0, TmProd([TmUnit(), TmUnit()])), {}) == [VUnit()]

def test_proj1():
    assert evaluate(TmProj(1, TmProd([TmUnit(), TmUnit()])), {}) == [VUnit()]

def test_proj3():
    triple = TmProd([TmUnit(), TmUnit(), TmUnit()])
    assert evaluate(TmProj(2, triple), {}) == [VUnit()]

# --- Injections and Case ---

def test_inj0():
    ty = TySum([TyUnit(), TyUnit()])
    assert evaluate(TmInj(0, TmUnit(), ty), {}) == [VInj(0, VUnit(), ty)]

def test_inj1():
    ty = TySum([TyUnit(), TyUnit()])
    assert evaluate(TmInj(1, TmUnit(), ty), {}) == [VInj(1, VUnit(), ty)]

def test_inj2():
    ty = TySum([TyUnit(), TyUnit(), TyUnit()])
    assert evaluate(TmInj(2, TmUnit(), ty), {}) == [VInj(2, VUnit(), ty)]

def test_case_inj0():
    ty = TySum([TyUnit(), TyUnit()])
    tm = TmCase(TmInj(0, TmUnit(), ty), ['x', 'y'], [TmVar('x'), TmUnit()])
    assert evaluate(tm, {}) == [VUnit()]

def test_case_inj1():
    ty = TySum([TyUnit(), TyUnit()])
    tm = TmCase(TmInj(1, TmUnit(), ty), ['x', 'y'], [TmUnit(), TmVar('y')])
    assert evaluate(tm, {}) == [VUnit()]

# --- Sequencing ---

def test_seq():
    tm = TmSeq(TmUnit(), TmUnit())
    assert evaluate(tm, {}) == [VUnit()]

def test_seq_returns_second():
    ty = TySum([TyUnit(), TyUnit()])
    tm = TmSeq(TmUnit(), TmInj(0, TmUnit(), ty))
    assert evaluate(tm, {}) == [VInj(0, VUnit(), ty)]

def test_seq_with_env():
    tm = TmSeq(TmUnit(), TmVar('x'))
    assert evaluate(tm, {'x': VUnit()}) == [VUnit()]

# --- Error propagation ---

# A failed lookup: produces VError at runtime
_err = TmLookup(TmDict(TmProd([]), TmProd([])), TmUnit(), lambda x, y: True)

def test_inj_error():
    assert evaluate(TmInj(0, _err, TySum([TyUnit(), TyUnit()])), {}) == [VError()]

def test_case_error():
    tm = TmCase(_err, ['x', 'y'], [TmUnit(), TmUnit()])
    assert evaluate(tm, {}) == [VError()]

def test_seq_error_in_first():
    tm = TmSeq(_err, TmUnit())
    assert evaluate(tm, {}) == [VError()]

def test_seq_error_in_second():
    tm = TmSeq(TmUnit(), _err)
    assert evaluate(tm, {}) == [VError()]

# --- Choice ---

def test_choice():
    ty = TySum([TyUnit(), TyUnit()])
    tm = TmChoice(TmInj(0, TmUnit(), ty), TmInj(1, TmUnit(), ty))
    assert sorted(evaluate(tm, {}), key=repr) == sorted([VInj(0, VUnit(), ty), VInj(1, VUnit(), ty)], key=repr)

def test_choice_with_error():
    tm = TmChoice(TmUnit(), _err)
    assert sorted(evaluate(tm, {}), key=repr) == sorted([VUnit(), VError()], key=repr)

def test_choice_multiset():
    # Both branches return the same value — should appear twice
    tm = TmChoice(TmUnit(), TmUnit())
    assert evaluate(tm, {}) == [VUnit(), VUnit()]

def test_nested_choice():
    # TmChoice(TmChoice(a, b), c) should yield 3 results
    ty = TySum([TyUnit(), TyUnit(), TyUnit()])
    a, b, c = TmInj(0, TmUnit(), ty), TmInj(1, TmUnit(), ty), TmInj(2, TmUnit(), ty)
    results = evaluate(TmChoice(TmChoice(a, b), c), {})
    expected = [VInj(0, VUnit(), ty), VInj(1, VUnit(), ty), VInj(2, VUnit(), ty)]
    assert sorted(results, key=repr) == sorted(expected, key=repr)

# --- Let ---

def test_let():
    tm = TmLet('x', TmUnit(), TmVar('x'))
    assert evaluate(tm, {}) == [VUnit()]

def test_let_with_choice():
    # let x = (unit | unit) in x  — x is bound to each outcome, yielding [VUnit(), VUnit()]
    tm = TmLet('x', TmChoice(TmUnit(), TmUnit()), TmVar('x'))
    assert evaluate(tm, {}) == [VUnit(), VUnit()]

# --- Dict and Lookup ---

def test_dict():
    tm = TmDict(TmProd([TmUnit()]), TmProd([TmUnit()]))
    assert evaluate(tm, {}) == [VDict(VProd([TmUnit()]), VProd([TmUnit()]))]

def test_lookup():
    tm = TmLookup(TmDict(TmProd([TmUnit()]), TmProd([TmUnit()])), TmUnit(), lambda x, y: True)
    assert evaluate(tm, {}) == [VUnit()]

def test_error_propagation():
    assert evaluate(_err, {}) == [VError()]
