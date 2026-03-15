from cajal.syntax import *
from cajal.evaluating import evaluate


# --- Atoms ---

def test_unit():
    assert evaluate(TmUnit(), {}) == VUnit()

def test_var():
    assert evaluate(TmVar('x'), {'x': VUnit()}) == VUnit()

# --- Pairs and Projections ---

def test_pair():
    assert evaluate(TmPair(TmUnit(), TmUnit()), {}) == VPair(TmUnit(), TmUnit())

def test_proj1():
    assert evaluate(TmProj1(TmPair(TmUnit(), TmUnit())), {}) == VUnit()

def test_proj2():
    assert evaluate(TmProj2(TmPair(TmUnit(), TmUnit())), {}) == VUnit()

# --- Injections and Case ---

def test_inj1():
    assert evaluate(TmInj1(TmUnit(), TySum(TyUnit(), TyUnit())), {}) == VInj1(VUnit(), TySum(TyUnit(), TyUnit()))

def test_inj2():
    assert evaluate(TmInj2(TmUnit(), TySum(TyUnit(), TyUnit())), {}) == VInj2(VUnit(), TySum(TyUnit(), TyUnit()))

def test_case_inj1():
    tm = TmCase(TmInj1(TmUnit(), TySum(TyUnit(), TyUnit())), 'x', TmVar('x'), 'y', TmUnit())
    assert evaluate(tm, {}) == VUnit()

def test_case_inj2():
    tm = TmCase(TmInj2(TmUnit(), TySum(TyUnit(), TyUnit())), 'x', TmUnit(), 'y', TmVar('y'))
    assert evaluate(tm, {}) == VUnit()

# --- Error propagation ---

# A failed lookup: produces VError at runtime
_err = TmLookup(TmDict([], []), TmUnit(), lambda x, y: True)

def test_inj1_error():
    assert evaluate(TmInj1(_err, TySum(TyUnit(), TyUnit())), {}) == VError()

def test_case_error():
    tm = TmCase(_err, 'x', TmUnit(), 'y', TmUnit())
    assert evaluate(tm, {}) == VError()

# --- Choice ---

def test_choice():
    tm = TmChoice(TmUnit(), _err)
    assert evaluate(tm, {}) in [VUnit(), VError()]

# --- Let ---

def test_let():
    tm = TmLet('x', TmUnit(), TmVar('x'))
    assert evaluate(tm, {}) == VUnit()

# --- Dict and Lookup ---

def test_dict():
    tm = TmDict([TmUnit()], [TmUnit()])
    assert evaluate(tm, {}) == VDict([VUnit()], [VUnit()])

def test_lookup():
    tm = TmLookup(TmDict([TmUnit()], [TmUnit()]), TmUnit(), lambda x, y: True)
    assert evaluate(tm, {}) == VUnit()
