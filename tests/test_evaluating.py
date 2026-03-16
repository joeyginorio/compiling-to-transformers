from cajal.syntax import *
from cajal.evaluating import evaluate


# --- Atoms ---

def test_unit():
    assert evaluate(TmUnit(), {}) == VUnit()

def test_var():
    assert evaluate(TmVar('x'), {'x': VUnit()}) == VUnit()

# --- Pairs and Projections ---

def test_prod():
    assert evaluate(TmProd([TmUnit(), TmUnit()]), {}) == VProd([TmUnit(), TmUnit()])

def test_proj0():
    assert evaluate(TmProj(0, TmProd([TmUnit(), TmUnit()])), {}) == VUnit()

def test_proj1():
    assert evaluate(TmProj(1, TmProd([TmUnit(), TmUnit()])), {}) == VUnit()

def test_proj3():
    triple = TmProd([TmUnit(), TmUnit(), TmUnit()])
    assert evaluate(TmProj(2, triple), {}) == VUnit()

# --- Injections and Case ---

def test_inj0():
    ty = TySum([TyUnit(), TyUnit()])
    assert evaluate(TmInj(0, TmUnit(), ty), {}) == VInj(0, VUnit(), ty)

def test_inj1():
    ty = TySum([TyUnit(), TyUnit()])
    assert evaluate(TmInj(1, TmUnit(), ty), {}) == VInj(1, VUnit(), ty)

def test_inj2():
    ty = TySum([TyUnit(), TyUnit(), TyUnit()])
    assert evaluate(TmInj(2, TmUnit(), ty), {}) == VInj(2, VUnit(), ty)

def test_case_inj0():
    ty = TySum([TyUnit(), TyUnit()])
    tm = TmCase(TmInj(0, TmUnit(), ty), ['x', 'y'], [TmVar('x'), TmUnit()])
    assert evaluate(tm, {}) == VUnit()

def test_case_inj1():
    ty = TySum([TyUnit(), TyUnit()])
    tm = TmCase(TmInj(1, TmUnit(), ty), ['x', 'y'], [TmUnit(), TmVar('y')])
    assert evaluate(tm, {}) == VUnit()

# --- Error propagation ---

# A failed lookup: produces VError at runtime
_err = TmLookup(TmDict(TmProd([]), TmProd([])), TmUnit(), lambda x, y: True)

def test_inj_error():
    assert evaluate(TmInj(0, _err, TySum([TyUnit(), TyUnit()])), {}) == VError()

def test_case_error():
    tm = TmCase(_err, ['x', 'y'], [TmUnit(), TmUnit()])
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
    tm = TmDict(TmProd([TmUnit()]), TmProd([TmUnit()]))
    assert evaluate(tm, {}) == VDict(VProd([TmUnit()]), VProd([TmUnit()]))

def test_lookup():
    tm = TmLookup(TmDict(TmProd([TmUnit()]), TmProd([TmUnit()])), TmUnit(), lambda x, y: True)
    assert evaluate(tm, {}) == VUnit()
