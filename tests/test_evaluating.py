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
    # proj_0 (tt, unit) = tt, verifying the first component is selected
    Bool = TySum([TyUnit(), TyUnit()])
    tt = TmInj(0, TmUnit(), Bool)
    assert evaluate(TmProj(0, TmProd([tt, TmUnit()])), {}) == [VInj(0, VUnit(), Bool)]

def test_proj1():
    # proj_1 (tt, unit) = unit, verifying the second component is selected
    Bool = TySum([TyUnit(), TyUnit()])
    tt = TmInj(0, TmUnit(), Bool)
    assert evaluate(TmProj(1, TmProd([tt, TmUnit()])), {}) == [VUnit()]

def test_proj2():
    # proj_2 of a triple selects the third component
    Bool = TySum([TyUnit(), TyUnit()])
    tt = TmInj(0, TmUnit(), Bool)
    triple = TmProd([TmUnit(), TmUnit(), tt])
    assert evaluate(TmProj(2, triple), {}) == [VInj(0, VUnit(), Bool)]

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

# --- Reverse ---

def test_reverse():
    n = 4
    ty = TySum([TyUnit()] * n)
    idx = lambda i: TmInj(i, TmUnit(), ty)

    reverse = TmProd([
        TmLookup(
            TmDict(TmProd([idx(i) for i in range(n)]),
                   TmProd([TmProj(i, TmVar("xs")) for i in range(n)])),
            idx(n-1-i),
            lambda v1, v2: v1 == v2
        ) for i in range(n)
    ])

    xs = TmProd([idx(3), idx(0), idx(1), idx(1)])  # [c,a,b,b] using idx as elem
    for i, e in enumerate([1, 1, 0, 3]):            # [b,b,a,c]
        assert evaluate(TmProj(i, TmLet("xs", xs, reverse)), {})[0] == VInj(e, VUnit(), ty)

# --- Counting sort ---

def test_counting_sort_trinary():
    # Counting sort on a 4-element lazy product of Fin3 = {0, 1, 2}.
    #
    # Algorithm: linear fold with a Fin15 accumulator encoding the pair (n0, n1),
    # where n0 = count of 0s seen and n1 = count of 1s seen so far
    # (n2 is implicit: n2 = elements_processed - n0 - n1).
    # A single accumulator is necessary because the additive product rule prevents
    # storing two independent counts derived from the same variables.
    #
    # Triangular state encoding — (n0, n1) pairs with n0+n1 <= 4:
    #   n0=0: states  0- 4  (n1 = 0,1,2,3,4)
    #   n0=1: states  5- 8  (n1 = 0,1,2,3)
    #   n0=2: states  9-11  (n1 = 0,1,2)
    #   n0=3: states 12-13  (n1 = 0,1)
    #   n0=4: state  14     (n1 = 0)
    #
    # Input (2, 0, 1, 2) -> sorted (0, 1, 2, 2).

    Elem  = TySum([TyUnit()] * 3)
    State = TySum([TyUnit()] * 15)

    def e(i):  return TmInj(i, TmUnit(), Elem)
    def st(i): return TmInj(i, TmUnit(), State)

    inp = TmProd([e(2), e(0), e(1), e(2)])

    def step(s_var, elem_tm):
        # e=0: n0 += 1; st(0) marks unreachable invalid transitions
        on_e0 = TmCase(TmVar(s_var), ['_'] * 15, [
            st(5),  st(6),  st(7),  st(8),  st(0),  # n0=0, n1=0..4
            st(9),  st(10), st(11), st(0),           # n0=1, n1=0..3
            st(12), st(13), st(0),                   # n0=2, n1=0..2
            st(14), st(0),                           # n0=3, n1=0..1
            st(0),                                   # n0=4 (unreachable)
        ])
        # e=1: n1 += 1
        on_e1 = TmCase(TmVar(s_var), ['_'] * 15, [
            st(1),  st(2),  st(3),  st(4),  st(0),  # n0=0, n1=0..4
            st(6),  st(7),  st(8),  st(0),           # n0=1, n1=0..3
            st(10), st(11), st(0),                   # n0=2, n1=0..2
            st(13), st(0),                           # n0=3, n1=0..1
            st(0),                                   # n0=4 (unreachable)
        ])
        # e=2: n2 += 1, so (n0, n1) unchanged
        on_e2 = TmVar(s_var)

        return TmCase(elem_tm, ['_', '_', '_'], [on_e0, on_e1, on_e2])

    def reconstruct(state_tm):
        z  = lambda: e(0)
        o  = lambda: e(1)
        tw = lambda: e(2)
        return TmCase(state_tm, ['_'] * 15, [
            TmProd([tw(), tw(), tw(), tw()]),  #  0: (n0=0,n1=0,n2=4)
            TmProd([o(),  tw(), tw(), tw()]),  #  1: (n0=0,n1=1,n2=3)
            TmProd([o(),  o(),  tw(), tw()]),  #  2: (n0=0,n1=2,n2=2)
            TmProd([o(),  o(),  o(),  tw()]),  #  3: (n0=0,n1=3,n2=1)
            TmProd([o(),  o(),  o(),  o() ]),  #  4: (n0=0,n1=4,n2=0)
            TmProd([z(),  tw(), tw(), tw()]),  #  5: (n0=1,n1=0,n2=3)
            TmProd([z(),  o(),  tw(), tw()]),  #  6: (n0=1,n1=1,n2=2)
            TmProd([z(),  o(),  o(),  tw()]),  #  7: (n0=1,n1=2,n2=1)
            TmProd([z(),  o(),  o(),  o() ]),  #  8: (n0=1,n1=3,n2=0)
            TmProd([z(),  z(),  tw(), tw()]),  #  9: (n0=2,n1=0,n2=2)
            TmProd([z(),  z(),  o(),  tw()]),  # 10: (n0=2,n1=1,n2=1)
            TmProd([z(),  z(),  o(),  o() ]),  # 11: (n0=2,n1=2,n2=0)
            TmProd([z(),  z(),  z(),  tw()]),  # 12: (n0=3,n1=0,n2=1)
            TmProd([z(),  z(),  z(),  o() ]),  # 13: (n0=3,n1=1,n2=0)
            TmProd([z(),  z(),  z(),  z() ]),  # 14: (n0=4,n1=0,n2=0)
        ])

    sort = TmLet('s0', st(0),
           TmLet('s1', step('s0', TmProj(0, inp)),
           TmLet('s2', step('s1', TmProj(1, inp)),
           TmLet('s3', step('s2', TmProj(2, inp)),
           TmLet('s4', step('s3', TmProj(3, inp)),
           reconstruct(TmVar('s4')))))))

    [prod] = evaluate(sort, {})   # exactly one result, no nondeterminism
    assert evaluate(TmProj(0, TmVar('p')), {'p': prod}) == [VInj(0, VUnit(), Elem)]  # 0
    assert evaluate(TmProj(1, TmVar('p')), {'p': prod}) == [VInj(1, VUnit(), Elem)]  # 1
    assert evaluate(TmProj(2, TmVar('p')), {'p': prod}) == [VInj(2, VUnit(), Elem)]  # 2
    assert evaluate(TmProj(3, TmVar('p')), {'p': prod}) == [VInj(2, VUnit(), Elem)]  # 2
