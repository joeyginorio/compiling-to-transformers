import torch
import hypothesis.strategies as st
from cajal.syntax import *
from cajal.compiling import compile, compile_val, dim, zero
from cajal.evaluating import evaluate
from cajal.typing import _check
from hypothesis import given, assume, settings, HealthCheck
from strategies import gen_prog, gen_val                             


# ============= `compile`: Property-Based Testing

@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(gen_prog(), st.data())
def test_compiler_correctness(prog, data):
    ctx, tm, _ = prog
    _check(tm, ctx.flat())
    env = {x: data.draw(gen_val(ty_x, set())) for x, ty_x in ctx.flat().items()}
    vs = evaluate(tm, env)

    env_compiled = {x: compile_val(v)({}) for x, v in env.items()}
    tm_compiled = compile(tm)(env_compiled)
    vs_compiled = [compile_val(v)(env_compiled) for v in vs]
    vs_compiled_sum = torch.stack(vs_compiled).sum(dim=0)
    assert torch.equal(tm_compiled, vs_compiled_sum)

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(gen_prog(), st.data())
def test_compiler_propagates_gradients(prog, data):
    ctx, tm, _ = prog
    _check(tm, ctx.flat())
    env = {x: data.draw(gen_val(ty_x, set())) for x, ty_x in ctx.flat().items()}
    
    assume(ctx.flat())  # require at least one variable

    env_compiled = {x: compile_val(v)({}).detach().requires_grad_() for x, v in env.items()}
    y = compile(tm)(env_compiled).sum()
    y.backward()
    assert any(v.grad is not None for v in env_compiled.values())



# ============= `compile`: Unit Testing

# --- TmUnit ---

def test_compile_unit():
    module = compile(TmUnit())
    result = module({})
    assert torch.equal(result, torch.tensor([1.0]))


# --- TmVar ---

def test_compile_var():
    module = compile(TmVar('x'))
    result = module({'x': torch.tensor([1.0])})
    assert torch.equal(result, torch.tensor([1.0]))

def test_compile_var_selects_correct_binding():
    module = compile(TmVar('y'))
    env = {'x': torch.tensor([1.0]), 'y': torch.tensor([2.0])}
    result = module(env)
    assert torch.equal(result, torch.tensor([2.0]))


# --- TmProd ---

def test_compile_prod_pair():
    module = compile(TmProd([TmUnit(), TmUnit()]))
    result = module({})
    assert torch.equal(result, torch.tensor([1.0, 1.0]))

def test_compile_prod_triple():
    module = compile(TmProd([TmUnit(), TmUnit(), TmUnit()]))
    result = module({})
    assert torch.equal(result, torch.tensor([1.0, 1.0, 1.0]))

def test_compile_prod_from_vars():
    x = torch.tensor([2.0])
    y = torch.tensor([3.0])
    module = compile(TmProd([TmVar('x'), TmVar('y')]))
    result = module({'x': x, 'y': y})
    assert torch.equal(result, torch.tensor([2.0, 3.0]))


# --- TmInj ---

def test_compile_inj0_binary():
    # inj_0 unit : Unit + Unit  ->  (1.0, 0.0)
    ty = TySum([TyUnit(), TyUnit()])
    module = compile(TmInj(0, TmUnit(), ty))
    result = module({})
    assert torch.equal(result, torch.tensor([1.0, 0.0]))

def test_compile_inj1_binary():
    # inj_1 unit : Unit + Unit  ->  (0.0, 1.0)
    ty = TySum([TyUnit(), TyUnit()])
    module = compile(TmInj(1, TmUnit(), ty))
    result = module({})
    assert torch.equal(result, torch.tensor([0.0, 1.0]))

def test_compile_inj0_ternary():
    # inj_0 unit : Unit + Unit + Unit  ->  (1.0, 0.0, 0.0)
    ty = TySum([TyUnit(), TyUnit(), TyUnit()])
    module = compile(TmInj(0, TmUnit(), ty))
    result = module({})
    assert torch.equal(result, torch.tensor([1.0, 0.0, 0.0]))

def test_compile_inj1_ternary():
    # inj_1 unit : Unit + Unit + Unit  ->  (0.0, 1.0, 0.0)
    ty = TySum([TyUnit(), TyUnit(), TyUnit()])
    module = compile(TmInj(1, TmUnit(), ty))
    result = module({})
    assert torch.equal(result, torch.tensor([0.0, 1.0, 0.0]))

def test_compile_inj2_ternary():
    # inj_2 unit : Unit + Unit + Unit  ->  (0.0, 0.0, 1.0)
    ty = TySum([TyUnit(), TyUnit(), TyUnit()])
    module = compile(TmInj(2, TmUnit(), ty))
    result = module({})
    assert torch.equal(result, torch.tensor([0.0, 0.0, 1.0]))


# --- TmLet ---

def test_compile_let_basic():
    # let x = unit in x
    module = compile(TmLet('x', TmUnit(), TmVar('x')))
    result = module({})
    assert torch.equal(result, torch.tensor([1.0]))

def test_compile_let_binding_visible_in_body():
    # let x = unit in (x, x)  - x bound to [1.0], body produces [1.0, 1.0]
    module = compile(TmLet('x', TmUnit(), TmProd([TmVar('x'), TmVar('x')])))
    result = module({})
    assert torch.equal(result, torch.tensor([1.0, 1.0]))

def test_compile_let_outer_env_visible_in_body():
    # let x = unit in y, with y=[2.0] in outer env
    module = compile(TmLet('x', TmUnit(), TmVar('y')))
    result = module({'y': torch.tensor([2.0])})
    assert torch.equal(result, torch.tensor([2.0]))

def test_compile_let_nested():
    # let x = unit in let y = unit in (x, y)
    inner = TmLet('y', TmUnit(), TmProd([TmVar('x'), TmVar('y')]))
    module = compile(TmLet('x', TmUnit(), inner))
    result = module({})
    assert torch.equal(result, torch.tensor([1.0, 1.0]))


# --- TmCase ---

def test_compile_case_binary_inj0_identity_branches():
    # case x of { x0 -> x0 | x1 -> x1 } with x = inj_0 unit = [1.0, 0.0]
    # branch 0 contributes [1.0], branch 1 contributes [0.0]; sum = [1.0]
    scrutinee = TmVar('x')
    scrutinee.ty_checked = TySum([TyUnit(), TyUnit()])
    tm_case = TmCase(scrutinee, ['x0', 'x1'], [TmVar('x0'), TmVar('x1')])
    module = compile(tm_case)
    result = module({'x': torch.tensor([1.0, 0.0])})
    assert torch.equal(result, torch.tensor([1.0]))


def test_compile_case_binary_inj1_identity_branches():
    # case x of { x0 -> x0 | x1 -> x1 } with x = inj_1 unit = [0.0, 1.0]
    # branch 0 contributes [0.0], branch 1 contributes [1.0]; sum = [1.0]
    scrutinee = TmVar('x')
    scrutinee.ty_checked = TySum([TyUnit(), TyUnit()])
    tm_case = TmCase(scrutinee, ['x0', 'x1'], [TmVar('x0'), TmVar('x1')])
    module = compile(tm_case)
    result = module({'x': torch.tensor([0.0, 1.0])})
    assert torch.equal(result, torch.tensor([1.0]))


def test_compile_case_ternary_inj0():
    # case x of { x0->x0 | x1->x1 | x2->x2 } with x = inj_0 unit = [1,0,0]
    scrutinee = TmVar('x')
    scrutinee.ty_checked = TySum([TyUnit(), TyUnit(), TyUnit()])
    xs = ['x0', 'x1', 'x2']
    tm_case = TmCase(scrutinee, xs, [TmVar(xi) for xi in xs])
    module = compile(tm_case)
    result = module({'x': torch.tensor([1.0, 0.0, 0.0])})
    assert torch.equal(result, torch.tensor([1.0]))


def test_compile_case_ternary_inj1():
    scrutinee = TmVar('x')
    scrutinee.ty_checked = TySum([TyUnit(), TyUnit(), TyUnit()])
    xs = ['x0', 'x1', 'x2']
    tm_case = TmCase(scrutinee, xs, [TmVar(xi) for xi in xs])
    module = compile(tm_case)
    result = module({'x': torch.tensor([0.0, 1.0, 0.0])})
    assert torch.equal(result, torch.tensor([1.0]))


def test_compile_case_ternary_inj2():
    scrutinee = TmVar('x')
    scrutinee.ty_checked = TySum([TyUnit(), TyUnit(), TyUnit()])
    xs = ['x0', 'x1', 'x2']
    tm_case = TmCase(scrutinee, xs, [TmVar(xi) for xi in xs])
    module = compile(tm_case)
    result = module({'x': torch.tensor([0.0, 0.0, 1.0])})
    assert torch.equal(result, torch.tensor([1.0]))


def test_compile_case_raises_on_non_sum_ty_checked():
    # If tm.ty_checked is not a TySum, compile should raise TypeError
    tm_var = TmVar('x')
    tm_var.ty_checked = TyUnit()  # wrong: not a sum type
    tm_case = TmCase(tm_var, ['x0'], [TmVar('x0')])
    import pytest
    with pytest.raises(TypeError):
        compile(tm_case)


# --- compile: if(x) then(ff) else(tt) via sum types ---
#
# Boolean encoding:
#   Bool = TySum([TyUnit(), TyUnit()])
#   tt   = inj_0 unit : Bool  ->  [1.0, 0.0]
#   ff   = inj_1 unit : Bool  ->  [0.0, 1.0]
#
# if(x) then(ff) else(tt) is boolean NOT, encoded as:
#   case x of { x0 -> inj_1 x0 : Bool | x1 -> inj_0 x1 : Bool }
#
# NnCase evaluates all branches and sums the results:
#   x = tt = [1, 0]:  branch 0 -> inj_1([1]) = [0,1];  branch 1 -> inj_0([0]) = [0,0];  sum = [0,1] = ff
#   x = ff = [0, 1]:  branch 0 -> inj_1([0]) = [0,0];  branch 1 -> inj_0([1]) = [1,0];  sum = [1,0] = tt

def test_compile_if_tt_then_ff_else_tt():
    # if tt then ff else tt  =  ff
    Bool = TySum([TyUnit(), TyUnit()])
    scrutinee = TmVar('x')
    scrutinee.ty_checked = Bool
    not_x = TmCase(scrutinee, ['x0', 'x1'], [TmInj(1, TmVar('x0'), Bool), TmInj(0, TmVar('x1'), Bool)])
    module = compile(not_x)
    result = module({'x': torch.tensor([1.0, 0.0])})  # tt
    assert torch.equal(result, torch.tensor([0.0, 1.0]))  # ff


def test_compile_if_ff_then_ff_else_tt():
    # if ff then ff else tt  =  tt
    Bool = TySum([TyUnit(), TyUnit()])
    scrutinee = TmVar('x')
    scrutinee.ty_checked = Bool
    not_x = TmCase(scrutinee, ['x0', 'x1'], [TmInj(1, TmVar('x0'), Bool), TmInj(0, TmVar('x1'), Bool)])
    module = compile(not_x)
    result = module({'x': torch.tensor([0.0, 1.0])})  # ff
    assert torch.equal(result, torch.tensor([1.0, 0.0]))  # tt


# --- TmSeq ---

def test_compile_seq_unit_unit():
    # unit; unit = [1.0] * [1.0] = [1.0]
    module = compile(TmSeq(TmUnit(), TmUnit()))
    result = module({})
    assert torch.equal(result, torch.tensor([1.0]))

def test_compile_seq_second_is_inj():
    # unit; inj_0 unit : Bool = [1.0] * [1.0, 0.0] = [1.0, 0.0]
    Bool = TySum([TyUnit(), TyUnit()])
    module = compile(TmSeq(TmUnit(), TmInj(0, TmUnit(), Bool)))
    result = module({})
    assert torch.equal(result, torch.tensor([1.0, 0.0]))

def test_compile_seq_scales_second():
    # x; y with x=[2.0], y=[3.0] = [6.0]
    module = compile(TmSeq(TmVar('x'), TmVar('y')))
    result = module({'x': torch.tensor([2.0]), 'y': torch.tensor([3.0])})
    assert torch.equal(result, torch.tensor([6.0]))


# --- TmProj ---

def test_compile_proj0():
    # proj_0 (unit, inj_0 unit) = [1.0]
    Bool = TySum([TyUnit(), TyUnit()])
    inner = TmProd([TmUnit(), TmInj(0, TmUnit(), Bool)])
    inner.ty_checked = TyProd([TyUnit(), Bool])
    module = compile(TmProj(0, inner))
    result = module({})
    assert torch.equal(result, torch.tensor([1.0]))

def test_compile_proj1():
    # proj_1 (unit, inj_0 unit) = [1.0, 0.0]
    Bool = TySum([TyUnit(), TyUnit()])
    inner = TmProd([TmUnit(), TmInj(0, TmUnit(), Bool)])
    inner.ty_checked = TyProd([TyUnit(), Bool])
    module = compile(TmProj(1, inner))
    result = module({})
    assert torch.equal(result, torch.tensor([1.0, 0.0]))

def test_compile_proj_raises_on_non_prod_ty_checked():
    inner = TmVar('x')
    inner.ty_checked = TyUnit()
    import pytest
    with pytest.raises(TypeError):
        compile(TmProj(0, inner))


# --- Compiler Correctness ---

def test_correctness_var():
    tm = TmVar('x')
    val_env: dict[str, Val] = {'x': VUnit()}
    tensor_env = {'x': torch.tensor([1.0])}
    values = evaluate(tm, val_env)
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)(tensor_env), expected)


def test_correctness_prod():
    tm = TmProd([TmUnit(), TmUnit()])
    values = evaluate(tm, {})
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)({}), expected)


def test_correctness_prod_triple():
    tm = TmProd([TmUnit(), TmUnit(), TmUnit()])
    values = evaluate(tm, {})
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)({}), expected)


def test_correctness_prod_open_unit():
    # TmProd([TmVar('x'), TmVar('x')]) with x : Unit
    # VProd stores the original terms lazily, so compile_val needs tensor_env
    tm = TmProd([TmVar('x'), TmVar('x')])
    val_env: dict[str, Val] = {'x': VUnit()}
    tensor_env = {'x': torch.tensor([1.0])}
    values = evaluate(tm, val_env)
    expected = torch.stack([compile_val(v)(tensor_env) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)(tensor_env), expected)


def test_correctness_prod_open_bool():
    # TmProd([TmVar('x'), TmVar('x')]) with x : Bool = tt
    Bool = TySum([TyUnit(), TyUnit()])
    tm = TmProd([TmVar('x'), TmVar('x')])
    val_env: dict[str, Val] = {'x': VInj(0, VUnit(), Bool)}
    tensor_env = {'x': torch.tensor([1.0, 0.0])}
    values = evaluate(tm, val_env)
    expected = torch.stack([compile_val(v)(tensor_env) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)(tensor_env), expected)


def test_correctness_proj0_closed():
    # proj_0 (unit, unit) = unit
    inner = TmProd([TmUnit(), TmUnit()])
    inner.ty_checked = TyProd([TyUnit(), TyUnit()])
    tm = TmProj(0, inner)
    values = evaluate(tm, {})
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)({}), expected)


def test_correctness_proj1_closed():
    # proj_1 (unit, unit) = unit
    inner = TmProd([TmUnit(), TmUnit()])
    inner.ty_checked = TyProd([TyUnit(), TyUnit()])
    tm = TmProj(1, inner)
    values = evaluate(tm, {})
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)({}), expected)


def test_correctness_proj0_wider():
    # proj_0 (tt, unit) = tt, where tt : Bool, unit : Unit
    Bool = TySum([TyUnit(), TyUnit()])
    inner = TmProd([TmInj(0, TmUnit(), Bool), TmUnit()])
    inner.ty_checked = TyProd([Bool, TyUnit()])
    tm = TmProj(0, inner)
    values = evaluate(tm, {})
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)({}), expected)


def test_correctness_proj1_wider():
    # proj_1 (tt, unit) = unit
    Bool = TySum([TyUnit(), TyUnit()])
    inner = TmProd([TmInj(0, TmUnit(), Bool), TmUnit()])
    inner.ty_checked = TyProd([Bool, TyUnit()])
    tm = TmProj(1, inner)
    values = evaluate(tm, {})
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)({}), expected)


def test_correctness_proj0_open():
    # proj_0 x where x : (Unit, Bool) = (unit, tt)
    Bool = TySum([TyUnit(), TyUnit()])
    inner = TmVar('x')
    inner.ty_checked = TyProd([TyUnit(), Bool])
    tm = TmProj(0, inner)
    val_env: dict[str, Val] = {'x': VProd([TmUnit(), TmInj(0, TmUnit(), Bool)])}
    tensor_env = {'x': torch.tensor([1.0, 1.0, 0.0])}   # unit concat tt
    values = evaluate(tm, val_env)
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)(tensor_env), expected)


def test_correctness_proj1_open():
    # proj_1 x where x : (Unit, Bool) = (unit, tt)
    Bool = TySum([TyUnit(), TyUnit()])
    inner = TmVar('x')
    inner.ty_checked = TyProd([TyUnit(), Bool])
    tm = TmProj(1, inner)
    val_env: dict[str, Val] = {'x': VProd([TmUnit(), TmInj(0, TmUnit(), Bool)])}
    tensor_env = {'x': torch.tensor([1.0, 1.0, 0.0])}   # unit concat tt
    values = evaluate(tm, val_env)
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)(tensor_env), expected)


def test_correctness_seq_unit_unit():
    # unit; unit  evaluates to VUnit
    tm = TmSeq(TmUnit(), TmUnit())
    values = evaluate(tm, {})
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)({}), expected)


def test_correctness_seq_unit_inj():
    # unit; inj_0 unit : Bool  evaluates to VInj(0, VUnit(), Bool)
    Bool = TySum([TyUnit(), TyUnit()])
    tm = TmSeq(TmUnit(), TmInj(0, TmUnit(), Bool))
    values = evaluate(tm, {})
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)({}), expected)


def test_correctness_unit():
    tm = TmUnit()
    values = evaluate(tm, {})
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)({}), expected)


def test_correctness_inj0():
    Bool = TySum([TyUnit(), TyUnit()])
    tm = TmInj(0, TmUnit(), Bool)
    values = evaluate(tm, {})
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)({}), expected)


def test_correctness_inj1():
    Bool = TySum([TyUnit(), TyUnit()])
    tm = TmInj(1, TmUnit(), Bool)
    values = evaluate(tm, {})
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)({}), expected)


def test_correctness_let():
    # let x = unit in x
    tm = TmLet('x', TmUnit(), TmVar('x'))
    values = evaluate(tm, {})
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)({}), expected)


def test_correctness_case_not_tt():
    # NOT tt = ff  (closed: scrutinee is TmInj, not a variable)
    Bool = TySum([TyUnit(), TyUnit()])
    scrutinee = TmInj(0, TmUnit(), Bool)
    scrutinee.ty_checked = Bool
    tm = TmCase(scrutinee, ['x0', 'x1'], [TmInj(1, TmVar('x0'), Bool), TmInj(0, TmVar('x1'), Bool)])
    values = evaluate(tm, {})
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)({}), expected)


def test_correctness_case_not_ff():
    # NOT ff = tt  (closed: scrutinee is TmInj, not a variable)
    Bool = TySum([TyUnit(), TyUnit()])
    scrutinee = TmInj(1, TmUnit(), Bool)
    scrutinee.ty_checked = Bool
    tm = TmCase(scrutinee, ['x0', 'x1'], [TmInj(1, TmVar('x0'), Bool), TmInj(0, TmVar('x1'), Bool)])
    values = evaluate(tm, {})
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)({}), expected)


def test_correctness_case_open_not_tt():
    # NOT x with x bound to tt in the environment
    Bool = TySum([TyUnit(), TyUnit()])
    scrutinee = TmVar('x')
    scrutinee.ty_checked = Bool
    tm = TmCase(scrutinee, ['x0', 'x1'], [TmInj(1, TmVar('x0'), Bool), TmInj(0, TmVar('x1'), Bool)])
    val_env: dict[str, Val] = {'x': VInj(0, VUnit(), Bool)}   # tt as a Val
    tensor_env = {'x': torch.tensor([1.0, 0.0])}              # tt as a tensor
    values = evaluate(tm, val_env)
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)(tensor_env), expected)

def test_correctness_case_open_not_ff():
    # NOT x with x bound to ff in the environment
    Bool = TySum([TyUnit(), TyUnit()])
    scrutinee = TmVar('x')
    scrutinee.ty_checked = Bool
    tm = TmCase(scrutinee, ['x0', 'x1'], [TmInj(1, TmVar('x0'), Bool), TmInj(0, TmVar('x1'), Bool)])
    val_env: dict[str, Val] = {'x': VInj(1, VUnit(), Bool)}   # ff as a Val
    tensor_env = {'x': torch.tensor([0.0, 1.0])}              # ff as a tensor
    values = evaluate(tm, val_env)
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)(tensor_env), expected)


# --- Reverse ---

def test_compile_reverse():
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

    xs = TmProd([idx(3), idx(0), idx(1), idx(1)])  # [3,0,1,1] reversed is [1,1,0,3]
    let_tm = TmLet("xs", xs, reverse)
    _check(let_tm, {})
    print("xs:       ", compile(xs)({}).data)
    print("reversed: ", compile(let_tm)({}).data)
    for i, e in enumerate([1, 1, 0, 3]):
        tm = TmProj(i, let_tm)
        values = evaluate(tm, {})
        expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
        assert torch.equal(compile(tm)({}), expected)


# --- TmDict ---

def test_compile_dict_n1_unit():
    # dict(unit, unit) : Unit ↦ Unit
    # K = [[1.0]] (1×1), V = [[1.0]] (1×1)
    # K.T @ V = [[1.0]]
    tm_ks = TmUnit()
    tm_ks.ty_checked = TyProd([TyUnit()])
    tm_vs = TmUnit()
    tm_vs.ty_checked = TyProd([TyUnit()])
    module = compile(TmDict(tm_ks, tm_vs))
    result = module({})
    assert torch.equal(result, torch.tensor([1.0]))


def test_compile_dict_n1_wider_value():
    # dict(unit, inj_0 unit : Bool) : Unit ↦ Bool
    # K = [[1.0]] (1×1), V = [[1.0, 0.0]] (1×2)
    # (K.T @ V).flatten() = [1.0, 0.0]
    Bool = TySum([TyUnit(), TyUnit()])
    tm_ks = TmUnit()
    tm_ks.ty_checked = TyProd([TyUnit()])
    tm_vs = TmInj(0, TmUnit(), Bool)
    tm_vs.ty_checked = TyProd([Bool])
    module = compile(TmDict(tm_ks, tm_vs))
    result = module({})
    assert torch.equal(result, torch.tensor([1.0, 0.0]))


def test_compile_dict_n1_wider_key():
    # dict(inj_0 unit : Bool, unit) : Bool ↦ Unit
    # K = [[1.0, 0.0]] (1×2), V = [[1.0]] (1×1)
    # (K.T @ V).flatten() = [1.0, 0.0]
    Bool = TySum([TyUnit(), TyUnit()])
    tm_ks = TmInj(0, TmUnit(), Bool)
    tm_ks.ty_checked = TyProd([Bool])
    tm_vs = TmUnit()
    tm_vs.ty_checked = TyProd([TyUnit()])
    module = compile(TmDict(tm_ks, tm_vs))
    result = module({})
    assert torch.equal(result, torch.tensor([1.0, 0.0]))


def test_compile_dict_n2_unit():
    # dict([unit, unit], [unit, unit]) : Unit ↦ Unit with n=2
    # K = [[1.0], [1.0]] (2×1), V = [[1.0], [1.0]] (2×1)
    # (K.T @ V).flatten() = [2.0]  (entries sum, not outer product)
    tm_ks = TmProd([TmUnit(), TmUnit()])
    tm_ks.ty_checked = TyProd([TyUnit(), TyUnit()])
    tm_vs = TmProd([TmUnit(), TmUnit()])
    tm_vs.ty_checked = TyProd([TyUnit(), TyUnit()])
    module = compile(TmDict(tm_ks, tm_vs))
    result = module({})
    assert torch.equal(result, torch.tensor([2.0]))


def test_compile_dict_n2_no_cross_terms():
    # dict([k1, k2], [v1, v2]) with k1=2, k2=3, v1=4, v2=5
    # K = [[2.0], [3.0]], V = [[4.0], [5.0]]
    # (K.T @ V).flatten() = [23.0]
    # NOT torch.outer([2,3],[4,5]).flatten() = [8,10,12,15] (wrong shape, cross terms)
    tm_ks = TmProd([TmVar('k1'), TmVar('k2')])
    tm_ks.ty_checked = TyProd([TyUnit(), TyUnit()])
    tm_vs = TmProd([TmVar('v1'), TmVar('v2')])
    tm_vs.ty_checked = TyProd([TyUnit(), TyUnit()])
    module = compile(TmDict(tm_ks, tm_vs))
    result = module({'k1': torch.tensor([2.0]), 'k2': torch.tensor([3.0]),
                     'v1': torch.tensor([4.0]), 'v2': torch.tensor([5.0])})
    assert torch.equal(result, torch.tensor([23.0]))


# --- TmLookup ---

def test_compile_lookup_unit_key_unit_value_match():
    # dict({unit -> unit}) lookup unit, always-match -> [1.0]
    tm_ks = TmProd([TmUnit()])
    tm_ks.ty_checked = TyProd([TyUnit()])
    tm_vs = TmProd([TmUnit()])
    tm_vs.ty_checked = TyProd([TyUnit()])
    tm1 = TmDict(tm_ks, tm_vs)
    tm1.ty_checked = TyDict(TyUnit(), TyUnit())
    tm2 = TmUnit()
    tm2.ty_checked = TyUnit()
    module = compile(TmLookup(tm1, tm2, lambda k, q: True))
    result = module({})
    assert torch.equal(result, torch.tensor([1.0]))


def test_compile_lookup_bool_key_unit_value_query_tt_hit():
    # dict({tt -> unit}) lookup tt, equality -> [1.0]
    Bool = TySum([TyUnit(), TyUnit()])
    tm_ks = TmProd([TmInj(0, TmUnit(), Bool)])
    tm_ks.ty_checked = TyProd([Bool])
    tm_vs = TmProd([TmUnit()])
    tm_vs.ty_checked = TyProd([TyUnit()])
    tm1 = TmDict(tm_ks, tm_vs)
    tm1.ty_checked = TyDict(Bool, TyUnit())
    tm2 = TmInj(0, TmUnit(), Bool)
    tm2.ty_checked = Bool
    module = compile(TmLookup(tm1, tm2, lambda k, q: k == q))
    result = module({})
    assert torch.equal(result, torch.tensor([1.0]))


def test_compile_lookup_bool_key_unit_value_query_ff_miss():
    # dict({tt -> unit}) lookup ff, equality -> [0.0] (no match)
    Bool = TySum([TyUnit(), TyUnit()])
    tm_ks = TmProd([TmInj(0, TmUnit(), Bool)])
    tm_ks.ty_checked = TyProd([Bool])
    tm_vs = TmProd([TmUnit()])
    tm_vs.ty_checked = TyProd([TyUnit()])
    tm1 = TmDict(tm_ks, tm_vs)
    tm1.ty_checked = TyDict(Bool, TyUnit())
    tm2 = TmInj(1, TmUnit(), Bool)
    tm2.ty_checked = Bool
    module = compile(TmLookup(tm1, tm2, lambda k, q: k == q))
    result = module({})
    assert torch.equal(result, torch.tensor([0.0]))


def test_compile_lookup_bool_key_two_entries_selects_correct():
    # dict({tt -> unit, ff -> unit}) lookup ff, equality -> [1.0]
    Bool = TySum([TyUnit(), TyUnit()])
    tm_ks = TmProd([TmInj(0, TmUnit(), Bool), TmInj(1, TmUnit(), Bool)])
    tm_ks.ty_checked = TyProd([Bool, Bool])
    tm_vs = TmProd([TmUnit(), TmUnit()])
    tm_vs.ty_checked = TyProd([TyUnit(), TyUnit()])
    tm1 = TmDict(tm_ks, tm_vs)
    tm1.ty_checked = TyDict(Bool, TyUnit())
    tm2 = TmInj(1, TmUnit(), Bool)
    tm2.ty_checked = Bool
    module = compile(TmLookup(tm1, tm2, lambda k, q: k == q))
    result = module({})
    assert torch.equal(result, torch.tensor([1.0]))


def test_correctness_lookup_unit_key_unit_value():
    # dict({unit -> unit}) lookup unit
    tm_ks = TmProd([TmUnit()])
    tm_ks.ty_checked = TyProd([TyUnit()])
    tm_vs = TmProd([TmUnit()])
    tm_vs.ty_checked = TyProd([TyUnit()])
    tm1 = TmDict(tm_ks, tm_vs)
    tm1.ty_checked = TyDict(TyUnit(), TyUnit())
    tm2 = TmUnit()
    tm2.ty_checked = TyUnit()
    tm = TmLookup(tm1, tm2, lambda k, q: True)
    values = evaluate(tm, {})
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)({}), expected)


def test_correctness_lookup_bool_key_hit():
    # dict({tt -> unit}) lookup tt
    Bool = TySum([TyUnit(), TyUnit()])
    tm_ks = TmProd([TmInj(0, TmUnit(), Bool)])
    tm_ks.ty_checked = TyProd([Bool])
    tm_vs = TmProd([TmUnit()])
    tm_vs.ty_checked = TyProd([TyUnit()])
    tm1 = TmDict(tm_ks, tm_vs)
    tm1.ty_checked = TyDict(Bool, TyUnit())
    tm2 = TmInj(0, TmUnit(), Bool)
    tm2.ty_checked = Bool
    tm = TmLookup(tm1, tm2, lambda k, q: k == q)
    values = evaluate(tm, {})
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)({}), expected)


def test_correctness_lookup_bool_key_miss():
    # dict({tt -> unit}) lookup ff -> VError -> [0.0]
    Bool = TySum([TyUnit(), TyUnit()])
    tm_ks = TmProd([TmInj(0, TmUnit(), Bool)])
    tm_ks.ty_checked = TyProd([Bool])
    tm_vs = TmProd([TmUnit()])
    tm_vs.ty_checked = TyProd([TyUnit()])
    tm1 = TmDict(tm_ks, tm_vs)
    tm1.ty_checked = TyDict(Bool, TyUnit())
    tm2 = TmInj(1, TmUnit(), Bool)
    tm2.ty_checked = Bool
    tm = TmLookup(tm1, tm2, lambda k, q: k == q)
    values = evaluate(tm, {})
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)({}), expected)


def test_correctness_lookup_bool_key_two_entries():
    # dict({tt -> unit, ff -> unit}) lookup ff
    Bool = TySum([TyUnit(), TyUnit()])
    tm_ks = TmProd([TmInj(0, TmUnit(), Bool), TmInj(1, TmUnit(), Bool)])
    tm_ks.ty_checked = TyProd([Bool, Bool])
    tm_vs = TmProd([TmUnit(), TmUnit()])
    tm_vs.ty_checked = TyProd([TyUnit(), TyUnit()])
    tm1 = TmDict(tm_ks, tm_vs)
    tm1.ty_checked = TyDict(Bool, TyUnit())
    tm2 = TmInj(1, TmUnit(), Bool)
    tm2.ty_checked = Bool
    tm = TmLookup(tm1, tm2, lambda k, q: k == q)
    values = evaluate(tm, {})
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)({}), expected)


# --- TmChoice ---

def test_compile_choice_unit_unit():
    # choice(unit, unit) compiles to [1.0] + [1.0] = [2.0]
    module = compile(TmChoice(TmUnit(), TmUnit()))
    result = module({})
    assert torch.equal(result, torch.tensor([2.0]))


def test_compile_choice_inj0_inj1():
    # choice(inj_0 unit, inj_1 unit) : Bool + Bool
    # [1.0, 0.0] + [0.0, 1.0] = [1.0, 1.0]
    Bool = TySum([TyUnit(), TyUnit()])
    module = compile(TmChoice(TmInj(0, TmUnit(), Bool), TmInj(1, TmUnit(), Bool)))
    result = module({})
    assert torch.equal(result, torch.tensor([1.0, 1.0]))


def test_compile_choice_from_vars():
    # choice(x, x) with x = [3.0] -> [6.0]
    module = compile(TmChoice(TmVar('x'), TmVar('x')))
    result = module({'x': torch.tensor([3.0])})
    assert torch.equal(result, torch.tensor([6.0]))


def test_compile_choice_asymmetric():
    # choice(x, y) sums the two independent tensors
    module = compile(TmChoice(TmVar('x'), TmVar('y')))
    result = module({'x': torch.tensor([2.0]), 'y': torch.tensor([5.0])})
    assert torch.equal(result, torch.tensor([7.0]))


def test_correctness_choice_unit_unit():
    # choice(unit, unit): evaluates to [VUnit, VUnit], compiled sum = [2.0]
    tm = TmChoice(TmUnit(), TmUnit())
    values = evaluate(tm, {})
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)({}), expected)


def test_correctness_choice_inj0_inj1():
    # choice(inj_0 unit, inj_1 unit) : Bool
    Bool = TySum([TyUnit(), TyUnit()])
    tm = TmChoice(TmInj(0, TmUnit(), Bool), TmInj(1, TmUnit(), Bool))
    values = evaluate(tm, {})
    expected = torch.stack([compile_val(v)({}) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)({}), expected)


def test_correctness_choice_open():
    # choice(x, x) with x : Unit
    tm = TmChoice(TmVar('x'), TmVar('x'))
    val_env: dict[str, Val] = {'x': VUnit()}
    tensor_env = {'x': torch.tensor([1.0])}
    values = evaluate(tm, val_env)
    expected = torch.stack([compile_val(v)(tensor_env) for v in values]).sum(dim=0)
    assert torch.equal(compile(tm)(tensor_env), expected)


def test_correctness_all_constructors():
    # Verifies the compiler correctness theorem on a single open term that uses all 11
    # term constructors. The output type is Unit+Unit+Unit.
    #
    # Program (free variables x, y, z : Unit+Unit):
    #   let d = dict((x,) -> (y,)) in       -- TmLet, TmDict, TmProd, TmVar
    #   let r = lookup(d, z, ==) in          -- TmLet, TmLookup, TmVar
    #   case r of                            -- TmCase, TmVar
    #     | inj_0 a -> (proj_0 (unit,unit)) ; choice(inj_0 a, inj_1 a)
    #                                        -- TmSeq, TmProj, TmProd, TmUnit, TmChoice, TmInj, TmVar
    #     | inj_1 b -> inj_2 b              -- TmInj, TmVar
    output_ty = TySum([TyUnit(), TyUnit(), TyUnit()])
    Bool = TySum([TyUnit(), TyUnit()])

    tm_prod_units = TmProd([TmUnit(), TmUnit()])
    branch0 = TmSeq(
        TmProj(0, tm_prod_units),
        TmChoice(
            TmInj(0, TmVar('a'), output_ty),
            TmInj(1, TmVar('a'), output_ty),
        )
    )
    branch1 = TmInj(2, TmVar('b'), output_ty)

    tm = TmLet('d',
        TmDict(
            TmProd([TmVar('x')]),
            TmProd([TmVar('y')]),
        ),
        TmLet('r',
            TmLookup(TmVar('d'), TmVar('z'), lambda a, b: a == b),
            TmCase(TmVar('r'), ['a', 'b'], [branch0, branch1])
        )
    )

    ctx: Ctx = {'x': Bool, 'y': Bool, 'z': Bool}
    _check(tm, ctx)

    def assert_correctness(env: dict[str, Val]) -> None:
        vs = evaluate(tm, env)
        env_t = {name: compile_val(v)({}) for name, v in env.items()}
        lhs = compile(tm)(env_t)
        rhs = torch.stack([compile_val(v)(env_t) for v in vs]).sum(dim=0)
        assert torch.equal(lhs, rhs)

    # Case 1: lookup matches, result is inj_0 -> branch0 -> TmChoice yields two nondeterministic values
    env1: dict[str, Val] = {'x': VInj(0, VUnit(), Bool), 'y': VInj(0, VUnit(), Bool), 'z': VInj(0, VUnit(), Bool)}
    assert evaluate(tm, env1) == [VInj(0, VUnit(), output_ty), VInj(1, VUnit(), output_ty)]
    assert_correctness(env1)

    # Case 2: lookup matches, result is inj_1 -> branch1 -> deterministic inj_2
    env2: dict[str, Val] = {'x': VInj(0, VUnit(), Bool), 'y': VInj(1, VUnit(), Bool), 'z': VInj(0, VUnit(), Bool)}
    assert evaluate(tm, env2) == [VInj(2, VUnit(), output_ty)]
    assert_correctness(env2)

    # Case 3: lookup fails (key != query) -> VError propagates as zero vector
    env3: dict[str, Val] = {'x': VInj(0, VUnit(), Bool), 'y': VInj(0, VUnit(), Bool), 'z': VInj(1, VUnit(), Bool)}
    assert_correctness(env3)


# ============= `dim`: Unit Testing

def test_dim_unit():
    assert dim(TyUnit()) == 1

def test_dim_prod():
    assert dim(TyProd([TyUnit(), TyUnit()])) == 2

def test_dim_prod_triple():
    assert dim(TyProd([TyUnit(), TyUnit(), TyUnit()])) == 3

def test_dim_sum():
    assert dim(TySum([TyUnit(), TyUnit()])) == 2

def test_dim_sum_triple():
    assert dim(TySum([TyUnit(), TyUnit(), TyUnit()])) == 3

def test_dim_dict():
    assert dim(TyDict(TyUnit(), TyUnit())) == 1

def test_dim_dict_prod_key():
    assert dim(TyDict(TyProd([TyUnit(), TyUnit()]), TyUnit())) == 2


# ============= `zero`: Unit Testing

def test_zero_unit():
    assert torch.equal(zero(TyUnit()), torch.zeros(1))

def test_zero_sum():
    assert torch.equal(zero(TySum([TyUnit(), TyUnit()])), torch.zeros(2))

def test_zero_prod():
    assert torch.equal(zero(TyProd([TyUnit(), TyUnit()])), torch.zeros(2))