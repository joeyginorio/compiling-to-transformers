import torch
from cajal.syntax import *
from cajal.compiling import compile, dim, zero


# --- dim ---

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


# --- zero ---

def test_zero_unit():
    assert torch.equal(zero(TyUnit()), torch.zeros(1))

def test_zero_sum():
    assert torch.equal(zero(TySum([TyUnit(), TyUnit()])), torch.zeros(2))

def test_zero_prod():
    assert torch.equal(zero(TyProd([TyUnit(), TyUnit()])), torch.zeros(2))


# --- compile: TmUnit ---

def test_compile_unit():
    module = compile(TmUnit())
    result = module({})
    assert torch.equal(result, torch.tensor([1.0]))


# --- compile: TmVar ---

def test_compile_var():
    module = compile(TmVar('x'))
    result = module({'x': torch.tensor([1.0])})
    assert torch.equal(result, torch.tensor([1.0]))

def test_compile_var_selects_correct_binding():
    module = compile(TmVar('y'))
    env = {'x': torch.tensor([1.0]), 'y': torch.tensor([2.0])}
    result = module(env)
    assert torch.equal(result, torch.tensor([2.0]))


# --- compile: TmProd ---

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


# --- compile: TmInj ---

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


# --- compile: TmLet ---

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
