import torch
from hypothesis import given, settings, HealthCheck
from cajal.syntax import TyProd, TySum, TyUnit, TmProj, TmVar
from cajal.evaluating import evaluate
from cajal.compiling import compile_val
from experiments.reverse.models import reverse, reverse_attn, reverse_module, reverse_attn_module
from strategies import gen_val

_elem_ty = TySum([TyUnit()] * 4)
_seq_ty  = TyProd([_elem_ty] * 15)


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(gen_val(_seq_ty, set()))
def test_reverse_attn_eval_equiv(v):
    # VProd is lazy, so we force both outputs by projecting each element.
    r1 = evaluate(reverse, {'x': v})[0]
    r2 = evaluate(reverse_attn, {'x': v})[0]
    for i in range(15):
        assert evaluate(TmProj(i, TmVar('r')), {'r': r1}) == evaluate(TmProj(i, TmVar('r')), {'r': r2})


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(gen_val(_seq_ty, set()))
def test_reverse_attn_compiled_equiv(v):
    x_vec = compile_val(v)({})
    torch.testing.assert_close(reverse_module({'x': x_vec}), reverse_attn_module({'x': x_vec}))
