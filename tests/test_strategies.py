import hypothesis.strategies as st
from hypothesis import given, settings, HealthCheck
from strategies import gen_prog, gen_val, gen_ty
from cajal.typing import check, check_val

# ============= `gen_tm`: Property-based Testing

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(gen_prog())
def test_gen_tm(prog):
    ctx, tm, ty = prog
    ty_check = check(tm, ctx.flat())
    assert ty == ty_check

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(gen_ty(), st.data())
def test_gen_val(ty, data):
    val = data.draw(gen_val(ty, set()))
    ty_check = check_val(val)
    assert ty == ty_check