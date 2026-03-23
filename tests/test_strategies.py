from hypothesis import given, settings, HealthCheck
from strategies import gen_prog
from cajal.typing import check

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(gen_prog())
def test_gen_tm(prog):
    ctx, tm, ty = prog
    ty_check = check(tm, ctx.flat())
    assert ty == ty_check