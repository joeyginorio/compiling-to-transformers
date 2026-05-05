"""
First-token routed rotation in Cajal.

The first input character chooses a left-rotation amount. The program builds a
runtime dictionary from input positions to characters, then each output slot
looks up the position `(slot + shift) mod N`, where `shift` is the first
character's injection index.

Run: python -m experiments.rotate.rotate
"""

from collections import Counter

from cajal.syntax import (
    Ctx,
    Tm,
    TmDict,
    TmInj,
    TmLet,
    TmLookup,
    TmProd,
    TmProj,
    TmUnit,
    TmVar,
    TyProd,
    TySum,
    TyUnit,
    VInj,
    VProd,
)
from cajal.typing import check
from cajal.evaluating import evaluate


# --- The example we're verifying --------------------------------------------

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
INPUT = "cdefghijklmnoab"

K = 15
N = len(INPUT)
SHIFT = ALPHABET.index(INPUT[0]) % N
EXPECTED = INPUT[SHIFT:] + INPUT[:SHIFT]


# --- Types ------------------------------------------------------------------

CHAR = TySum([TyUnit()] * K)
POS = TySum([TyUnit()] * N)
SEQ = TyProd([CHAR] * N)


# --- Program ----------------------------------------------------------------

ROTATE: Tm = TmLet(
    "d",
    TmDict(
        TmProd([TmInj(i, TmUnit(), POS) for i in range(N)]),
        TmProd([TmProj(i, TmVar("xs_value")) for i in range(N)]),
    ),
    TmProd([
        TmLookup(
            TmVar("d"),
            TmProj(0, TmVar("xs_ctrl")),
            lambda key, query, slot=slot: key.n == (slot + query.n) % N,
        )
        for slot in range(N)
    ]),
)


# --- Running and verifying --------------------------------------------------

def encode_input(s: str) -> dict:
    terms = [
        TmInj(ALPHABET.index(ch), TmUnit(), CHAR)
        for ch in s
    ]
    value = VProd(terms)
    return {
        "xs_ctrl": value,
        "xs_value": value,
    }


def run(s: str) -> str:
    [out] = evaluate(ROTATE, encode_input(s))
    assert isinstance(out, VProd), f"expected VProd output, got {out}"
    chars = []
    for slot in out.tm:
        [v] = evaluate(slot, out.env)
        assert isinstance(v, VInj), f"unexpected: {v}"
        chars.append(ALPHABET[v.n])
    return "".join(chars)


def count_nodes(tm: Tm) -> int:
    counts: Counter[str] = Counter()

    def visit(term: Tm) -> None:
        counts[type(term).__name__] += 1
        match term:
            case TmVar() | TmUnit():
                return
            case TmProd(tms):
                for child in tms:
                    visit(child)
            case TmInj(_, inner, _):
                visit(inner)
            case TmDict(left, right) | TmLookup(left, right, _):
                visit(left)
                visit(right)
            case TmProj(_, inner):
                visit(inner)
            case TmLet(_, bound, body):
                visit(bound)
                visit(body)
            case _:
                raise TypeError(f"unhandled term while counting nodes: {term!r}")

    visit(tm)
    return sum(counts.values())


def main() -> None:
    ctx: Ctx = {
        "xs_ctrl": SEQ,
        "xs_value": SEQ,
    }
    check(ROTATE, ctx)

    print("=" * 60)
    print(f"Cajal first-token routed rotation   n = {N},  k = {K}")
    print(f"shift = index({INPUT[0]!r}) mod {N} = {SHIFT}")
    print(f"AST nodes = {count_nodes(ROTATE)}")
    print("=" * 60)
    print()

    got = run(INPUT)
    ok = "ok" if got == EXPECTED else "FAIL"
    print(f"  input     :  ({', '.join(INPUT)})")
    print(f"  output    :  ({', '.join(got)})    {ok}")
    print(f"  expected  :  ({', '.join(EXPECTED)})")
    print()


if __name__ == "__main__":
    main()
