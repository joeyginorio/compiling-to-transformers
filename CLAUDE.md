# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cajal is a typed functional programming language designed to compile to transformer (linear map) architectures. It includes a formal specification (`spec/`), from which this code implements an interpreter, and a compiler. `preamble.sty` contains custom commands used in `spec/specification.md`. The language supports unit types, sum types, product types, dictionaries, nondeterministic choice, and let bindings.

## Workflow

- After making code changes, always run the relevant tests in `test/` before reporting completion.
- After making code changes, always check whether the code is violating the intent of the specification in `spec/specification.md`.
- Avoid solutions which are likely to incur technical debt, and strive for simplicity when possible.

## Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run a single test
python -m pytest tests/test_evaluating.py::test_unit -v

# Type check with pyright
pyright src/

# Install dev dependencies
pip install -e ".[dev]"
```

## Architecture

The source lives in `src/cajal/` with three modules that build on each other:

- **`syntax.py`** — Defines the core data types using dataclasses: types (`Ty`), terms (`Tm`), values (`Val`), and contexts (`Ctx`). Everything else imports from here.
- **`evaluating.py`** — Interpreter implementing operational semantics. The `evaluate(tm, env)` function pattern-matches on term constructors and recursively evaluates. Pairs are lazy (store terms, not values). Errors propagate through all compound forms.
- **`typing.py`** — Implements a linear type system where context variables are consumed on use. Returns `tuple[Ty, Ctx]` (the type and remaining context).

The formal specification in `spec/specification.md` defines the language's syntax, dynamics, and type system in LaTeX. The evaluator and type checker implement rules from this spec.

## Key Design Decisions

- **Linear type system**: Variables are consumed when used; the type checker threads context through and removes bindings on use.
- **Nondeterministic choice**: `Choice(tm1, tm2)` randomly picks one branch (50/50).
- **Dictionary values use relations**: `Lookup` takes a relation function and a key to query dictionaries.
- **Python 3.12+ required**: Uses modern match statements for pattern matching throughout.
