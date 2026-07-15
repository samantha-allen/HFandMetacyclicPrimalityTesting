# HF and Metacyclic Primality Testing Repository
Authors:
Samantha Allen (Duquesne University)
Charles Livingston (Indiana University)

# This repository currently contains two files:

(1) HFandMetacyclicPrimeness.sage (described below)
(2) UseExamples.ipynb (instructions for basic use-cases)

# HFandMetacyclicPrimeness.sage

A single-file SageMath script that tests whether a knot is prime, given its
PD or DT code. It runs the full procedure of Section 10.2 ("The general
approach") of Allen-Livingston: enumerate positive-symmetric factorizations of 
the HF polynomial, prune with the Jones and HOMFLY tests, prune with homology-order
consistency on cyclic branched covers, then prune with metacyclic
Betti-number obstructions.

The file has two parts. The **Testing** section (top) is the driver you
actually call: `HFandMetaPrimeTest`, `is_prime_knot`, and the private
`_step*` helpers that walk through the procedure stage by stage. The
**Computation** section (everything below it) is the underlying math
library — Wirtinger presentations, Fox calculus, metacyclic
representations, the Jones/HOMFLY tests, and the hardcoded reference data
for the seven "HF-detected" knots those two tests check against. Everything
the driver needs is defined in this same file, so `load()`-ing it is the
only setup step.

## Requirements

This needs an actual **SageMath** kernel — not a plain Python kernel with
`sage` installed as a library. `load()`, the `R.<s,t> = PolynomialRing(...)`
syntax, and `^` for exponentiation are all Sage preparser features and will
raise `NameError`/`SyntaxError` under a regular Python/IPython kernel. If
you're in PyCharm, make sure the notebook's kernel is set to SageMath (see
Troubleshooting below if it isn't listed).

It also needs `snappy` (for link/PD/DT handling and the Jones/Alexander/
HOMFLY polynomials).

## Quick start

```python
load('HFandMetacyclicPrimeness.sage')

# Any SnapPy-recognized knot name works for getting a PD code to test
trefoil_pd = snappy.Link('3_1').PD_code()
result = HFandMetaPrimeTest(trefoil_pd, verbose=True)
result
# -> {'prime': True, 'stage': 'step1_factorization_jones_homfly', 'pd': [...], 'remaining_factorizations': []}

# A PD code you already have (list of 4-lists)
result = HFandMetaPrimeTest([[4,2,5,1],[8,6,1,5],[6,3,7,4],[2,7,3,8]])

# A DT code
result = HFandMetaPrimeTest('DT: [(4,6,2)]')

# Just the yes/no answer
is_prime_knot(trefoil_pd)
```

## API

### `HFandMetaPrimeTest(code, primes=[2,3,5,7], verbose=False)`

Runs the full procedure and returns a dict:

| key | meaning |
|---|---|
| `prime` | `True` if confirmed prime, `None` if undetermined |
| `stage` | which test resolved it, or `"inconclusive"` |
| `pd` | the normalized PD code that was actually tested |
| `remaining_factorizations` | surviving candidate factorizations `[Omega_1, Omega_2]`, if any |

**`prime` is never `False`.** These tests can only ever *confirm* primeness
— they can't prove a knot is composite — so an unresolved case is reported
as `None`, not `False`. Treat `None` as "not yet determined by this run,"
not as evidence the knot is a connect sum.

`code` accepts either:
- a **PD code**: a list of 4-element lists, one per crossing, in the
  standard SnapPy/spherogram convention (the same format `Link.PD_code()`
  returns), or
- a **DT code**: a string, with or without the `'DT: '` prefix.

Either way it's normalized through `snappy.Link(code).PD_code()` before
anything else runs, so any consistent indexing convention on input is fine.

`primes` controls which cover orders are used in the homology-order check
(stage 2) and the metacyclic tests (stages 3-5). The default is
`[2,3,5,7]`; for larger primes — pass a longer list (e.g. `[2,3,5,7,11,13]`). 
Larger prime lists take longer to run.

`verbose=True` prints which stage resolved the knot (or how many
factorizations survived) as it runs.

### `is_prime_knot(code, primes=[2,3,5,7])`

Thin wrapper: `HFandMetaPrimeTest(code, primes=primes)["prime"]`. Returns
`True` or `None`.

## Stage labels

`stage` in the returned dict tells you which test resolved the knot:

- `step1_factorization_jones_homfly` — resolved by factorization
  enumeration (the HF polynomial had no nontrivial positive-symmetric
  factorization, or every candidate factorization was ruled out by the
  Jones or HOMFLY test).
- `step2_homology_order_pN` — resolved because no surviving factorization's
  implied `H_1` orders were consistent with the actual `H_1` of the `n`-fold
  branched cover.
- `steps3-5_metacyclic` — resolved by a metacyclic Betti-number obstruction
  (either the `d1 != d2` or `d1 = d2` case).
- `inconclusive` — none of the above resolved it; `remaining_factorizations`
  holds what's left. Try a larger `primes` list before concluding anything.

## A scope note on Jones_test / HOMFLY_TEST

These two only ever rule out a candidate factorization when one of its two
factors' HF polynomial exactly matches one of **seven hardcoded (families
of) knots**: T(2,3), 4_1, T(2,5), 5_2, Wh⁺(T(2,3),2), the P(-3,3,2n+1)
family, and 15n43522/Wh⁻(T(2,3),2). If neither candidate factor is one of
those seven, both tests correctly report "still possible" and let the
factorization through — that's expected, not a failure to detect anything,
and it just means the knot falls through to the homology-order and
metacyclic stages instead.
