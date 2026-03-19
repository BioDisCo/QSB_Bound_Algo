# QSB Bound Algorithm

**A Python library for computing rigorous upper bounds on fault probabilities in Chemical Reaction Networks (CRNs), using Quasi-Stationary Distribution (QSD) theory.**

This repository accompanies the paper:

> **A Quasi-Stationary Distribution Bound for Fault Analysis in Gene Regulatory Networks**
> Fabricio Cravo, Matthias Fuegger, Thomas Nowak
> *bioRxiv*, 2025. https://www.biorxiv.org/content/10.1101/2025.10.21.683707v1

Models are specified using the **MobsPy** language. See the [MobsPy documentation](https://mobspy-doc.readthedocs.io/en/latest/) for a full reference on how to write models.

---

## What This Does

Markov theorey predicts that biological systems spend long periods near a desired steady state before eventually transitioning to an undesirable one. This undesirable transition is a **fault**. Computing the exact probability of a fault is generally intractable.

This library computes a **mathematically rigorous upper bound** on that probability. Given:
- A CRN model written in MobsPy,
- A bounded region of the state space to analyze (a count interval per species),
- An initial condition,
- A definition of which boundary of that region constitutes a "fault,"

the algorithm returns a probability value `p` such that the true fault probability is guaranteed to be smaller than `p`

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/BioDisCo/QSB_Bound_Algo.git
cd QSB_Bound_Algo
pip install mobspy numpy scipy numba matplotlib joblib
```

There is no `pip install` for this package itself. Scripts are run from the repository root.

---

## Quick Start

The entire workflow is handled by the class `Jump_chain_qsd_bound`, imported from `functional_compiler.py`.

```python
from mobspy import *
from functional_compiler import Jump_chain_qsd_bound
```

### Step 1 — Write a MobsPy model

Define your CRN as a standard MobsPy model. The example below is a toogle switch and we provide an upper bound on the probability of transition between both stable states in it.

```python
from mobspy import *
from functional_compiler import Jump_chain_qsd_bound

initial_count, promoter_number = 100, 25

Promoters, Mortal = BaseSpecies()
Promoters.active, Promoters.inactive

P1, P2 = New(Promoters)
S1, S2 = New(Mortal)

Mortal >> Zero [0.01]

Rev[4*S1 + P2.active >> P2.inactive] [10**(1/4), 1]
Rev[4*S2 + P1.active >> P1.inactive] [10**(1/4), 1]

P1.active >> P1 + S1 [1/promoter_number]
P2.active >> P2 + S2 [1/promoter_number]

S1(initial_count)
P1(promoter_number), P2(promoter_number)

S = Simulation(S1 | S2 | P1 | P2)
```

### Step 2 — Define the state-space interval

Specify the range of molecule counts to consider for each species. This defines the bounded region within which the algorithm operates. States outside this region are treated as the absorbing (fault) set.

```python
interval = {
    S1: [0, 200],
    S2: [0, 200],
    P1.active: [0, 25],
    P2.active: [0, 25],
}
```

### Step 3 — Define the fault boundary (optional)

By default, any transition leaving the interval is treated as a fault. You can also specify which direction of exit counts as a fault using `exit_direction`. The options are `'above'` (or `'+'`) and `'below'` (or `'-'`).

```python
# Fault = protein S1 dropping to zero (system flips to the other stable state)
exit_direction = {S1: 'below'}
```

If `exit_direction` is omitted, all exits from the interval are treated as faults.

### Step 4 — Construct the bound object and run

```python
bound_obj = Jump_chain_qsd_bound(
    simulation_object=S,
    interval=interval,
    exit_direction=exit_direction,
    verbose=True          # prints progress messages
)

probability, decay_parameter = bound_obj.calculate_bound()

print(f"Upper bound on fault probability: {probability}")
print(f"Decay parameter (absorption rate): {decay_parameter}")
```

---

## The `Jump_chain_qsd_bound` Class

### Constructor

```python
Jump_chain_qsd_bound(simulation_object, interval, exit_direction=None, verbose=True)
```

| Parameter | Type | Description |
|---|---|---|
| `simulation_object` | MobsPy `Simulation` | A compiled MobsPy simulation object containing the model |
| `interval` | `dict` | Maps each species (or species object) to a `[min, max]` count range |
| `exit_direction` | `dict` or `None` | Maps species to `'above'`/`'+'` or `'below'`/`'-'` to restrict which boundary is a fault. If `None`, all exits are faults |
| `verbose` | `bool` | If `True`, prints progress messages during matrix compilation and bound estimation |

### Methods

**`calculate_bound(...)`** — Runs the full bound estimation. Returns `(probability, decay_parameter)`.

```python
bound_obj.calculate_bound(
    max_iterations_per_diameter=5,  # max iterations = this * number of states
    probability_epsilon_bound=1e-5, # stop if bound changes by less than this per step
    stopping_probability=1e-5,      # stop if residual probability drops below this
    animation=False                 # show live animation of the algorithm running
)
```

**`calculate_qsd()`** — Computes and returns just the Quasi-Stationary Distribution (the left eigenvector of the conditional Q-matrix near eigenvalue 0), without running the full bound. Returns a numpy array of length equal to the number of states.

```python
qsd = bound_obj.calculate_qsd()
```

**`partial_qsd(meta_species)`** — Returns the marginal QSD for a single species — i.e., the QSD summed over all states for a given species count. Useful for understanding which molecule counts the system spends its time near.

```python
marginal = bound_obj.partial_qsd(S1)
```

**`plot_qsd()`** — Plots the full QSD over all state indices using matplotlib.

**`animate_bound()`** — Shows a live matplotlib animation of the bound algorithm running, illustrating how the initial probability mass and the QSD interact step by step.

---

## Reducing State Space with `Assign`

If your model has a **conservation law** — for example, the total number of promoters (active + inactive) is always fixed — you can use the `Assign` class to express one species as a deterministic function of others. This collapses one dimension of the state space, which can dramatically reduce computation time.

```python
from functional_compiler import Jump_chain_qsd_bound
from assign import Assign

# Total promoters P1.active + P1.inactive = promoter_number (always)
# So we do not enumerate P1.inactive independently

interval = {
    S1: [0, 200],
    S2: [0, 200],
    P1.active: [0, 25],
    P2.active: [0, 25],
    P1.inactive: Assign(lambda p1_active: promoter_number - p1_active, P1.active),
    P2.inactive: Assign(lambda p2_active: promoter_number - p2_active, P2.active),
}
```

When a species entry in the `interval` dict is an `Assign` object instead of a `[min, max]` list, it is not enumerated as an independent dimension — its value is derived from the other species at each state, keeping the state space smaller.

---

## How the Algorithm Works

The algorithm runs in three stages internally:

**1. Compilation.** `q_compile()` processes the MobsPy `Simulation` object, resolving meta-species inheritance and producing a list of reactions — each represented as a callable rate function and a `delta_species` dict encoding stoichiometric changes.

**2. Q-matrix construction.** `q_matrix_generator()` enumerates the full Cartesian product of the species count intervals to define the state space, then builds a sparse generator matrix Q (using `scipy.sparse`) over those states plus one absorbing fault state. Any transition that leaves the defined interval is routed to the absorbing state.

**3. Bound estimation.** `bound_estimator()` extracts the conditional Q-matrix (absorbing column zeroed out) and finds the QSD via sparse eigenvector decomposition near eigenvalue 0. It then constructs the **jump chain** (the embedded discrete-time Markov chain) of Q, and iterates it from the initial state — subtracting the QSD mass at each step. The residual probability that escapes this subtraction is the returned upper bound. The decay parameter comes from the two smallest-magnitude eigenvalues of the full Q-matrix. A bidirectional BFS connectivity check is run before estimation; if the conditional state space is not irreducible, an informative error is raised.

For performance, the inner loop of the bound propagation is compiled to native code with **Numba** (`@njit`, `prange`), and the Q-matrix is stored in sparse CSR format throughout.

---

## Repository Structure

```
QSB_Bound_Algo/
│
├── functional_compiler.py      # Core: pipeline, Q-matrix, bound algorithm, main class
├── reaction_construction.py    # Translates MobsPy meta-reactions to compiled rate functions
├── rate_function_scripts.py    # Mass-action and callable rate function construction
├── called_argument_class.py    # Wraps species counts for use in callable rate laws
├── assign.py                   # Assign class for conservation law constraints
├── bound_animator.py           # Matplotlib animation of the bound algorithm
├── utils.py                    # Utilities for MobsPy's meta-species structure
├── for_local_use.py            # Canonical usage example (bistable toggle switch)
│
├── Circuit_Examples/           # Additional CRN circuit examples
├── Rare_Event_Detection/       # Rare-event detection examples
└── test_scripts/               # Test scripts
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{cravo2025qsd,
  title   = {A Quasi-Stationary Distribution Bound for Fault Analysis in Gene Regulatory Networks},
  author  = {Cravo, Fabricio and Fuegger, Matthias and Nowak, Thomas},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.10.21.683707}
}
```

For the MobsPy language:

```bibtex
@article{cravo2025mobspy,
  title   = {MobsPy: A programming language for biochemical reaction networks},
  author  = {Cravo, Fabricio and Prakash, Gayathri and Fuegger, Matthias and Nowak, Thomas},
  journal = {PLOS Computational Biology},
  year    = {2025},
  doi     = {10.1371/journal.pcbi.1013024}
}
```

---

## Issues and Bugs

Please report issues in the Issues tab. 
If necessary contact Fabricio Cravo by email at fabriciocravo [at] gmail [dot] com, he can help with debugging and tutorials for those wanting to use this repo.

---

## Authors

**Fabricio Cravo** — NeuroPRISM Lab, Northeastern University
**Matthias Fuegger** — CNRS, LMF, ENS Paris-Saclay
**Thomas Nowak** — CNRS, LMF, ENS Paris-Saclay

---

## License

MIT — see `LICENSE` for details.
