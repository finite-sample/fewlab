"""Balanced sampling that delivers the inclusion probabilities it promises.

Everything fewlab estimates is weighted by ``1 / pi``, which is unbiased only if
item ``j`` really is included with probability ``pi[j]``. Two things make that
hard to arrange at once:

* **Fixed size.** The sample must contain exactly ``budget`` items, so the
  inclusion indicators cannot be independent.
* **Balance.** The point of the exercise is to make the sample representative --
  ``sum_j (I_j / pi_j - 1) g_j`` near zero -- which correlates the indicators
  further.

The obvious approaches sacrifice the first requirement to get the other two. A
draw of ``budget`` items without replacement with probabilities *proportional to*
``pi`` (``rng.choice(..., replace=False, p=pi)``) does not have inclusion
probabilities ``pi``: successive draws condition on what came before, and the
gap is large, not a rounding artefact. On the package's own test fixture that
draw gave items with ``pi == 1`` an inclusion rate of 0.75. Repairing a sample by
greedily swapping items to improve balance is worse still, since the swap
decisions depend on the data and distort ``pi`` further -- the same fixture fell
to 0.45.

The cube method solves all three at once. Write the problem geometrically: the
vector of inclusion probabilities is a point in the cube ``[0, 1]^m``, and a
sample is a vertex. Balance and fixed size are linear constraints, so they cut
out an affine subspace through that point.

**Flight phase.** Repeatedly take a random step along a direction lying in the
null space of the constraint matrix, as far as the cube permits, choosing between
the two directions with probabilities that make the walk a martingale. Because
the step stays in the null space the constraints hold exactly throughout; because
the walk is a martingale the expectation is exactly the ``pi`` we started with;
and because each step runs until it hits a face, at least one coordinate is
settled to 0 or 1 every iteration. The whole flight is therefore O(m) steps.

**Landing phase.** The flight stalls once fewer coordinates remain unsettled than
there are constraints. The remaining constraints are then dropped one at a time,
least important first, and the flight resumes. Balance degrades gracefully;
``pi`` and the fixed size do not degrade at all, because the sample-size
constraint is the last one standing and is never dropped.

Reference:
    Deville, J.-C. and Tille, Y. (2004). Efficient balanced sampling: the cube
    method. *Biometrika* 91(4), 893-912.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cube_sample", "scale_pi_to_budget"]

# A coordinate this close to a face of the cube is treated as having landed on
# it. Steps are computed to reach a face exactly, so the slack absorbs rounding
# only -- it is not a convergence tolerance.
LANDED: float = 1e-9


def scale_pi_to_budget(
    pi: np.ndarray, budget: int, *, max_iter: int = 1000
) -> np.ndarray:
    """Rescale inclusion probabilities to sum to ``budget``, respecting the cap.

    ``sum(pi)`` is the expected sample size, so a fixed-size design of ``budget``
    items is only coherent when the probabilities sum to ``budget``. Scaling by a
    constant would push large entries above one, which is not a probability, so
    entries that would exceed one are pinned there and the remaining budget is
    redistributed over the rest. Repeating that to a fixed point is the standard
    construction, and it terminates because each pass pins at least one entry.

    Args:
        pi: Non-negative target probabilities.
        budget: Desired sample size, at most ``len(pi)``.
        max_iter: Guard on the redistribution loop.

    Returns:
        np.ndarray: Probabilities in ``[0, 1]`` summing to ``budget``.

    Raises:
        ValueError: If ``budget`` is negative or exceeds the number of items, or
            if ``pi`` has no positive entry to carry the budget.

    Examples:
    >>> import numpy as np
    >>> scaled = scale_pi_to_budget(np.array([0.9, 0.2, 0.2, 0.2]), 2)
    >>> float(scaled.sum())
    2.0
    >>> bool((scaled <= 1.0).all())
    True
    """
    values = np.asarray(pi, dtype=float)
    n_items = values.size
    if not 0 <= budget <= n_items:
        raise ValueError(f"budget must be in [0, {n_items}], got {budget}")
    if budget == 0:
        return np.zeros(n_items)
    if budget == n_items:
        return np.ones(n_items)
    if not (values > 0).any():
        raise ValueError("pi must have at least one positive entry")

    scaled = np.clip(values, 0.0, 1.0)
    pinned = np.zeros(n_items, dtype=bool)
    for _ in range(max_iter):
        free = ~pinned
        remaining = budget - float(pinned.sum())
        free_mass = float(scaled[free].sum())
        if free_mass <= 0.0:
            break
        scaled[free] = scaled[free] * (remaining / free_mass)
        over = free & (scaled >= 1.0)
        if not over.any():
            break
        scaled[over] = 1.0
        pinned |= over
    return np.clip(scaled, 0.0, 1.0)


def _null_direction(constraints: np.ndarray) -> np.ndarray | None:
    """A unit vector in the null space of ``constraints``.

    Args:
        constraints: Matrix with strictly fewer rows than columns, so that a null
            direction is guaranteed to exist.

    Returns:
        np.ndarray | None: A unit null vector, or None if none was found to
        acceptable accuracy.
    """
    _, _, right = np.linalg.svd(constraints)
    direction = right[-1]
    if np.linalg.norm(constraints @ direction) > 1e-8 * max(
        1.0, float(np.abs(constraints).max())
    ):
        return None
    return direction


def _step_lengths(point: np.ndarray, direction: np.ndarray) -> tuple[float, float]:
    """How far the point can move each way along ``direction`` inside the cube.

    Args:
        point: Current coordinates, all within ``[0, 1]``.
        direction: Direction to travel.

    Returns:
        tuple of float: Maximum steps forward and backward before a face is hit.
    """
    forward, backward = [], []
    positive, negative = direction > 0, direction < 0
    if positive.any():
        forward.append((1.0 - point[positive]) / direction[positive])
        backward.append(point[positive] / direction[positive])
    if negative.any():
        forward.append(point[negative] / -direction[negative])
        backward.append((1.0 - point[negative]) / -direction[negative])
    if not forward:
        return 0.0, 0.0
    return (
        max(0.0, float(np.concatenate(forward).min())),
        max(0.0, float(np.concatenate(backward).min())),
    )


def _unsettled(state: np.ndarray) -> np.ndarray:
    """Positions still strictly inside the cube.

    Args:
        state: Current coordinates.

    Returns:
        np.ndarray: Indices whose value is neither 0 nor 1.
    """
    return np.flatnonzero((state > LANDED) & (state < 1.0 - LANDED))


def _flight_phase(
    state: np.ndarray, constraints: np.ndarray, rng: np.random.Generator
) -> None:
    """Run the martingale walk until too few coordinates remain to continue.

    Modifies ``state`` in place. Only ``rank + 1`` coordinates are worked on at a
    time, which keeps each step a small SVD rather than one over all items; this
    is the fast flight phase of Chauvet and Tille (2006).

    Args:
        state: Coordinates in ``[0, 1]``, updated in place.
        constraints: Rows that must be preserved exactly.
        rng: Source of randomness.
    """
    n_constraints = constraints.shape[0]
    pool = list(_unsettled(state))
    if len(pool) < n_constraints + 1:
        return
    working = pool[: n_constraints + 1]
    pool = pool[n_constraints + 1 :]

    # Every step lands at least one coordinate, so the walk cannot outlast the
    # coordinates. The bound is a guard against a pathological direction, not
    # part of the algorithm.
    for _ in range(state.size + n_constraints + 1):
        direction = _null_direction(constraints[:, working])
        if direction is None:
            return
        forward, backward = _step_lengths(state[working], direction)
        if forward + backward <= 0.0:
            return
        if rng.random() < backward / (forward + backward):
            state[working] = state[working] + forward * direction
        else:
            state[working] = state[working] - backward * direction

        # Snap to the faces the step was constructed to reach exactly.
        block = np.asarray(working, dtype=np.intp)
        state[block] = np.clip(state[block], 0.0, 1.0)
        state[block[state[block] <= LANDED]] = 0.0
        state[block[state[block] >= 1.0 - LANDED]] = 1.0

        working = [j for j in working if LANDED < state[j] < 1.0 - LANDED]
        while len(working) < n_constraints + 1 and pool:
            working.append(pool.pop())
        if len(working) < n_constraints + 1:
            return


def cube_sample(
    pi: np.ndarray, constraints: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw a balanced sample whose inclusion probabilities are exactly ``pi``.

    Args:
        pi: Target inclusion probabilities. Must sum to an integer for the sample
            size to be fixed; see :func:`scale_pi_to_budget`.
        constraints: Balancing rows, most dispensable first. The walk preserves
            ``constraints @ state`` exactly for as long as it can, and drops rows
            from the top when it stalls, so the **last** row is the one that
            survives to the end and should be the sample-size constraint.
        rng: Source of randomness.

    Returns:
        np.ndarray: Boolean mask of the selected items.

    Raises:
        ValueError: If ``constraints`` has a column for every item is violated,
            that is, if its width does not match ``pi``.
    """
    if constraints.shape[1] != pi.size:
        raise ValueError(
            f"constraints has {constraints.shape[1]} columns but pi has {pi.size} items"
        )

    state = np.clip(np.asarray(pi, dtype=float).copy(), 0.0, 1.0)
    for first_row in range(constraints.shape[0]):
        _flight_phase(state, constraints[first_row:], rng)
        if _unsettled(state).size == 0:
            break

    # Reached only when the probabilities did not sum to an integer, so no
    # fixed-size design existed to begin with. Settling independently keeps the
    # inclusion probabilities exact and lets the size vary by one.
    stragglers = _unsettled(state)
    if stragglers.size:
        state[stragglers] = (rng.random(stragglers.size) < state[stragglers]).astype(
            float
        )
    return state > 0.5
