"""Small shared helpers."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _get_random_generator(
    random_state: int | np.random.Generator | None,
) -> np.random.Generator:
    """Create or validate a numpy random generator.

    Args:
        random_state: Random state specification. Can be:
            - None: Creates a new generator with default seed
            - int: Creates a generator with the given seed
            - np.random.Generator: Returns the generator as-is

    Returns:
        A numpy random number generator.

    Examples:
        >>> rng = _get_random_generator(42)
        >>> rng = _get_random_generator(None)
        >>> existing_rng = np.random.default_rng(123)
        >>> rng = _get_random_generator(existing_rng)
    """
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def get_item_positions(
    items: pd.Index | Sequence[str], reference: pd.Index
) -> np.ndarray:
    """Map item names to their positions in a reference index.

    Args:
        items: Item identifiers to map.
        reference: Index containing the full set of items.

    Returns:
        Integer positions of `items` within `reference`.

    Raises:
        ValueError: If any item is missing from the reference index.
    """
    col_to_pos = {col: i for i, col in enumerate(reference)}
    try:
        positions = np.array([col_to_pos[item] for item in items], dtype=int)
    except KeyError as e:
        raise ValueError(f"Item {e} not found in reference index") from e
    return positions
