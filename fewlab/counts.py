"""A count matrix that stays sparse, because real usage data is.

A browsing panel is the motivating case and it sets the shape: 1,134 users by
34,078 domains, 0.27% of cells non-zero. Dense, that is 309 MB of mostly zeros;
as triples it is a few megabytes. Panels arrive in long form anyway -- one row
per user-entity pair -- so this holds them that way and implements only the four
operations the rest of the package needs.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class CountMatrix:
    """Counts in coordinate form, with dense operations computed on demand.

    Attributes:
        rows: Row position of each stored count.
        cols: Column position of each stored count.
        values: The counts themselves.
        index: Unit identifiers, indexed by row position.
        columns: Entity identifiers, indexed by column position.
    """

    rows: np.ndarray
    cols: np.ndarray
    values: np.ndarray
    index: pd.Index
    columns: pd.Index

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix dimensions.

        Returns:
            Units by entities.
        """
        return (len(self.index), len(self.columns))

    @property
    def nnz(self) -> int:
        """Number of stored counts.

        Returns:
            Count of non-zero entries.
        """
        return int(self.values.size)

    @property
    def density(self) -> float:
        """Share of cells that are non-zero.

        Returns:
            Density in [0, 1].
        """
        n, m = self.shape
        return self.nnz / (n * m) if n and m else 0.0

    @classmethod
    def from_long(
        cls,
        frame: pd.DataFrame,
        *,
        unit: str,
        entity: str,
        count: str,
        entities: pd.Index | None = None,
    ) -> "CountMatrix":
        """Build from one row per unit-entity pair.

        Args:
            frame: Long-format counts.
            unit: Column naming the unit.
            entity: Column naming the labelable entity.
            count: Column holding the count.
            entities: Optional full entity frame. Supply it when some entities
                have zero recorded activity but should still belong to the
                population; they contribute nothing to the estimand and will be
                assigned zero probability, which costs no bias.

        Returns:
            The sparse count matrix.

        Raises:
            ValueError: If counts are negative.
        """
        values = np.asarray(frame[count], dtype=float)
        if (values < 0).any():
            raise ValueError("counts must be non-negative")
        unit_values = np.asarray(frame[unit])
        entity_values = np.asarray(frame[entity])
        index = pd.Index(pd.unique(unit_values), name=unit).sort_values()
        columns = (
            pd.Index(entities, name=entity)
            if entities is not None
            else pd.Index(pd.unique(entity_values), name=entity).sort_values()
        )
        rows = index.get_indexer(unit_values)
        cols = columns.get_indexer(entity_values)
        if (cols < 0).any():
            raise ValueError("frame contains entities absent from `entities`")
        return cls(rows=rows, cols=cols, values=values, index=index, columns=columns)

    @classmethod
    def from_dense(cls, frame: pd.DataFrame) -> "CountMatrix":
        """Build from a dense DataFrame.

        Args:
            frame: Dense count matrix, units by entities.

        Returns:
            The sparse count matrix, keeping only non-zero cells.
        """
        arr = frame.to_numpy(float)
        rows, cols = np.nonzero(arr)
        return cls(
            rows=rows,
            cols=cols,
            values=arr[rows, cols],
            index=frame.index,
            columns=frame.columns,
        )

    def row_totals(self) -> np.ndarray:
        """Sum each unit's counts across entities.

        Returns:
            Row totals, one per unit.
        """
        return np.bincount(self.rows, weights=self.values, minlength=self.shape[0])

    def project(self, weights: np.ndarray) -> np.ndarray:
        """Compute `weights' C`, the entity-wise projection of unit weights.

        With `weights = X / T` this is `G = X' V`, the only contraction of the
        count matrix the design needs, and it costs one pass over the stored
        entries rather than a dense multiply.

        Args:
            weights: `(n, p)` array of per-unit weights.

        Returns:
            The `(p, m)` projection.
        """
        m = self.shape[1]
        p = weights.shape[1]
        out = np.zeros((m, p))
        np.add.at(out, self.cols, weights[self.rows] * self.values[:, None])
        return out.T

    def column_sq_norms(self, scale: np.ndarray) -> np.ndarray:
        """Compute `sum_i (scale_i * C_ij)^2` for every entity.

        With `scale = 1/T` this is `||v_j||^2`, the total squared share entity
        `j` accounts for across units. It is the piece the design needs when the
        label enters a regression as a covariate rather than as the outcome,
        because there the identifying variation is the part of `v_j` orthogonal
        to the controls rather than the part aligned with them.

        Args:
            scale: Per-unit multiplier, one entry per unit.

        Returns:
            One squared norm per entity.
        """
        weighted = scale[self.rows] * self.values
        return np.bincount(self.cols, weights=weighted**2, minlength=self.shape[1])

    def dense_columns(self, positions: np.ndarray) -> np.ndarray:
        """Materialize a subset of columns.

        Args:
            positions: Column positions to extract.

        Returns:
            An `(n, len(positions))` dense array.
        """
        n = self.shape[0]
        lookup = np.full(self.shape[1], -1, dtype=np.int64)
        lookup[positions] = np.arange(positions.size)
        take = lookup[self.cols] >= 0
        out = np.zeros((n, positions.size))
        out[self.rows[take], lookup[self.cols[take]]] = self.values[take]
        return out

    def select_rows(self, keep: np.ndarray) -> "CountMatrix":
        """Drop units.

        Args:
            keep: Boolean mask over units.

        Returns:
            A matrix restricted to the kept units.
        """
        remap = np.full(self.shape[0], -1, dtype=np.int64)
        remap[np.flatnonzero(keep)] = np.arange(int(keep.sum()))
        take = remap[self.rows] >= 0
        return CountMatrix(
            rows=remap[self.rows[take]],
            cols=self.cols[take],
            values=self.values[take],
            index=self.index[keep],
            columns=self.columns,
        )


def as_counts(counts: "pd.DataFrame | CountMatrix") -> CountMatrix:
    """Accept either representation.

    Args:
        counts: A dense DataFrame or an already-sparse matrix.

    Returns:
        The sparse representation.
    """
    return counts if isinstance(counts, CountMatrix) else CountMatrix.from_dense(counts)
