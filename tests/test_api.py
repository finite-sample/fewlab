import pandas as pd
from fewlab import items_to_label
from .data_synth import make_synth

def test_basic_api():
    counts, X = make_synth(n=50, m=80, p=5, seed=0)
    K = 15
    chosen = items_to_label(counts, X, K=K)
    assert isinstance(chosen, list)
    assert len(chosen) == K
    # ensure they are item column names
    assert set(chosen).issubset(set(counts.columns))

def test_zero_rows_are_dropped():
    counts, X = make_synth(n=30, m=40, p=4, seed=1)
    # force some zero-total rows
    counts.iloc[:3, :] = 0
    chosen = items_to_label(counts, X, K=10)
    assert len(chosen) == 10
