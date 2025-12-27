from fewlab import items_to_label

from .data_synth import make_synth


def test_basic_api():
    from fewlab.results import SelectionResult

    counts, X = make_synth(n=50, m=80, p=5, random_state=0)
    budget = 15
    result = items_to_label(counts, X, budget=budget)
    assert isinstance(result, SelectionResult)
    assert len(result.selected) == budget
    # ensure they are item column names
    assert set(result.selected).issubset(set(counts.columns))


def test_zero_rows_are_dropped():
    counts, X = make_synth(n=30, m=40, p=4, random_state=1)
    # force some zero-total rows
    counts.iloc[:3, :] = 0
    result = items_to_label(counts, X, budget=10)
    assert len(result.selected) == 10
