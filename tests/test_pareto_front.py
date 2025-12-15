from arcticroute.core.pareto import ParetoSolution, pareto_front

def test_pareto_front_basic():
    # minimize f1,f2
    cands = [
        ParetoSolution("a", {"f1": 1, "f2": 5}, [], {}, {}),
        ParetoSolution("b", {"f1": 2, "f2": 4}, [], {}, {}),
        ParetoSolution("c", {"f1": 3, "f2": 3}, [], {}, {}),
        ParetoSolution("d", {"f1": 4, "f2": 2}, [], {}, {}),
        ParetoSolution("e", {"f1": 5, "f2": 1}, [], {}, {}),
        ParetoSolution("bad", {"f1": 6, "f2": 6}, [], {}, {}),
    ]
    front = pareto_front(cands, fields=["f1", "f2"])
    keys = sorted([s.key for s in front])
    assert "bad" not in keys
    assert set(keys) == {"a", "b", "c", "d", "e"}
