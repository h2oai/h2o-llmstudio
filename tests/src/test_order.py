import pytest

from llm_studio.src.order import Order


def test_order_initialization():
    # Test empty initialization
    order1 = Order()
    assert len(order1) == 0

    # Test initialization with keys
    keys = ["a", "b", "c"]
    order2 = Order(keys)
    assert list(order2) == keys


def test_append():
    order = Order()
    order.append("a")
    assert list(order) == ["a"]

    order.append("b")
    assert list(order) == ["a", "b"]

    with pytest.raises(ValueError):
        order.append("a")  # Attempting to add a duplicate key


def test_extend():
    order = Order(["a", "b"])
    order.extend(["c", "d"])
    assert list(order) == ["a", "b", "c", "d"]

    with pytest.raises(ValueError):
        order.extend(["e", "a"])  # Attempting to add a duplicate key


def test_insert():
    order = Order(["a", "b", "c"])

    order.insert("x", before="b")
    assert list(order) == ["a", "x", "b", "c"]

    order.insert("y", after="c")
    assert list(order) == ["a", "x", "b", "c", "y"]

    order.insert("z", "w", before="a")
    assert list(order) == ["z", "w", "a", "x", "b", "c", "y"]

    with pytest.raises(ValueError):
        order.insert("v", before="non_existent")

    with pytest.raises(ValueError):
        order.insert("v", after="non_existent")

    with pytest.raises(ValueError):
        # Attempting to specify both before and after
        order.insert("v", before="a", after="b")

    with pytest.raises(ValueError):
        order.insert("a")  # Attempting to add a duplicate key

    with pytest.raises(ValueError):
        order.insert("v")  # Not specifying before or after


def test_getitem():
    order = Order(["a", "b", "c"])
    assert order[0] == "a"
    assert order[1] == "b"
    assert order[2] == "c"

    with pytest.raises(IndexError):
        order[3]


def test_len():
    order = Order()
    assert len(order) == 0

    order.append("a")
    assert len(order) == 1

    order.extend(["b", "c"])
    assert len(order) == 3


def test_iter():
    keys = ["a", "b", "c"]
    order = Order(keys)
    assert list(iter(order)) == keys


def test_complex_scenario():
    order = Order(["dataset", "training", "validation", "logging"])
    order.insert("architecture", before="training")
    order.insert("environment", after="validation")

    assert list(order) == [
        "dataset",
        "architecture",
        "training",
        "validation",
        "environment",
        "logging",
    ]

    order.append("results")
    order.extend(["analysis", "reporting"])

    assert list(order) == [
        "dataset",
        "architecture",
        "training",
        "validation",
        "environment",
        "logging",
        "results",
        "analysis",
        "reporting",
    ]

    with pytest.raises(ValueError):
        order.insert("dataset", before="reporting")  # Attempting to add a duplicate key
