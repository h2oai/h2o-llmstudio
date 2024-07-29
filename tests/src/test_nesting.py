import unittest

import pytest

from llm_studio.src.nesting import Dependency, Nesting


class TestDependency:
    @pytest.mark.parametrize(
        "key, value, is_set",
        [
            ("personalize", True, True),
            ("validation_strategy", "automatic", True),
            ("deepspeed_method", "ZeRO2", True),
            ("lora", False, False),
        ],
    )
    def test_dependency_init(self, key, value, is_set):
        dep = Dependency(key=key, value=value, is_set=is_set)
        assert dep.key == key
        assert dep.value == value
        assert dep.is_set == is_set

    @pytest.mark.parametrize(
        "dep, dependency_values, expected",
        [
            (Dependency("tkey", value=True, is_set=True), [True], True),
            (Dependency("tkey", value=True, is_set=True), [False], False),
            (Dependency("tkey", value=True, is_set=False), [True], False),
            (Dependency("tkey", value=True, is_set=False), [False], True),
            (Dependency("tkey", value=False, is_set=True), [False], True),
            (Dependency("tkey", value=False, is_set=True), [True], False),
            (Dependency("tkey", value=False, is_set=False), [False], False),
            (Dependency("tkey", value=False, is_set=False), [True], True),
            (Dependency("tkey", value="value", is_set=True), ["value"], True),
            (Dependency("tkey", value="value", is_set=True), ["other_value"], False),
            (Dependency("tkey", value="value", is_set=False), ["value"], False),
            (Dependency("tkey", value="value", is_set=False), ["other_value"], True),
            (Dependency("tkey", value=None, is_set=True), [], False),
            (Dependency("tkey", value=None, is_set=True), ["value"], False),
            (Dependency("tkey", value=None, is_set=False), [], False),
            (Dependency("tkey", value=None, is_set=False), ["value"], True),
        ],
    )
    def test_dependency_check(self, dep, dependency_values, expected):
        assert dep.check(dependency_values) == expected


class TestNesting(unittest.TestCase):
    def setUp(self):
        self.nesting = Nesting()

    def test_nesting_init(self):
        self.assertEqual(len(self.nesting.dependencies), 0)
        self.assertEqual(len(self.nesting.triggers), 0)

    def test_nesting_add(self):
        keys = ["key1", "key2"]
        dependencies = [
            Dependency("dep1", value=True, is_set=True),
            Dependency("dep2", value=True, is_set=True),
        ]
        self.nesting.add(keys, dependencies)

        self.assertEqual(len(self.nesting.dependencies), 2)
        self.assertEqual(len(self.nesting.triggers), 2)
        self.assertIn("dep1", self.nesting.triggers)
        self.assertIn("dep2", self.nesting.triggers)

    def test_nesting_add_duplicate_keys(self):
        keys = ["key1", "key1"]
        dependencies = [Dependency("dep1", value=True, is_set=True)]

        with self.assertRaises(ValueError):
            self.nesting.add(keys, dependencies)

    def test_nesting_multiple_adds(self):
        self.nesting.add(["key1"], [Dependency("dep1", value=True, is_set=True)])
        self.nesting.add(["key2"], [Dependency("dep2", value=True, is_set=True)])
        self.nesting.add(
            ["key1", "key2"], [Dependency("dep3", value=True, is_set=True)]
        )

        self.assertEqual(len(self.nesting.dependencies), 2)
        self.assertEqual(len(self.nesting.triggers), 3)
        self.assertEqual(len(self.nesting.dependencies["key1"]), 2)
        self.assertEqual(len(self.nesting.dependencies["key2"]), 2)
