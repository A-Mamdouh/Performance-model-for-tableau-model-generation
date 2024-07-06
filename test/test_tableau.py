"""Testing Tableau data structure implementation"""

import itertools
from test import tableau_utils

import src.logic.base.syntax as S
import src.logic.base.tableau as T


class TestTableauMembers:
    """Check if members in Tableau class behave as expected"""

    def test_branch_entities_correct(self) -> None:
        """Check if branch_ members are retrieved properly"""
        # pylint: disable=unbalanced-tuple-unpacking
        second, third, fourth = tableau_utils.create_tableau_chain()[1:]
        # Assert branch entities
        assert list(second.entities) == list(second.branch_entities)
        assert list(itertools.chain(third.entities, second.entities)) == list(
            third.branch_entities
        )
        assert list(
            itertools.chain(fourth.entities, third.entities, second.entities)
        ) == list(fourth.branch_entities)

    def test_branch_formulas_correct(self) -> None:
        """Check if branch_ members are retrieved properly"""
        # pylint: disable=unbalanced-tuple-unpacking
        second, third, fourth = tableau_utils.create_tableau_chain()[1:]
        # Assert branch formulas
        assert list(second.formulas) == list(second.formulas)
        assert list(itertools.chain(third.formulas, second.formulas)) == list(
            third.branch_formulas
        )
        assert list(
            itertools.chain(fourth.formulas, third.formulas, second.formulas)
        ) == list(fourth.branch_formulas)

    def test_branch_literals(self) -> None:
        """Check if branch_ members are retrieved properly"""
        # pylint: disable=unbalanced-tuple-unpacking
        second, third, fourth = tableau_utils.create_tableau_chain()[1:]
        # Assert branch literals
        assert list(second.literals) == list(second.branch_literals)
        assert list(itertools.chain(third.literals, second.literals)) == list(
            third.branch_literals
        )
        assert list(
            itertools.chain(fourth.literals, third.literals, second.literals)
        ) == list(fourth.branch_literals)

    def test_literals_correct(self) -> None:
        """Check if the literals member only contains valid literals"""
        tableau = tableau_utils.create_tableau_chain()[-1]
        assert all(map(S.is_literal, tableau.branch_literals))


class TestTableauMethods:
    """Check if methods in tableau node class behave as expected"""

    def test_tableau_merge_has_all_formulas(self) -> None:
        """Check if output merged tableau has all of the formulas of the merged tableaus"""
        tableaus = tableau_utils.create_tableau_chain(start_empty=False, parent=None)[
            1:
        ]
        merged = T.Tableau.merge(*tableaus)
        merged_formulas = list(merged.formulas)
        base_formulas = list(
            itertools.chain(*map(lambda tableau: tableau.formulas, tableaus))
        )
        for formula in base_formulas:
            assert formula in merged_formulas

    def test_tableau_merge_has_only_base_formulas(self) -> None:
        """Check if output merged tableau has only formulas from the merged tableaus"""
        tableaus = tableau_utils.create_tableau_chain(start_empty=False, parent=None)[
            1:
        ]
        merged = T.Tableau.merge(*tableaus)
        merged_formulas = list(merged.formulas)
        base_formulas = list(
            itertools.chain(*map(lambda tableau: tableau.formulas, tableaus))
        )
        for formula in merged_formulas:
            assert formula in base_formulas

    def test_tableau_merge_no_formula_repeats(self) -> None:
        """Check if output merged tableau has unique formulas"""
        tableaus = tableau_utils.create_tableau_chain(start_empty=False, parent=None)[
            1:
        ]
        merged = T.Tableau.merge(*tableaus)
        merged_formulas = list(merged.formulas)
        assert len(merged_formulas) == len(set(merged_formulas))

    def test_tableau_merge_has_all_entities(self) -> None:
        """Check if output merged tableau has all of the entities of the merged tableaus"""
        tableaus = tableau_utils.create_tableau_chain(start_empty=False, parent=None)[
            1:
        ]
        merged = T.Tableau.merge(*tableaus)
        merged_entities = list(merged.entities)
        base_entities = list(
            itertools.chain(*map(lambda tableau: tableau.entities, tableaus))
        )
        for formula in base_entities:
            assert formula in merged_entities

    def test_tableau_merge_has_only_base_entities(self) -> None:
        """Check if output merged tableau has only entities from the merged tableaus"""
        tableaus = tableau_utils.create_tableau_chain(start_empty=False, parent=None)[
            1:
        ]
        merged = T.Tableau.merge(*tableaus)
        merged_entities = list(merged.entities)
        base_entities = list(
            itertools.chain(*map(lambda tableau: tableau.entities, tableaus))
        )
        for formula in merged_entities:
            assert formula in base_entities

    def test_tableau_merge_no_entities_repeats(self) -> None:
        """Check if output merged tableau has unique entities"""
        tableaus = tableau_utils.create_tableau_chain(start_empty=False, parent=None)[
            1:
        ]
        merged = T.Tableau.merge(*tableaus)
        merged_entities = list(merged.formulas)
        assert len(merged_entities) == len(set(merged_entities))

    def test_tableau_merge_correct_parent(self) -> None:
        """Check if parent parameter is set properly to output merged tableau"""
        parent, *tableaus = tableau_utils.create_tableau_chain(
            start_empty=False, parent=None
        )
        merged = T.Tableau.merge(*tableaus, parent=parent)
        assert merged.parent is parent

        parent, *tableaus = tableau_utils.create_tableau_chain(
            start_empty=False, parent=None
        )
        merged = T.Tableau.merge(*tableaus, parent=None)
        assert merged.parent is None
