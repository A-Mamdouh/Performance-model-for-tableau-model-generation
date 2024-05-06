"""Tests for calculus implementaion"""

import itertools
from test import tableau_utils
from typing import Iterable, List, Set

import src.calculus as C
import src.syntax as S


class TestContradictions:
    """Test contradiction rule"""

    def test_no_contradiction_ok_branch(self) -> None:
        """Check if no contradiction when passed an ok branch"""
        assert not any(map(C.check_contradiction, tableau_utils.create_tableau_chain()))

    def test_false_creates_contradiction(self) -> None:
        """Check if contradiction is detected when false is in the branch"""
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            tableau = tableaus[i]
            tableau.formulas = *tableau.formulas, S.False_
            assert all(map(C.check_contradiction, tableaus[i:]))
        # Check when there is only the False Formula
        for i in range(chain_length):
            tableaus = list(
                tableau_utils.create_tableau_chain(chain_length, formulas=[])
            )
            tableau = tableaus[i]
            tableau.formulas = *tableau.formulas, S.False_
            assert all(map(C.check_contradiction, tableaus[i:]))

    def test_not_true_creates_contradiction(self) -> None:
        """Check if contradiction is detected when -True is in the branch"""
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            tableau = tableaus[i]
            tableau.formulas = *tableau.formulas, S.Not(S.True_)
            # Remove True_ from branch to not trigger other contradiction condition
            for tableau in tableaus:
                tableau.formulas = list(
                    filter(lambda f: f is not S.True_, tableau.formulas)
                )
            assert all(map(C.check_contradiction, tableaus[i:]))
        # Check when there is only the -True Formula
        for i in range(chain_length):
            tableaus = list(
                tableau_utils.create_tableau_chain(chain_length, formulas=[])
            )
            tableau = tableaus[i]
            tableau.formulas = *tableau.formulas, S.Not(S.True_)
            # Remove True_ from branch to not trigger other contradiction condition
            for tableau in tableaus:
                tableau.formulas = list(
                    filter(lambda f: f is not S.True_, tableau.formulas)
                )
            assert all(map(C.check_contradiction, tableaus[i:]))

    def test_f_and_not_f_contradiction(self) -> None:
        """Check if contradiction is detected when a formula and Not(formula) exist"""
        chain_length = 4
        for i in range(chain_length):
            for j in range(chain_length):
                tableaus = list(tableau_utils.create_tableau_chain(chain_length))
                formula1 = S.Or(S.True_, S.False_)
                formula2 = S.Not(S.Or(S.True_, S.False_))
                tableaus[i].formulas = *tableaus[i].formulas, formula1
                tableaus[j].formulas = *tableaus[j].formulas, formula2
        # Check again with just the contradiction in the branch
        for i in range(chain_length):
            for j in range(chain_length):
                tableaus = list(
                    tableau_utils.create_tableau_chain(chain_length, formulas=[])
                )
                formula1 = S.Or(S.True_, S.False_)
                formula2 = S.Not(S.Or(S.True_, S.False_))
                tableaus[i].formulas = *tableaus[i].formulas, formula1
                tableaus[j].formulas = *tableaus[j].formulas, formula2

    def test_event_multiple_agents_fails(self) -> None:
        """Check if contradiction is detected when an event has multiple agents"""
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            tableau = tableaus[i]
            event = S.Constant.Event()
            agent1 = S.Constant.Agent()
            agent2 = S.Constant.Agent()
            tableau.formulas = (
                *tableau.formulas,
                S.Agent(event, agent1),
                S.Agent(event, agent2),
            )
            # Remove True_ from branch to not trigger other contradiction condition
            for tableau in tableaus:
                tableau.formulas = list(
                    filter(lambda f: f is not S.True_, tableau.formulas)
                )
            assert all(map(C.check_contradiction, tableaus[i:]))
        # Check when only the contradiction exists
        for i in range(chain_length):
            tableaus = list(
                tableau_utils.create_tableau_chain(chain_length, formulas=[])
            )
            tableau = tableaus[i]
            event = S.Constant.Event()
            agent1 = S.Constant.Agent()
            agent2 = S.Constant.Agent()
            tableau.formulas = (
                *tableau.formulas,
                S.Agent(event, agent1),
                S.Agent(event, agent2),
            )
            # Remove True_ from branch to not trigger other contradiction condition
            for tableau in tableaus:
                tableau.formulas = list(
                    filter(lambda f: f is not S.True_, tableau.formulas)
                )
            assert all(map(C.check_contradiction, tableaus[i:]))

    def test_event_multiple_types_fails(self) -> None:
        """Check if contradiction is detected when an event has multiple types"""
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            tableau = tableaus[i]
            event = S.Constant.Event()
            type1 = S.Constant.Type()
            type2 = S.Constant.Type()
            tableau.formulas = (
                *tableau.formulas,
                S.Type_(event, type1),
                S.Type_(event, type2),
            )
            # Remove True_ from branch to not trigger other contradiction condition
            for tableau in tableaus:
                tableau.formulas = list(
                    filter(lambda f: f is not S.True_, tableau.formulas)
                )
            assert all(map(C.check_contradiction, tableaus[i:]))
        # Check when only the contradiction exists
        for i in range(chain_length):
            tableaus = list(
                tableau_utils.create_tableau_chain(chain_length, formulas=[])
            )
            tableau = tableaus[i]
            event = S.Constant.Event()
            type1 = S.Constant.Type()
            type2 = S.Constant.Type()
            tableau.formulas = (
                *tableau.formulas,
                S.Type_(event, type1),
                S.Type_(event, type2),
            )
            # Remove True_ from branch to not trigger other contradiction condition
            for tableau in tableaus:
                tableau.formulas = list(
                    filter(lambda f: f is not S.True_, tableau.formulas)
                )
            assert all(map(C.check_contradiction, tableaus[i:]))

    def test_equality_contradictions(self) -> None:
        """Check if incorrect equalities are detected"""
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            c1 = S.Constant.Agent()
            c2 = S.Constant.Agent()
            tableaus[i].formulas = *tableaus[i].formulas, S.Eq(c1, c2)
            assert C.check_contradiction(tableaus[i])

    def test_inequality_contradictions(self) -> None:
        """Check if incorrect inequalities are detected"""
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            c1 = S.Constant.Agent()
            tableaus[i].formulas = *tableaus[i].formulas, S.Not(S.Eq(c1, c1))
            assert C.check_contradiction(tableaus[i])


class TestConjunctionElim:
    """Test and_elim rule"""

    def test_only_current_leaf_is_checked_for_conjunctions(self) -> None:
        """Make sure only conjuncts from the passed node are resolved"""
        p = S.Predicate("P", 1)
        formulas = [p(S.Constant.Agent()) for _ in range(2)]
        conjunctions = [S.And(a, b) for a, b in itertools.product(formulas, formulas)]
        formulas2 = [p(S.Constant.Agent()) for _ in range(2)]
        conjunctions2 = [S.And(a, b) for a, b in itertools.product(formulas, formulas)]
        chain_length = 4
        for i in range(chain_length):
            for j in range(chain_length):
                if j == i:
                    continue
                tableaus = list(tableau_utils.create_tableau_chain(chain_length))
                # Add conjunctions and remove existing ones
                tableaus[i].formulas = (
                    filter(lambda f: not isinstance(f, S.And), tableaus[i].formulas),
                    *conjunctions,
                )
                tableaus[j].formulas = (
                    filter(lambda f: not isinstance(f, S.And), tableaus[i].formulas),
                    *conjunctions2,
                )
                output = C.try_and_elim(tableaus[i])
                assert all(formula not in output.formulas for formula in formulas2)

    def test_only_unique_formluas_are_added(self) -> None:
        """Check if new formulas are checked for uniqueness against each other"""
        p = S.Predicate("P", 1)
        formulas = [p(S.Constant.Agent()) for _ in range(2)]
        conjunctions = [S.And(a, b) for a, b in itertools.product(formulas, formulas)]
        chain_length = 4
        for i in range(1, chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            # Add conjunctions and remove existing ones
            tableaus[i].formulas = (
                filter(lambda f: not isinstance(f, S.And), tableaus[i].formulas),
                *conjunctions,
            )
            tableaus[i - 1].formulas = *tableaus[i - 1].formulas, formulas[0]
            output = C.try_and_elim(tableaus[i])
            assert len(list(output.formulas)) == len(set(output.formulas))

    def test_only_branch_unique_formluas_are_added(self) -> None:
        """Check if new formulas are checked for uniqueness against the branch formulas"""
        p = S.Predicate("P", 1)
        formulas = [p(S.Constant.Agent()) for _ in range(2)]
        conjunctions = [S.And(a, b) for a, b in itertools.product(formulas, formulas)]
        chain_length = 4
        for i in range(1, chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            # Add conjunctions and remove existing ones
            tableaus[i].formulas = (
                filter(lambda f: not isinstance(f, S.And), tableaus[i].formulas),
                *conjunctions,
            )
            tableaus[i - 1].formulas = *tableaus[i - 1].formulas, formulas[0]
            output = C.try_and_elim(tableaus[i])
            assert formulas[0] not in output.formulas

    def test_all_conjuncts_are_resolved(self) -> None:
        """Check if all conjuncts are produced, and output tableau only has the conjuncts"""
        p = S.Predicate("P", 1)
        formulas = [p(S.Constant.Agent()) for _ in range(2)]
        conjunctions = [S.And(a, b) for a, b in itertools.product(formulas, formulas)]
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            # Add conjunctions and remove existing ones
            tableaus[i].formulas = (
                filter(lambda f: not isinstance(f, S.And), tableaus[i].formulas),
                *conjunctions,
            )
            output = C.try_and_elim(tableaus[i])
            assert output is not None
            # Only the conjuncts exist
            assert all(formula in formulas for formula in output.formulas)
            # All conjuncts exist
            assert all(formula in output.formulas for formula in formulas)

    def test_output_tableau_parent_is_input_tableau(self) -> None:
        """Check if the parent of the output tableau is the input tableau"""
        p = S.Predicate("P", 1)
        formulas = [p(S.Constant.Agent()) for _ in range(1)]
        conjunctions = [S.And(a, b) for a, b in itertools.product(formulas, formulas)]
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            # Add conjunctions
            tableaus[i].formulas = *tableaus[i].formulas, *conjunctions
            output = C.try_and_elim(tableaus[i])
            assert output.parent is tableaus[i]

    def test_returns_none_when_passed_empty(self) -> None:
        """Check if None is returned when the node has no formulas"""
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            # Empty out formulas
            tableaus[i].formulas = []
            assert C.try_and_elim(tableaus[i]) is None

    def test_returns_none_when_passed_no_conjunctions(self) -> None:
        """Check if None is returned when the node has no conjunctions"""
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            # Filter out conjunctions
            tableaus[i].formulas = filter(
                lambda f: not isinstance(f, S.And), tableaus[i].formulas
            )
            assert C.try_and_elim(tableaus[i]) is None


class TestDoubleNegation:
    """Test d_neg rule"""

    @staticmethod
    def _remove_double_negations(formulas: Iterable[S.Formula]) -> Iterable[S.Formula]:
        return filter(
            lambda f: not (isinstance(f, S.Not) and isinstance(f.formula, S.Not)),
            formulas,
        )

    def test_double_negation_removed(self) -> None:
        """Test that double nedation removes double negations"""
        p = S.Predicate("P", 1)
        formulas = [p(S.Constant.Agent()) for _ in range(1)]
        d_negations = [S.Not(S.Not(f)) for f in formulas]
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            # Add double negations and remove existing ones
            tableaus[i].formulas = (
                *self._remove_double_negations(tableaus[i].formulas),
                *d_negations,
            )
            output = C.try_double_negation(tableaus[i])
            # Check that the double negated formulas are present
            assert all(formula in output.formulas for formula in formulas)
            # Check that only the double negated formulas are present
            assert all(formula in formulas for formula in output.formulas)

    def test_double_negation_returns_none_when_no_double_negations(self) -> None:
        """Check that None is returned when no double negations are on node"""
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            # Remove double negations
            tableaus[i].formulas = self._remove_double_negations(tableaus[i].formulas)
            output = C.try_double_negation(tableaus[i])
            assert output is None
            # Try again with empty tableaus
            tableaus = list(
                tableau_utils.create_tableau_chain(chain_length, formulas=[])
            )
            # Remove double negations
            tableaus[i].formulas = self._remove_double_negations(tableaus[i].formulas)
            output = C.try_double_negation(tableaus[i])
            assert output is None

    def test_double_negation_correct_parent(self) -> None:
        """Check that the output tableau has the correct parent set"""
        p = S.Predicate("P", 1)
        formulas = [p(S.Constant.Agent()) for _ in range(1)]
        d_negations = [S.Not(S.Not(f)) for f in formulas]
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            # Add double negations and remove existing ones
            tableaus[i].formulas = (
                *self._remove_double_negations(tableaus[i].formulas),
                *d_negations,
            )
            output = C.try_double_negation(tableaus[i])
            assert output.parent is tableaus[i]

    def test_double_negation_unique_formulas(self) -> None:
        """Check that the output tableau has unique formulas (no branch or local duplicates)"""
        p = S.Predicate("P", 1)
        formulas = [p(S.Constant.Agent()) for _ in range(1)]
        d_negations = [S.Not(S.Not(f)) for f in formulas]
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            # Add double negations and remove existing ones
            tableaus[i].formulas = (
                *self._remove_double_negations(tableaus[i].formulas),
                *d_negations,
            )
            output = C.try_double_negation(tableaus[i])
            assert len(set(output.formulas)) == len(output.formulas)
            assert len(
                set(output.formulas).difference(tableaus[i].branch_formulas)
            ) == len(output.formulas)


class TestDisjunctionElim:
    """Test or_elim rule"""

    def test_only_current_node_is_checked_for_disjunctions(self) -> None:
        """Make sure only conjuncts from the passed node are resolved"""
        p = S.Predicate("P", 1)
        formulas = [p(S.Constant.Agent()) for _ in range(2)]
        disjunctions = [S.Or(a, b) for a, b in itertools.product(formulas, formulas)]
        formulas2 = [p(S.Constant.Agent()) for _ in range(2)]
        disjunctions2 = [S.Or(a, b) for a, b in itertools.product(formulas, formulas)]
        chain_length = 4
        for i in range(chain_length):
            for j in range(chain_length):
                if j == i:
                    continue
                tableaus = list(tableau_utils.create_tableau_chain(chain_length))
                # Add disjunctions and remove existing ones
                tableaus[i].formulas = (
                    filter(lambda f: not isinstance(f, S.Or), tableaus[i].formulas),
                    *disjunctions,
                )
                tableaus[j].formulas = (
                    filter(lambda f: not isinstance(f, S.Or), tableaus[i].formulas),
                    *disjunctions2,
                )
                output = C.try_or_elim(tableaus[i])
                for tableau in output:
                    assert all(formula not in tableau.formulas for formula in formulas2)

    def test_only_unique_formluas_are_added(self) -> None:
        """Check if new formulas are checked for uniqueness against each other"""
        p = S.Predicate("P", 1)
        formulas = [p(S.Constant.Agent()) for _ in range(2)]
        disjunctions = [S.Or(a, b) for a, b in itertools.product(formulas, formulas)]
        chain_length = 4
        for i in range(1, chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            # Add disjunctions and remove existing ones
            tableaus[i].formulas = (
                filter(lambda f: not isinstance(f, S.Or), tableaus[i].formulas),
                *disjunctions,
            )
            tableaus[i - 1].formulas = *tableaus[i - 1].formulas, formulas[0]
            output = C.try_or_elim(tableaus[i])
            for tableau in output:
                assert len(list(tableau.formulas)) == len(set(tableau.formulas))

    def test_only_branch_unique_formluas_are_added(self) -> None:
        """Check if new formulas are checked for uniqueness against the branch formulas"""
        p = S.Predicate("P", 1)
        formulas = [p(S.Constant.Agent()) for _ in range(2)]
        disjunctions = [S.Or(a, b) for a, b in itertools.product(formulas, formulas)]
        chain_length = 4
        for i in range(1, chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            # Add disjunctions and remove existing ones
            tableaus[i].formulas = (
                filter(lambda f: not isinstance(f, S.Or), tableaus[i].formulas),
                *disjunctions,
            )
            tableaus[i - 1].formulas = *tableaus[i - 1].formulas, formulas[0]
            output = C.try_or_elim(tableaus[i])
            for tableau in output:
                assert formulas[0] not in tableau.formulas

    def test_all_disjuncts_are_resolved(self) -> None:
        """Check if all conjuncts are produced, and output tableau only has the conjuncts"""
        # pylint: disable=too-many-locals
        p = S.Predicate("P", 1)
        formulas = [p(S.Constant.Agent()) for _ in range(4)]
        # Only add unique pairs (no order)
        disjunctions: List[S.Or] = []
        for i, left in enumerate(formulas):
            for j in range(i + 1, len(formulas)):
                disjunctions.append(S.Or(left, formulas[j]))
        # Get all possible unique branches. All possible branches are 2^#(disjunctions)
        expected_outputs: List[Set[S.Formula]] = []
        for i in range(int(2 ** len(disjunctions))):
            branch = set()
            for j, disjunction in enumerate(disjunctions):
                if (1 << j) & i:
                    branch.add(disjunction.left)
                else:
                    branch.add(disjunction.right)
            if branch not in expected_outputs:
                expected_outputs.append(branch)
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            # Add disjunctions and remove existing ones
            tableaus[i].formulas = (
                filter(lambda f: not isinstance(f, S.And), tableaus[i].formulas),
                *disjunctions,
            )
            output = C.try_or_elim(tableaus[i])
            assert output is not None
            seen_outputs = []
            for tableau in output:
                # Only the disjuncts exist
                assert all(formula in formulas for formula in tableau.formulas)
                # Output matches one of the expected outputs
                formulas_set = set(tableau.formulas)
                assert formulas_set in expected_outputs
                assert formulas_set not in seen_outputs
                seen_outputs.append(formulas_set)
            # Check if all expected outputs were covered
            assert len(seen_outputs) == len(expected_outputs)

    def test_output_tableau_parent_is_input_tableau(self) -> None:
        """Check if the parent of the output tableau is the input tableau"""
        p = S.Predicate("P", 1)
        formulas = [p(S.Constant.Agent()) for _ in range(1)]
        disjunctions = [S.Or(a, b) for a, b in itertools.product(formulas, formulas)]
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            # Add disjunctions
            tableaus[i].formulas = *tableaus[i].formulas, *disjunctions
            output = C.try_or_elim(tableau=tableaus[i])
            for tableau in output:
                assert tableau.parent is tableaus[i]

    def test_returns_none_when_passed_empty(self) -> None:
        """Check if None is returned when the node has no formulas"""
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            # Empty out formulas
            tableaus[i].formulas = []
            assert C.try_or_elim(tableaus[i]) is None

    def test_returns_none_when_passed_no_conjunctions(self) -> None:
        """Check if None is returned when the node has no conjunctions"""
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            # Filter out conjunctions
            tableaus[i].formulas = filter(
                lambda f: not isinstance(f, S.Or), tableaus[i].formulas
            )
            assert C.try_or_elim(tableaus[i]) is None


class TestForAll:
    """Test ForAll elim rule"""

    def test_returns_none_on_empty_tableau(self) -> None:
        """Test if forall-elim rule returns None when the input tableau has no formulas"""
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(
                tableau_utils.create_tableau_chain(chain_length, formulas=[])
            )
            # Remove formulas
            tableaus[i].formulas = []
            assert C.try_forall_elim(tableaus[i]) is None

    def test_returns_none_on_tableau_with_no_foralls(self) -> None:
        """Test if forall-elim rule returns None when the input tableau has no forall formulas"""
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            # Add axioms and remove existing ones
            for tableau in tableaus:
                tableau.formulas = filter(
                    lambda f: not isinstance(f, S.Forall), tableau.formulas
                )
            assert C.try_forall_elim(tableaus[i]) is None

    def test_returns_all_instances_of_axioms(self) -> None:
        """Test if all entities of the axiom sort are instantiated"""
        chain_length = 4
        axioms = [
            S.Forall(S.Predicate(f"p_{i}", 1), S.Term.Sort.EVENT) for i in range(4)
        ]
        for i in range(chain_length):
            for j in range(i + 1, chain_length):
                tableaus = list(tableau_utils.create_tableau_chain(chain_length))
                # Add axioms and remove existing ones
                for tableau in tableaus:
                    tableau.formulas = filter(
                        lambda f: not isinstance(f, S.Forall), tableau.formulas
                    )
                tableaus[i].formulas = *tableaus[i].formulas, *axioms
                output = C.try_forall_elim(tableaus[j])
                expected_output = []
                for axiom in axioms:
                    # pylint: disable=cell-var-from-loop
                    expected_output.extend(
                        map(
                            axiom.partial_formula,
                            filter(
                                lambda e: e.sort == axiom.sort,
                                tableaus[j].branch_entities,
                            ),
                        )
                    )
                # Assert that all entities of the axiom sort are produced
                # pylint: disable=cell-var-from-loop
                assert all(map(lambda e: e in output.formulas, expected_output))
                # Assert that only the axiom productions are in the output
                assert all(map(lambda e: e in expected_output, output.formulas))


class TestForAllFocused:
    """Test ForAllF elim rule"""

    def test_returns_none_on_empty_tableau(self) -> None:
        """Test if focused-forall-elim rule returns None when the input tableau has no formulas"""
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(
                tableau_utils.create_tableau_chain(chain_length, formulas=[])
            )
            # Add disjunctions and remove existing ones
            tableaus[i].formulas = []
            assert C.try_focused_forall_elim(tableaus[i]) is None

    def test_returns_none_on_tableau_with_no_foralls(self) -> None:
        """Test if focusedforall-elim rule returns None,
        when the input tableau has no forall formulas TODO
        """
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            # Remove focused axioms
            for tableau in tableaus:
                tableau.formulas = filter(
                    lambda f: not isinstance(f, S.ForallF), tableau.formulas
                )
            assert C.try_focused_forall_elim(tableaus[i]) is None
