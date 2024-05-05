import itertools
from typing import Iterable
import src.calculus as C
import src.syntax as S

from test import tableau_utils


class TestContradictions:
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
