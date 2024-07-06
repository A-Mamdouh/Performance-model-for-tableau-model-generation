"""Testing the implementation of the simple event semantics logic"""

import functools
from test import tableau_utils

import src.logic.base.calculus as C
import src.logic.base.syntax as S
from src.logic.simple_event_semantics.syntax import Axioms, Concepts, Sorts
from src.logic.simple_event_semantics.tableau import Tableau


def check_axioms_consistent(tableau: Tableau) -> bool:
    """Apply axioms and check if the tableau is consistent"""
    axioms = Axioms.get_axioms()
    output = functools.reduce(lambda tableau, axiom: axiom(tableau), axioms, tableau)
    return C.is_branch_consistent(output)


class TestSimpleEventSemanticsAxioms:
    """Test axioms and calculus for simple event semantics"""

    def test_event_multiple_agents_fails(self) -> None:
        """Check if contradiction is detected when an event has multiple agents"""
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            tableau = tableaus[i]
            event = Sorts.event.make_constant()
            agent1 = Sorts.agent.make_constant()
            agent2 = Sorts.agent.make_constant()
            tableau.formulas = (
                *tableau.formulas,
                Concepts.agent(event, agent1),
                Concepts.agent(event, agent2),
            )
            # Remove True_ from branch to not trigger other contradiction condition
            for tableau in tableaus:
                tableau.formulas = list(
                    filter(lambda f: f is not S.True_, tableau.formulas)
                )

            assert not any(map(check_axioms_consistent, tableaus[i:]))
        # Check when only the contradiction exists
        for i in range(chain_length):
            tableaus = list(
                tableau_utils.create_tableau_chain(chain_length, formulas=[])
            )
            tableau = tableaus[i]
            event = Sorts.event.make_constant()
            agent1 = Sorts.agent.make_constant()
            agent2 = Sorts.agent.make_constant()
            tableau.formulas = (
                *tableau.formulas,
                Concepts.agent(event, agent1),
                Concepts.agent(event, agent2),
            )
            # Remove True_ from branch to not trigger other contradiction condition
            for tableau in tableaus:
                tableau.formulas = list(
                    filter(lambda f: f is not S.True_, tableau.formulas)
                )
            assert not any(map(check_axioms_consistent, tableaus[i:]))

    def test_event_multiple_types_fails(self) -> None:
        """Check if contradiction is detected when an event has multiple types"""
        chain_length = 4
        for i in range(chain_length):
            tableaus = list(tableau_utils.create_tableau_chain(chain_length))
            tableau = tableaus[i]
            event = Sorts.event.make_constant()
            type1 = Sorts.type_.make_constant()
            type2 = Sorts.type_.make_constant()
            tableau.formulas = (
                *tableau.formulas,
                Concepts.type_(event, type1),
                Concepts.type_(event, type2),
            )
            # Remove True_ from branch to not trigger other contradiction condition
            for tableau in tableaus:
                tableau.formulas = list(
                    filter(lambda f: f is not S.True_, tableau.formulas)
                )
            assert not any(map(check_axioms_consistent, tableaus[i:]))
        # Check when only the contradiction exists
        for i in range(chain_length):
            tableaus = list(
                tableau_utils.create_tableau_chain(chain_length, formulas=[])
            )
            tableau = tableaus[i]
            event = Sorts.event.make_constant()
            type1 = Sorts.type_.make_constant()
            type2 = Sorts.type_.make_constant()
            tableau.formulas = (
                *tableau.formulas,
                Concepts.type_(event, type1),
                Concepts.type_(event, type2),
            )
            # Remove True_ from branch to not trigger other contradiction condition
            for tableau in tableaus:
                tableau.formulas = list(
                    filter(lambda f: f is not S.True_, tableau.formulas)
                )
            assert not any(map(check_axioms_consistent, tableaus[i:]))