"""This module extends the syntax base for an agent/type event logic"""

from abc import ABC
from typing import Callable, Iterable, List
from src.logic.base.syntax import (
    AppliedPredicate,
    False_,
    Formula,
    Predicate,
    Sort,
    Term,
)
from src.logic.base.tableau import Tableau


class Sorts(ABC):
    """A collection of sorts in this logic"""

    event = Sort("Event")
    agent = Sort("agent")
    type_ = Sort("type")


class Concepts(ABC):
    """A collection of predicates and concepts"""

    agent = Predicate("agent", 2)
    type_ = Predicate("type", 2)


class Axioms(ABC):
    """A collection of all axioms"""

    @staticmethod
    def single_agent_events(tableau: Tableau) -> Tableau:
        """All events have a max of 1 agents"""
        atoms = filter(
            lambda l: isinstance(l, AppliedPredicate), tableau.branch_literals
        )
        agents: List[AppliedPredicate] = filter(
            lambda a: a.predicate is Concepts.agent, atoms
        )
        seen_events = set()
        for agent in agents:
            event = agent.args[0]
            if event in seen_events:
                # The event is shared by multiple agents
                return Tableau([False_], parent=tableau)
            seen_events.add(event)
        return tableau

    @staticmethod
    def single_type_events(tableau: Tableau) -> Tableau:
        """All events have a max of 1 agents"""
        atoms = filter(
            lambda l: isinstance(l, AppliedPredicate), tableau.branch_literals
        )
        types: List[AppliedPredicate] = filter(
            lambda a: a.predicate is Concepts.type_, atoms
        )
        seen_events = set()
        for type_ in types:
            event = type_.args[0]
            if event in seen_events:
                # The event is shared by multiple types
                return Tableau([False_], parent=tableau)
            seen_events.add(event)
        return tableau

    @staticmethod
    def get_axioms() -> Iterable[Callable[[Tableau], Tableau]]:
        """Return all axioms as a list of callables"""
        return [Axioms.single_agent_events, Axioms.single_type_events]


class Filters:
    """A collection of filter utilities"""

    @staticmethod
    def is_event_term(term: Term) -> bool:
        """Return true if the input term is of type event"""
        return term.sort == Sorts.event

    @staticmethod
    def is_agent_formula(formula: Formula) -> bool:
        """Return true if the input formula is agent(e, a)"""
        return (
            isinstance(formula, AppliedPredicate)
            and formula.predicate is Concepts.agent
        )

    @staticmethod
    def is_type_formula(formula: Formula) -> bool:
        """Return true if the input formula is type(e, a)"""
        return (
            isinstance(formula, AppliedPredicate)
            and formula.predicate is Concepts.type_
        )
