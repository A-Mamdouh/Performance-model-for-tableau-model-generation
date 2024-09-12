"""Environment details"""

from dataclasses import dataclass
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from src.logic.base.calculus import generate_models
from src.logic.base.syntax import (
    AppliedPredicate,
    Constant,
    Eq,
    Exists,
    Predicate,
    Sort,
    Term,
    is_literal,
)
from src.logic.base.tableau import Axiom, Tableau
from src.query_environment.embeddings import EventInformationManager
from src.query_environment.learned_heuristic import HeuristicModel


class Sorts:
    """Domain sorts"""

    agent = Sort("agent")
    action = Sort("action")
    event = Sort("event")


class Predicates:
    """Domain relations/concepts"""

    subject = Predicate("subject", 2, (Sorts.event, Sorts.agent))
    action = Predicate("action", 2, sorts=(Sorts.event, Sorts.action))
    object = Predicate("object", 2, (Sorts.event, Sorts.agent))


class Constants:
    """Domain individuals"""

    alex = Sorts.agent.make_constant("alex")
    bob = Sorts.agent.make_constant("bob")
    charlie = Sorts.agent.make_constant("charlie")
    diana = Sorts.agent.make_constant("diana")

    fido = Sorts.agent.make_constant("fido")

    read = Sorts.action.make_constant("read")
    walk = Sorts.action.make_constant("walk")
    run = Sorts.action.make_constant("run")
    eat = Sorts.action.make_constant("eat")
    pet = Sorts.action.make_constant("pet")
    bite = Sorts.action.make_constant("bite")

    @classmethod
    def get_constant(cls, name: str, sort: Sort | None = None) -> Constant:
        """Get a constant using its name and an optional sort"""
        for key, value in cls.__dict__.items():
            if key == name:
                if not sort or sort is value.sort:
                    return value
        raise KeyError()


@dataclass
class Verb:
    """Verb information wrapper"""

    inf: str
    past: str


class Sentence:
    """Natural language sentence duck type/protocol"""

    def __init__(
        self, str_repr: str, get_readings: Callable[[], Iterable[Tableau]]
    ) -> None:
        self._get_readings = get_readings
        self._str_repr = str_repr

    def __str__(self) -> str:
        return self._str_repr

    def get_readings(self) -> Iterable[Tableau]:
        return self._get_readings()


def noun_verb_sentence(subject: str, verb: Verb) -> Sentence:
    """Create a sentence of the form subject did verb"""
    str_repr = f"{subject} {verb.past}"
    c_subject = Constants.get_constant(subject, Sorts.agent)
    c_action = Constants.get_constant(verb.inf, Sorts.action)

    def get_readings() -> Iterable[Tableau]:
        return (
            Tableau(
                [
                    Exists(
                        lambda e: Predicates.subject(e, c_subject)
                        & Predicates.action(e, c_action),
                        sort=Sorts.event,
                    )
                ],
                [c_subject, c_action],
            ),
        )

    return Sentence(str_repr, get_readings)


class Axioms:
    """Environment axioms"""

    @staticmethod
    def axiom_only_one_object(tableau: Tableau) -> Tableau | None:
        """Only one object per event"""
        return Axioms._only_one_kind_per_event(Predicates.object)(tableau)

    @staticmethod
    def axiom_only_one_subject(tableau: Tableau) -> Tableau | None:
        """Only one object per event"""
        return Axioms._only_one_kind_per_event(Predicates.subject)(tableau)

    @staticmethod
    def axiom_only_one_action(tableau: Tableau) -> Tableau | None:
        """Only one object per event"""
        return Axioms._only_one_kind_per_event(Predicates.action)(tableau)

    @staticmethod
    def _only_one_kind_per_event(pred: Predicate) -> Axiom:
        def axiom(tableau: Tableau) -> Tableau | None:
            pred_by_event: Dict[Term, List[Tuple]] = {}
            for literal in tableau.branch_literals:
                if not isinstance(literal, AppliedPredicate):
                    continue
                if literal.predicate is not pred:
                    continue
                event, *terms = literal.args
                equals = pred_by_event.get(event)
                if equals is None:
                    pred_by_event[event] = equals = []
                equals.append(terms)
            # Create equalities between terms
            f_equalities = set()
            for applications in pred_by_event.values():
                for equal_terms in zip(*applications):
                    for t1, t2 in itertools.product(equal_terms, equal_terms):
                        if t1 is t2:
                            continue
                        f_equalities.add(Eq(t1, t2))
            output = tableau.get_unique_tableau(Tableau(f_equalities, parent=tableau))
            if output.formulas:
                return output
            return None

        return axiom

    @classmethod
    def get_axioms(cls) -> List[Axiom]:
        """Return list of axiom callables"""
        output = [
            value for key, value in cls.__dict__.items() if key.startswith("axiom_")
        ]
        return output

