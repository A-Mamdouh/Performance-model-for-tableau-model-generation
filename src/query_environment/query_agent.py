"""Simple knowledge base agent"""

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
)
from src.logic.base.tableau import Axiom, Tableau


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


@dataclass
class AgentNode:
    """Search tree node"""

    tableau: Tableau
    depth: int
    parent: Optional["AgentNode"] = None

    @classmethod
    def make_initial(cls) -> "AgentNode":
        """Create an empty node"""
        return AgentNode(Tableau(), 0, None)


class Agent:
    """Logical agent"""

    def __init__(self):
        self.knowledge_base: List[Tableau] = []
        self._models: List[AgentNode] = [AgentNode.make_initial()]

    def add_information(self, tableau: Tableau) -> Optional[Tableau]:
        """Add world knowledge to the agent"""
        parent = None
        if self.knowledge_base:
            parent = self.knowledge_base[-1]
        tableau = tableau.merge(
            tableau, parent=parent
        )
        self.knowledge_base.append(tableau)
        while self._models:
            if self._models[-1].depth == len(self.knowledge_base):
                return self._models[-1].tableau
            current_model = self._models.pop()
            next_tableau = self.knowledge_base[current_model.depth]
            combined_tableau = Tableau.merge(next_tableau, parent=current_model.tableau)
            for new_model_tableau in generate_models(
                combined_tableau, Axioms.get_axioms()
            ):
                new_model = AgentNode(
                    new_model_tableau, current_model.depth + 1, current_model
                )
                self._models.append(new_model)
        return None

    def query(self, query_tableau: Tableau) -> Optional[Tableau]:
        """Query the agent for some information"""
        return self.add_information(query_tableau)


def test_model_generation():
    """Module entry point"""
    alex_pet_fido = Exists(
        lambda e: Predicates.subject(e, Constants.alex)
        & Predicates.action(e, Constants.bite)
        & Predicates.object(e, Constants.fido),
        sort=Sorts.event,
    )
    fido_bit_him = Exists(
        partial_formula=lambda e: Exists(
            partial_formula=lambda o: Predicates.subject(e, Constants.fido)
            & Predicates.action(e, Constants.bite)
            & Predicates.object(e, o),
            sort=Sorts.agent,
        ),
        sort=Sorts.event,
    )
    he_bit_him = Exists(
        partial_formula=lambda e: Exists(
            partial_formula=lambda s: Exists(
                partial_formula=lambda o: Predicates.subject(e, s)
                & Predicates.action(e, Constants.bite)
                & Predicates.object(e, o),
                sort=Sorts.agent,
            ),
            sort=Sorts.agent,
        ),
        sort=Sorts.event,
    )
    formulas = [
        # alex_pet_fido,
        # fido_bit_him,
        he_bit_him,
    ]
    entities = [Constants.alex, Constants.fido, Constants.bite]
    axioms = []
    for model_number, model in enumerate(
        generate_models(Tableau(formulas, entities), axioms)
    ):
        print(f"Model {model_number+1}:")
        for formula in sorted(
            model.branch_literals, key=lambda l: (str(l.args[0]), str(l))
        ):
            print("  ", formula)


def test_agent():
    """Module entry point"""
    agent = Agent()
    alex_pet_fido = Exists(
        lambda e: Predicates.subject(e, Constants.alex)
        & Predicates.action(e, Constants.pet)
        & Predicates.object(e, Constants.fido),
        sort=Sorts.event,
    )
    fido_bit_alex = Exists(
        lambda e: Predicates.subject(e, Constants.fido)
        & Predicates.action(e, Constants.bite)
        & Predicates.object(e, Constants.alex),
        sort=Sorts.event,
    )
    fido_bit_him = Exists(
        partial_formula=lambda e: Exists(
            partial_formula=lambda o: Predicates.subject(e, Constants.fido)
            & Predicates.action(e, Constants.bite)
            & Predicates.object(e, o),
            sort=Sorts.agent,
        ),
        sort=Sorts.event,
    )
    he_bit_him = Exists(
        partial_formula=lambda e: Exists(
            partial_formula=lambda s: Exists(
                partial_formula=lambda o: Predicates.subject(e, s)
                & Predicates.action(e, Constants.bite)
                & Predicates.object(e, o),
                sort=Sorts.agent,
            ),
            sort=Sorts.agent,
        ),
        sort=Sorts.event,
    )

    first_sentence = Tableau(
        [alex_pet_fido], [Constants.alex, Constants.pet, Constants.fido]
    )
    second_sentence = Tableau([fido_bit_alex], [Constants.bite])

    query = Tableau([fido_bit_him])

    story = [
        first_sentence,
        second_sentence,
        # query,
    ]

    for t in story:
        out = agent.add_information(t)
        if not out:
            print("No valid model")
            break
        for formula in sorted(
            out.branch_literals, key=lambda l: (str(l.args[0]), str(l))
        ):
            print(" ", formula)
        print()


if __name__ == "__main__":
    test_agent()
