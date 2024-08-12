from dataclasses import dataclass
import functools
import operator
from typing import Callable, Iterable
from src.logic.base.calculus import generate_models
from src.logic.base.syntax import Constant, Exists, Predicate, Sort
from src.logic.base.tableau import Tableau


class Sorts:
    """Domain sorts"""
    agent = Sort("agent")
    action = Sort("action")
    event = Sort("event")


class Predicates:
    """Domain relations/concepts"""
    subject = Predicate("subject",2, (Sorts.event, Sorts.agent))
    action = Predicate("action", 2, sorts=(Sorts.event, Sorts.action))
    object = Predicate("object",2, (Sorts.event, Sorts.agent))


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

    def __init__(self, str_repr: str, get_readings: Callable[[], Iterable[Tableau]]) -> None:
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
                        lambda e: Predicates.subject(e, c_subject) & Predicates.action(e, c_action),
                        sort=Sorts.event
                    )
                ],
                [c_subject, c_action],
            ),
        )
    return Sentence(str_repr, get_readings)


def main():
    """Module entry point"""
    alex_pet_fido = Exists(
        lambda e: Predicates.subject(e, Constants.alex) & Predicates.action(e, Constants.bite) & Predicates.object(e, Constants.fido),
        sort=Sorts.event
    )
    fido_bit_him = Exists(partial_formula=lambda e:
        Exists(partial_formula=lambda o:
            Predicates.subject(e, Constants.fido) & Predicates.action(e, Constants.bite) & Predicates.object(e, o),
        sort=Sorts.agent),
    sort=Sorts.event)
    he_bit_him = Exists(partial_formula=lambda e:
        Exists(partial_formula=lambda s:
            Exists(partial_formula=lambda o:
                Predicates.subject(e, s) & Predicates.action(e, Constants.bite) & Predicates.object(e, o),
            sort=Sorts.agent),
        sort=Sorts.agent),
    sort=Sorts.event)
    formulas = [
        # alex_pet_fido,
        # fido_bit_him,
        he_bit_him,
        ]
    entities = [Constants.alex, Constants.fido, Constants.bite]
    axioms = []
    for model in generate_models(Tableau(formulas, entities), axioms):
        print("Model:")
        for formula in model.branch_literals:
            print("  > ", formula)


if __name__ == "__main__":
    main()
