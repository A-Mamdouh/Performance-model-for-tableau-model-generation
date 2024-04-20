from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Sequence, Iterable
from enum import Enum

from .syntax import *


__all__ = (
    "Sentence",
    "Narrator",
    "Verb",
    "NounVerbSentence",
    "NounNotVerbSentence",
    "NounAlwaysVerbSentence",
    "Story",
    "Narrator",
)


@dataclass
class Verb:
    inf: str
    past: str


@dataclass
class Sentence(ABC):
    noun: str
    verb: Verb

    class Focus(Enum):
        FULL = 0
        NOUN = 1
        VERB = 2

    @abstractmethod
    def get_str(self, focus: Focus) -> Iterable[str]:
        pass

    def __str__(self) -> str:
        self.get_str(Sentence.Focus.ALL)

    @abstractmethod
    def get_formulas(self, focus: Focus = Focus.FULL) -> Iterable[Formula]:
        pass


class NounVerbSentence(Sentence):
    def get_str(self, focus: Sentence.Focus) -> Iterable[str]:
        match focus:
            case Sentence.Focus.FULL:
                return [f"{self.noun} {self.verb.past}"]
            case Sentence.Focus.NOUN:
                return [f"The one who {self.verb.past} is {self.noun}"]
            case Sentence.Focus.VERB:
                return [f"{self.noun} did something; {self.noun} {self.verb.past}"]
            case _:
                raise NotImplementedError()

    def get_formulas(self, focus: Sentence.Focus) -> Iterable[Formula]:
        n = Constant(Term.Sort.AGENT, self.noun)
        v = Constant(Term.Sort.TYPE, self.verb.inf)
        formulas: Formula = None
        match focus:
            case Sentence.Focus.FULL:
                formulas = [
                    # E_e.ag(e, n) /\ ty(e, v)
                    Exists( lambda e: And(Agent(e, n), Type_(e, v)), Term.Sort.EVENT )
                ]
            case Sentence.Focus.NOUN:
                formulas = [
                    # E_e.ty(e, v):ag(e, n)
                    ExistsF( lambda e: Type_(e, v), lambda e: Agent(e, n), Term.Sort.EVENT )
                ]
            case Sentence.Focus.VERB:
                
                formulas = [
                    # E_e.ag(e, n):ty(e, v)
                    ExistsF( lambda e: Agent(e, n), lambda e: Type_(e, v), Term.Sort.EVENT )
                ]
            case _:
                raise NotImplementedError()
        # Add annotation
        for formula, annotation in zip(formulas, self.get_str(focus)):
            formula.annotation = annotation
        return formulas


class NounAlwaysVerbSentence(Sentence):
    def get_str(self, focus: Sentence.Focus) -> Iterable[str]:
        match focus:
            case Sentence.Focus.FULL:
                return [f"{self.noun} always {self.verb.past}"] # TODO: This focus does not make sense. Since it will pick one of the others internally.
            case Sentence.Focus.NOUN:
                return [f"if someone {self.verb.past}, it was {self.noun}"]
            case Sentence.Focus.VERB:
                return [f"all {self.noun} does is {self.verb.inf}"]
            case _:
                raise NotImplementedError()

    def get_formulas(self, focus: Sentence.Focus) -> Iterable[Formula]:
        n = Constant(Term.Sort.AGENT, self.noun)
        v = Constant(Term.Sort.TYPE, self.verb.inf)
        formulas: Formula = None
        match focus:
            case Sentence.Focus.FULL:
                formulas = [
                    # A_e . ag(e, n) -> ty(e, v)
                    Forall( lambda e: Implies(Agent(e, n), Type_(e, v)), Term.Sort.EVENT )
                ]
            case Sentence.Focus.NOUN:
                formulas = [
                    # A_e.ty(e,v):ag(e,n)
                    ForallF( lambda e: Type_(e, v), lambda e: Agent(e, n), Term.Sort.EVENT )
                ]
            case Sentence.Focus.VERB:
                formulas = [
                    # A_e.ag(e,v):ty(e,n)
                    ForallF( lambda e: Agent(e, n), lambda e: Type_(e, v), Term.Sort.EVENT )
                ]
            case _:
                raise NotImplementedError()
        for formula, annotation in zip(formulas, self.get_str(focus)):
            formula.annotation = annotation
        return formulas


class NounNotVerbSentence(Sentence):
    def get_str(self, focus: Sentence.Focus) -> Iterable[str]:
        match focus:
            case Sentence.Focus.FULL:
                return [
                    # Event is not negated
                    f"Either {self.noun} did something that is not {self.verb.inf}, or someone other than {self.noun} {self.verb.past}",
                    # Event is negated
                    f"Every event is either not {self.noun}'s or not a {self.verb.inf} event"
                ]
            case Sentence.Focus.NOUN:
                return [
                        # Event is not engated
                        f"Someone {self.verb.past}, but it was not {self.noun}",
                        # Event is negated
                        f"{self.noun} never {self.verb.past}"
                    ]
            case Sentence.Focus.VERB:
                return [
                        # Event is not negated
                        f"{self.noun} did something, but it was not {self.verb.inf}",
                        # Event is negated
                        f"if someone {self.verb.past}, it's not {self.noun}"
                    ]
            case _:
                raise NotImplementedError()

    def get_formulas(self, focus: Sentence.Focus) -> Formula:
        n = Constant(Term.Sort.AGENT, self.noun)
        v = Constant(Term.Sort.TYPE, self.verb.inf)
        formulas: Iterable[Formula] = None
        match focus:
            case    Sentence.Focus.FULL:
                formulas = [
                    # E_e.(-ag(e, n) | -ty(e, v))
                    Exists(lambda e: -Agent(e, n) + -Type_(e, v), Term.Sort.EVENT),
                    # A_e.(-ag(e, n) | -ty(e, v))
                    Forall(lambda e: -Agent(e, n) + -Type_(e, v), Term.Sort.EVENT),
                ]
            case Sentence.Focus.NOUN:
                formulas = [
                    # E_e.Ty(e,v):-Ag(e,n)
                    ExistsF(lambda e: Type_(e, v), lambda e: -Agent(e, n), Term.Sort.EVENT),
                    # A_e.Ty(e,v):-Ag(e,n)
                    ForallF(lambda e: Type_(e, v), lambda e: -Agent(e, n), Term.Sort.EVENT),
                ]
            case Sentence.Focus.VERB:
                formulas = [
                    # E_e.Ag(e,n):-Ty(e,v)
                    ExistsF(lambda e: Agent(e, n), lambda e: -Type_(e, v), Term.Sort.EVENT),
                    # A_e.Ag(e,n):-Ty(e,v)
                    ForallF(lambda e: Agent(e, n), lambda e: -Type_(e, v), Term.Sort.EVENT),
                ]
            case _:
                raise NotImplementedError()
        for formula, annotation in zip(formulas, self.get_str(focus)):
            formula.annotation = annotation
        return formulas


Story = Sequence[Sentence]


class Narrator:
    def __init__(self, story: Story, position: int = 0) -> None:
        self.story = story
        self._position = position

    def copy(self) -> "Narrator":
        return Narrator(self.story, self._position)

    def __iter__(self) -> Iterable[Sentence]:
        return self

    def __next__(self) -> Sentence:
        if self._position >= len(self.story):
            raise StopIteration
        self._position += 1
        return self.story[self._position - 1]

    @property
    def story_so_far(self) -> str:
        return (
            ". ".join(
                x.get_str(Sentence.Focus.FULL) for x in self.story[: self._position]
            )
            + "."
        )
