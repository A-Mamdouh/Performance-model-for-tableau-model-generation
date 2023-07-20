from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Sequence, ClassVar, Iterable
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
    def get_str(self, focus: Focus) -> str:
        pass

    def __str__(self) -> str:
        self.get_str(Sentence.Focus.ALL)

    @abstractmethod
    def get_formula(self, focus: Focus = Focus.FULL) -> Formula:
        pass


class NounVerbSentence(Sentence):
    def get_str(self, focus: Sentence.Focus) -> str:
        if focus == Sentence.Focus.FULL:
            return f"{self.noun} {self.verb.past}"
        elif focus == Sentence.Focus.NOUN:
            return f"the one who {self.verb.past} is {self.noun}"
        elif focus == Sentence.Focus.VERB:
            return f"{self.noun} did something: {self.noun} {self.verb.past}"
        else:
            raise NotImplementedError()

    def get_formula(self, focus: Sentence.Focus) -> Formula:
        n = Constant(Term.Sort.AGENT, self.noun)
        v = Constant(Term.Sort.TYPE, self.verb.inf)
        if focus == Sentence.Focus.FULL:
            # E_e.ag(e, n) /\ ty(e, v)
            return Exists(lambda e: And(Agent(e, n), Type_(e, v)), Term.Sort.EVENT)
        elif focus == Sentence.Focus.NOUN:
            # E_e.ty(e, v):ag(e, n)
            return ExistsF(
                lambda e: Type_(e, v), lambda e: Agent(e, n), Term.Sort.EVENT
            )
        elif focus == Sentence.Focus.VERB:
            # E_e.ag(e, n):ty(e, v)
            return ExistsF(
                lambda e: Agent(e, n), lambda e: Type_(e, v), Term.Sort.EVENT
            )
        else:
            raise NotImplementedError()


class NounNotVerbSentence(Sentence):
    def get_str(self, focus: Sentence.Focus) -> str:
        if focus == Sentence.Focus.FULL:
            return f"{self.noun} did not {self.verb.inf}"
        elif focus == Sentence.Focus.NOUN:
            return f"the one who {self.verb.past} is not {self.noun}"
        elif focus == Sentence.Focus.VERB:
            return (
                f"{self.noun} did not {self.verb.inf} ({self.noun} did something else)"
            )
        else:
            raise NotImplementedError()

    def get_formula(self, focus: Sentence.Focus) -> Formula:
        n = Constant(Term.Sort.AGENT, self.noun)
        v = Constant(Term.Sort.TYPE, self.verb.inf)
        if focus == Sentence.Focus.FULL:
            # A_e.-(ag(e, n) & ty(e, v))
            return Forall(lambda e: Not(And(Agent(e, n), Type_(e, v))), Term.Sort.EVENT)
        elif focus == Sentence.Focus.NOUN:
            # A_e.ty(e,v):-ag(e,n)
            return ForallF(
                lambda e: Type_(e, v), lambda e: Not(Agent(e, n)), Term.Sort.EVENT
            )
        elif focus == Sentence.Focus.VERB:
            # A_e.ag(e,v):-ty(e,n)
            return ForallF(
                lambda e: Agent(e, n), lambda e: Not(Type_(e, v)), Term.Sort.EVENT
            )
        else:
            raise NotImplementedError()


class NounAlwaysVerbSentence(Sentence):
    def get_str(self, focus: Sentence.Focus) -> str:
        if focus == Sentence.Focus.FULL:
            return f"{self.noun} always {self.verb.past}"
        elif focus == Sentence.Focus.NOUN:
            return f"if someone {self.verb.past}, it was {self.noun}"
        elif focus == Sentence.Focus.VERB:
            return f"all {self.noun} does is {self.verb.inf}"
        else:
            raise NotImplementedError()

    def get_formula(self, focus: Sentence.Focus) -> Formula:
        n = Constant(Term.Sort.AGENT, self.noun)
        v = Constant(Term.Sort.TYPE, self.verb.inf)
        if focus == Sentence.Focus.FULL:
            # A_e . ag(e, n) -> ty(e, v)
            return Forall(
                (lambda e: Implies(Agent(e, n), Type_(e, v))), Term.Sort.EVENT
            )
        elif focus == Sentence.Focus.NOUN:
            # A_e.ty(e,v):ag(e,n)
            return ForallF(
                lambda e: Type_(e, v), lambda e: Agent(e, n), Term.Sort.EVENT
            )
        elif focus == Sentence.Focus.VERB:
            # A_e.ag(e,v):ty(e,n)
            return ForallF(
                lambda e: Agent(e, n), lambda e: Type_(e, v), Term.Sort.EVENT
            )
        else:
            raise NotImplementedError()


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
