"""Implmentation of the narrator of this fragment"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, ClassVar, List

from src.logic.base.syntax import Constant, Exists
from src.logic.base.tableau import Tableau
from . import environment as E

Noun = str
Location = str


@dataclass
class Verb:
    """Verb wrapper class"""

    inf: str
    past: str


@dataclass
class SimpleEventSentence(ABC):
    """Simple sentence productions"""

    subject: Noun | None = None
    verb: Verb | None = None
    object: Noun | None = None
    location: Location | None = None

    def __post_init__(self) -> None:
        """Make sure one of the details is given"""
        if not any([self.subject, self.verb, self.object, self.location]):
            raise ValueError("Cannot create an empty sentence with no information")

    @abstractmethod
    def get_all_readings(self) -> List[Tableau]:
        """Get all possible readings of this sentence"""


class AccusativeSentence(ABC):
    """Productions of x witnessed \phi. These sentences include 2 events at once"""

    @abstractmethod
    def get_all_readings(self) -> List[List[SimpleEventSentence]]:
        """Return all possible readings of the witnessed event"""


class NounVerbSentence(SimpleEventSentence):
    """Sentences of the form noun did verb"""

    # pylint: disable=C3001:unnecessary-lambda-assignment
    def get_all_readings(self) -> List[Tableau]:
        entities: Constant = []
        # Set subject lambda
        subject = lambda e: Exists(  # noqa: E731
            lambda a: E.Predicates.subject(e, a), sort=E.Sorts.agent
        )  # noqa: E731
        if self.subject:
            s_constant = E.Constants.get(self.subject, E.Sorts.agent)
            subject = lambda e: E.Predicates.subject(e, s_constant)  # noqa: E731
            entities.append(s_constant)
        # Set action lambda
        action = lambda e: Exists(  # noqa: E731
            lambda a: E.Predicates.action(e, a), sort=E.Sorts.action
        )  # noqa: E731
        if self.verb:
            v_constant = E.Constants.get(self.verb.inf, E.Sorts.action)
            entities.append(v_constant)
            action = lambda e: E.Predicates.action(e, v_constant)  # noqa: E731
        # Set object lambda
        object_ = lambda e: Exists(  # noqa: E731
            lambda a: E.Predicates.object(e, a), sort=E.Sorts.agent
        )  # noqa: E731
        if self.object:
            o_constant = E.Constants.get(self.object, E.Sorts.agent)
            entities.append(o_constant)
            object_ = lambda e: E.Predicates.object(e, o_constant)  # noqa: E731
        # Set location lambda
        location = lambda e: Exists(  # noqa: E731
            lambda a: E.Predicates.location(e, a), sort=E.Sorts.location
        )  # noqa: E731
        if self.location:
            l_constant = E.Constants.get(self.location, E.Sorts.location)
            entities.append(l_constant)
            location = lambda e: E.Predicates.location(e, l_constant)  # noqa: E731
        # Put together the partial lambda for the final quantified formula
        partial_formula = lambda e: subject(e) & action(e) & object_(e) & location(e)  # noqa: E731
        formula = Exists(partial_formula=partial_formula, sort=E.Sorts.event)
        return [Tableau([formula], entities=entities)]


@dataclass
class SomeoneHeardAnEvent(AccusativeSentence):
    """Someone heard someone else at a location"""

    accuser: Noun | None = None
    accused: Noun | None = None
    location: Location | None = None
    sentence_callable: Callable[[Location | None], SimpleEventSentence]

    _heard: ClassVar[Verb] = Verb("hear", "heard")

    def get_all_readings(self) -> List[List[SimpleEventSentence]]:
        return [
            # accuser is at location
            [
                NounVerbSentence(
                    subject=self.accuser,
                    verb=SomeoneHeardAnEvent._heard,
                    object=self.accused,
                    location=self.location,
                ),
                self.sentence_callable(None),
            ],
            # accused is at location
            [
                NounVerbSentence(
                    subject=self.accuser,
                    verb=SomeoneHeardAnEvent._heard,
                    object=self.accused,
                    location=None,
                ),
                self.sentence_callable(self.location),
            ],
        ]


if __name__ == "__main__":
    sentence = NounVerbSentence(
        subject="alex",
        verb=Verb("see", "saw"),
        object="alex",
        location="library",
    )
    f = sentence.get_all_readings()[0].formulas[0]
    print(f)
