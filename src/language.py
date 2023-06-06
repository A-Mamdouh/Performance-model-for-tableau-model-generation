from abc import ABC, abstractmethod
from typing import ClassVar, Optional
import dataclasses
from pydantic.dataclasses import dataclass as pydantic_dataclass


__VALIDATE__ = True


def dataclass(Cls=None, **kwargs):
    def _decorator(Cls):
        Dataclass = dataclasses.dataclass(Cls, **kwargs)
        if __VALIDATE__:
            pydantic_dataclass(Dataclass, config={"arbitrary_types_allowed": True})
        return Dataclass

    if Cls is None:
        return _decorator
    return _decorator(Cls)


### Grammar class declarations
## Term
class Term(ABC):
    pass


class Constant(Term):
    pass


## Formula
class Formula(ABC):
    pass


class Atom(Formula, ABC):
    pass


class Not(Formula):
    pass


class And(Formula):
    pass


class Or(Formula):
    pass


class Exists(Formula):
    pass


class Forall(Formula):
    pass


### Grammar class definitions
## Term
class Term(ABC):
    """Abstract class for the term nonterminal of the grammar"""

    @abstractmethod
    def __init__(self) -> None:
        pass


@dataclass()
class Constant(Term):
    """Constant production class"""

    # TODO: Handle equality with name ambiguities / contradictions
    # Name of the constant
    name: Optional[str] = None
    # Static variables for automatic constant creation
    _count: ClassVar[int] = 0

    def __post_init__(self) -> None:
        # Handle automatic constant generation (useful for quantifier rules)
        if self.name is None:
            self.name = f"c{self._count}"
            self._count += 1


## Formula
class Formula(ABC):
    """Abstract class for the formula nonterminal of the grammar"""

    @abstractmethod
    def __init__(self) -> None:
        pass


class Atom(Formula, ABC):
    """Abstract class for atomic formulas"""

    @abstractmethod
    def __init__(self) -> None:
        pass


@dataclass(frozen=True)
class Proposition(Atom):
    """Proposition production class"""

    name: str


# true and false are treated as atomic propositions
Top = Proposition("T")
Bot = Proposition("F")


@dataclass(frozen=True)
class Not(Formula):
    """Negation production class"""

    f: Formula


@dataclass(frozen=True)
class And(Formula):
    """Conjunction production class"""

    left: Formula
    right: Formula


@dataclass(frozen=True)
class Or(Formula):
    """Disjunction production class"""

    left: Formula
    right: Formula


@dataclass(frozen=True)
class Exists(Formula):
    """Existentially quantified formulas production class"""

    varName: str
    f: Formula


@dataclass(frozen=True)
class Forall(Formula):
    """Universally quantified formulas production class"""

    varName: str
    f: Formula


"""
TODO:
    - Add predicates
    - Add event-based semantics fragment of restricted quantified formulas
"""
