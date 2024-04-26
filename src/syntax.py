from typing import Callable, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum


__all__ = (
    "Term",
    "Constant",
    "Variable",
    "Formula",
    "Predicate",
    "AppliedPredicate",
    "And",
    "Not",
    "Or",
    "Implies",
    "Agent",
    "Type_",
    "PartialFormula",
    "Quantifier",
    "QuantifiedFormula",
    "FocusQuantifiedFormula",
    "Forall",
    "ForallF",
    "Exists",
    "ExistsF",
    "LogicalConstant",
    "True_",
    "False_",
    "Eq",
    "is_literal",
)


class Term(ABC):
    """Term productions"""

    class Sort(Enum):
        EVENT = 0
        TYPE = 1
        AGENT = 2

    def __init__(self, sort: Sort):
        self.sort = sort

    @abstractmethod
    def _get_str(self) -> str:
        pass

    def __str__(self) -> str:
        return self._get_str()

    def __eq__(self, obj) -> bool:
        if isinstance(obj, type(self)):
            if self.sort == obj.sort:
                return self._get_str() == obj._get_str()
        return False

    def __hash__(self) -> int:
        return hash(self.__dict__.items())


class Constant(Term):
    """Constant production"""

    _id: int = 0

    def __init__(self, sort: Term.Sort, name: Optional[str] = None):
        super().__init__(sort)
        if name is not None:
            assert not name.startswith("_")
        # If no name is provided a new constant is instantiated with an id
        # (for automatic constant generation)
        if name is None:
            self.name = f"_C{Constant._id}"
            Constant._id += 1
        else:
            self.name = name

    def _get_str(self) -> str:
        match self.sort:
            case Term.Sort.EVENT:
                return f"Event({self.name})"
            case Term.Sort.TYPE:
                return f"Type({self.name})"
            case Term.Sort.AGENT:
                return f"Agent({self.name})"
            case _:
                raise ValueError(f"Undefined constant sort: {self.sort}")


class Variable(Term):
    """Constant production"""

    _id: int = 0

    def __init__(self, sort: Term.Sort, name: Optional[str] = None):
        super().__init__(sort)
        if name is not None:
            assert not name.startswith("_")
        self.sort = sort
        # If no name is provided a new variable is instantiated with an id
        # (for automatic constant generation)
        if name is None:
            self.name = f"_V{Variable._id}"
            Variable._id += 1
        else:
            self.name = name

    def _get_str(self) -> str:
        return self.name


class Formula(ABC):
    """Formula productions"""

    def __init__(self):
        self.annotation: Optional[str] = None

    @abstractmethod
    def _get_str(self) -> str:
        pass

    def __str__(self) -> str:
        if self.annotation:
            return self._get_str() + f" | ({self.annotation})"
        return self._get_str()

    def __eq__(self, obj) -> bool:
        if isinstance(obj, type(self)):
            return self._get_str() == obj._get_str()
        return False

    def __hash__(self) -> int:
        return hash(self.__dict__.items())
    
    def __add__(self, other) -> "Or":
        return Or(self, other)
    
    def __mul__(self, other) -> "And":
        return And(self, other)
    
    def __neg__(self) -> "Not":
        return Not(self)


@dataclass
class Predicate:
    name: str
    arity: int

    def __str__(self) -> str:
        return f"{self.name}\{self.arity}"

    def __call__(self, *args: List[Term]) -> "AppliedPredicate":
        return AppliedPredicate(self, args)


@dataclass
class AppliedPredicate(Formula):
    predicate: Predicate
    args: List[Term]

    def __post_init__(self) -> None:
        assert self.predicate.arity == len(self.args)
        super().__init__()

    def _get_str(self) -> str:
        if self.predicate.arity == 0:
            return self.predicate.name
        args_string = ",".join(str(x) for x in self.args)
        return f"{self.predicate.name}({args_string})"

    def __eq__(self, o2: object) -> bool:
        if isinstance(o2, AppliedPredicate):
            if self.predicate == o2.predicate:
                return all(a1 == a2 for a1, a2 in zip(self.args, o2.args))
        return False


class LogicalConstant(AppliedPredicate):
    def __init__(self, name):
        self._predicate = Predicate(name, 0)
        super().__init__(self._predicate, [])


@dataclass
class And(Formula):
    left: Formula
    right: Formula

    def __post_init__(self):
        super().__init__()

    def _get_str(self) -> str:
        return f"({self.left} & {self.right})"

    def __eq__(self, o2: object) -> bool:
        if isinstance(o2, And):
            return self.left == o2.left and self.right == o2.right
        return False


@dataclass
class Not(Formula):
    formula: Formula

    def __post_init__(self):
        super().__init__()

    def _get_str(self) -> str:
        return f"-{self.formula}"

    def __eq__(self, o2: object) -> bool:
        if isinstance(o2, Not):
            return self.formula == o2.formula
        return False


class Or(Not):
    def __init__(self, left: Formula, right: Formula):
        super().__init__(And(Not(left), Not(right)))
        self.left = left
        self.right = right

    def _get_str(self) -> str:
        return f"({self.left} | {self.right})"


class Implies(Or):
    def __init__(self, pre: Formula, post: Formula):
        super().__init__(Not(pre), post)
        self.pre = pre
        self.post = post

    def _get_str(self) -> str:
        return f"({self.pre} -> {self.post})"


True_ = LogicalConstant("T")
False_ = LogicalConstant("F")


@dataclass
class Quantifier:
    name: str

    def __str__(self) -> str:
        return self.name


# PartialFormula = Callable[[Term], Union[Formula, "PartialFormula"]]


class PartialFormula:
    def __init__(
        self, callable: Callable[[Term], Union[Formula, "PartialFormula"]]
    ) -> None:
        self.callable = callable

    def __call__(self, term: Term) -> Union[Formula, "PartialFormula"]:
        out = self.callable(term)
        if isinstance(out, Formula):
            return out
        else:
            return PartialFormula(out)

    def _make_str(self, x: int = 0) -> str:
        v = Variable(f"x{x}")
        out = self(v)
        if isinstance(out, Formula):
            return f"\\{v}.{out}"
        ret_str = f"\\{v}.{out._make_str(x+1)}"
        return ret_str

    def __str__(self) -> str:
        return self._make_str()

    def __eq__(self, o2: object) -> str:
        if isinstance(o2, PartialFormula):
            v = Variable(Term.Sort.EVENT, "temp")
            return self(v) == o2(v)
        return False


@dataclass
class QuantifiedFormula(Formula):
    quantifier: Quantifier
    partial_formula: PartialFormula
    variable: Optional[Variable] = None
    sort: Optional[Term.Sort] = None

    def __post_init__(self):
        assert (self.sort is None) ^ (self.variable is None)
        if self.variable is None:
            self.variable = Variable(self.sort)
        if not isinstance(self.partial_formula, PartialFormula):
            self.partial_formula = PartialFormula(self.partial_formula)
        self.sort = self.variable.sort
        super().__init__()

    @property
    def applied(self) -> Union[Formula, PartialFormula]:
        """Returns the partial formula after applying the quantifying term"""
        return self.partial_formula(self.variable)

    def _get_str(self) -> str:
        return f"{self.quantifier}_{self.variable}.{self.applied}"

    def __eq__(self, o2: object) -> bool:
        if isinstance(o2, QuantifiedFormula):
            if self.quantifier == o2.quantifier and self.sort == o2.sort:
                return self.partial_formula(self.variable) == o2.partial_formula(
                    self.variable
                )
        return False


@dataclass
class FocusQuantifiedFormula(Formula):
    quantifier: Quantifier
    variable: Variable
    sort: Term.Sort
    unfocused_partial: PartialFormula
    focused_partial: PartialFormula

    def __post_init__(self):
        assert (self.variable is None) ^ (self.sort is None)
        if self.variable is None:
            self.variable = Variable(self.sort)
        if not isinstance(self.unfocused_partial, PartialFormula):
            self.unfocused_partial = PartialFormula(self.unfocused_partial)
        if not isinstance(self.focused_partial, PartialFormula):
            self.focused_partial = PartialFormula(self.focused_partial)
        self.sort = self.variable.sort
        super().__init__()

    @property
    def unfocused(self) -> Union[Formula, PartialFormula]:
        return self.unfocused_partial(self.variable)

    @property
    def focused(self) -> Union[Formula, PartialFormula]:
        return self.focused_partial(self.variable)

    def _get_str(self) -> str:
        return f"{self.quantifier}_{self.variable}:{self.unfocused}.{self.focused}"

    def __eq__(self, o2: object) -> bool:
        if isinstance(o2, FocusQuantifiedFormula):
            if self.quantifier == o2.quantifier and self.sort == o2.sort:
                return self.focused_partial(self.variable) == o2.focused_partial(
                    self.variable
                ) and self.unfocused_partial(self.variable) == o2.focused_partial(
                    self.variable
                )
        return False


class Forall(QuantifiedFormula):
    _quantifier = Quantifier("A")

    def __init__(
        self,
        partial_formula: PartialFormula,
        sort: Optional[Term.Sort] = None,
        variable: Optional[Variable] = None,
    ) -> None:
        super().__init__(self._quantifier, partial_formula, variable, sort)


class Exists(Not):
    def __init__(
        self,
        partial_formula: PartialFormula,
        sort: Optional[Term.Sort] = None,
        variable: Optional[Variable] = None,
    ):
        f = Forall(lambda x: Not(partial_formula(x)), sort, variable)
        self.variable = f.variable
        self.sort = f.sort
        self.partial_formula = partial_formula
        super().__init__(f)

    def _get_str(self) -> str:
        return f"E_{self.formula.variable}.{self.formula.applied.formula}"


class ForallF(FocusQuantifiedFormula):
    _quantifier = Quantifier("A")

    def __init__(
        self,
        unfocsed_partial_formula: PartialFormula,
        focused_partial_formula,
        sort: Optional[Term.Sort] = None,
        variable: Optional[Variable] = None,
    ) -> None:
        super().__init__(
            self._quantifier,
            variable,
            sort,
            unfocsed_partial_formula,
            focused_partial_formula,
        )


class ExistsF(Not):
    def __init__(
        self,
        unfocused_partial: PartialFormula,
        focused_partial: PartialFormula,
        sort: Optional[Term.Sort],
        variable: Optional[Variable] = None,
    ):
        f = ForallF(
            unfocused_partial, lambda x: Not(focused_partial(x)), sort, variable
        )
        self.unfocused_partial = f.unfocused
        self.focused_partial = focused_partial
        self.variable = f.variable
        self.sort = f.sort
        super().__init__(f)

    def _get_str(self) -> str:
        return f"E_{self.formula.variable}:{self.formula.unfocused}.{self.formula.focused.formula}"


def is_literal(f: Formula) -> bool:
    """Return true if f is an atom or the negation of one .i.e. a literal"""
    return isinstance(f, AppliedPredicate) or (
        isinstance(f, Not) and isinstance(f.formula, AppliedPredicate)
    )


class Agent(AppliedPredicate):
    _agent: Predicate = Predicate("ag", 2)

    def __init__(self, event: Term, agent: Term):
        super().__init__(Agent._agent, [event, agent])

    @property
    def event(self) -> Term:
        return self.args[0]

    @property
    def agent(self) -> Term:
        return self.args[1]


class Eq(AppliedPredicate):
    _eq: Predicate = Predicate("eq", 2)

    def __init__(self, left: Term, right: Term):
        super().__init__(Eq._eq, [left, right])
        self.left = left
        self.right = right

    def _get_str(self) -> str:
        return f"{self.left} = {self.right}"


class Type_(AppliedPredicate):
    _type_: Predicate = Predicate("ty", 2)

    def __init__(self, event: Term, type_: Term):
        super().__init__(Type_._type_, [event, type_])


if __name__ == "__main__":
    p = Predicate("p", 4)
    f = Forall(
        lambda x: Forall(
            lambda y: Implies(
                Eq(x, y), Or(Eq(y, x), Exists(lambda z: p(x, y, z, x), Term.Sort.EVENT))
            ),
            Term.Sort.EVENT,
        ),
        Term.Sort.EVENT,
    )
    print(f)
