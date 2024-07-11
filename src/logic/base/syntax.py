"""This module implements a syntax base for sorted logic fragments"""

import itertools
from typing import Callable, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


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
    "PartialFormula",
    "Quantifier",
    "QuantifiedFormula",
    "FocusQuantifiedFormula",
    "Forall",
    "ForallF",
    "Exists",
    "ExistsF",
    "Literal",
    "LogicalConstant",
    "True_",
    "False_",
    "Eq",
    "is_literal",
)


@dataclass(frozen=True, eq=True)
class Sort:
    """Sort of a term, since the logic is sorted"""

    name: str

    def make_constant(self, name: Optional[str] = None) -> "Constant":
        """Create a constant of this sort"""
        return Constant(self, name)


class Term(ABC):
    """Term productions"""

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
        return hash(str(self))


class Constant(Term):
    """Constant production"""

    _id: int = 0

    def __init__(self, sort: Sort, name: Optional[str] = None):
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
        return f"{self.sort.name}({self.name})"


class Variable(Term):
    """Constant production"""

    _id: int = 0

    def __init__(self, sort: Sort, name: Optional[str] = None):
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
        return hash(str(self))

    def __add__(self, other) -> "Or":
        return Or(self, other)

    def __mul__(self, other) -> "And":
        return And(self, other)

    def __or__(self, other) -> "Or":
        return self + other

    def __and__(self, other) -> "And":
        return self * other

    def __rshift__(self, other) -> "Implies":
        return Implies(self, other)

    def __invert__(self) -> "Not":
        return -self

    def __neg__(self) -> "Not":
        return Not(self)


@dataclass
class Predicate:
    """A predicate / n-ary relation"""

    name: str
    arity: int
    sorts: Optional[Tuple[Sort]] = None

    def __post_init__(self) -> None:
        if self.sorts is not None:
            assert len(self.sorts) == self.arity
    
    @property
    def is_typed(self) -> bool:
        """return true if the predicate has typed arguments"""
        return self.sorts is not None

    def __str__(self) -> str:
        return f"{self.name}\\{self.arity}"

    def __call__(self, *args: List[Term]) -> "AppliedPredicate":
        if self.sorts is not None:
            # Make sure arguments are properly sorted
            assert all(itertools.starmap(lambda term, sort: term.sort is sort, zip(args, self.sorts)))
        return AppliedPredicate(self, args)


@dataclass
class AppliedPredicate(Formula):
    """Predicate applied to terms"""

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

    def __hash__(self) -> int:
        return hash(str(self))


class LogicalConstant(AppliedPredicate):
    """Local constants"""

    def __init__(self, name):
        self._predicate = Predicate(name, 0)
        super().__init__(self._predicate, [])


@dataclass
class And(Formula):
    """Conjunction Formula"""

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

    def __hash__(self) -> int:
        return hash(str(self))


@dataclass
class Not(Formula):
    """Negation Formula"""

    formula: Formula

    def __post_init__(self):
        super().__init__()

    def _get_str(self) -> str:
        return f"-{self.formula}"

    def __eq__(self, o2: object) -> bool:
        if isinstance(o2, Not):
            return self.formula == o2.formula
        return False

    def __hash__(self) -> int:
        return hash(str(self))


class Or(Not):
    """Disjunction formulas"""

    def __init__(self, left: Formula, right: Formula):
        super().__init__(And(Not(left), Not(right)))
        self.left = left
        self.right = right

    def _get_str(self) -> str:
        return f"({self.left} | {self.right})"


class Implies(Or):
    """Implication formulas"""

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
    """A quantifier object"""

    name: str

    def __str__(self) -> str:
        return self.name


# PartialFormula = Callable[[Term], Union[Formula, "PartialFormula"]]


class PartialFormula:
    """Lambda handling object"""

    def __init__(
        self,
        callable_: Callable[[Term], Union[Formula, "PartialFormula"]],
        sort: Sort,
    ) -> None:
        self.callable = callable_
        self._sort = sort

    def __call__(self, term: Term) -> Union[Formula, "PartialFormula"]:
        return self.callable(term)

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
            v = Variable(self._sort, "temp")
            return self(v) == o2(v)
        return False

    def __hash__(self) -> int:
        return hash(str(self))


@dataclass
class QuantifiedFormula(Formula):
    """Implementation of a quantified formula"""

    quantifier: Quantifier
    partial_formula: PartialFormula
    variable: Optional[Variable] = None
    sort: Optional[Sort] = None

    def __post_init__(self):
        assert (self.sort is None) ^ (self.variable is None)
        if self.variable is None:
            self.variable = Variable(self.sort)
        if not isinstance(self.partial_formula, PartialFormula):
            self.partial_formula = PartialFormula(self.partial_formula, self.sort)
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

    def __hash__(self) -> int:
        return hash(str(self))


@dataclass
class FocusQuantifiedFormula(Formula):
    """Focused quantified formula wrapper"""

    quantifier: Quantifier
    variable: Variable
    sort: Sort
    unfocused_partial: PartialFormula
    focused_partial: PartialFormula

    def __post_init__(self):
        assert (self.variable is None) ^ (self.sort is None)
        if self.variable is None:
            self.variable = Variable(self.sort)
        if not isinstance(self.unfocused_partial, PartialFormula):
            self.unfocused_partial = PartialFormula(self.unfocused_partial, self.sort)
        if not isinstance(self.focused_partial, PartialFormula):
            self.focused_partial = PartialFormula(self.focused_partial, self.sort)
        self.sort = self.variable.sort
        super().__init__()

    @property
    def unfocused(self) -> Union[Formula, PartialFormula]:
        """Unfocused part of the formula with the variable applied"""
        return self.unfocused_partial(self.variable)

    @property
    def focused(self) -> Union[Formula, PartialFormula]:
        """Focused part of the formula with the variable applied"""
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

    def __hash__(self) -> int:
        return hash(str(self))


class Forall(QuantifiedFormula):
    """Forall formulas"""

    _quantifier = Quantifier("A")

    def __init__(
        self,
        partial_formula: PartialFormula,
        sort: Optional[Sort] = None,
        variable: Optional[Variable] = None,
    ) -> None:
        super().__init__(self._quantifier, partial_formula, variable, sort)


class Exists(Not):
    """Existential formulas"""

    def __init__(
        self,
        partial_formula: PartialFormula,
        sort: Optional[Sort] = None,
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
    """Focused forall formulas"""

    _quantifier = Quantifier("A")

    def __init__(
        self,
        unfocsed_partial_formula: PartialFormula,
        focused_partial_formula,
        sort: Optional[Sort] = None,
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
    """Focused existential formulas"""

    def __init__(
        self,
        unfocused_partial: PartialFormula,
        focused_partial: PartialFormula,
        sort: Optional[Sort],
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


class Eq(AppliedPredicate):
    """Identity relation"""

    _eq: Predicate = Predicate("eq", 2)

    def __init__(self, left: Term, right: Term):
        super().__init__(Eq._eq, [left, right])
        self.left = left
        self.right = right

    def _get_str(self) -> str:
        return f"{self.left} = {self.right}"


Literal = AppliedPredicate | Not


if __name__ == "__main__":
    p = Predicate("p", 4)
    event_sort = Sort("Event")
    f = Forall(
        lambda x: Forall(
            lambda y: Eq(x, y)
            >> (Eq(y, x) | Exists(lambda z: p(x, y, z, x), event_sort)),
            event_sort,
        ),
        event_sort,
    )
    print(f)
