from .syntax import *
from typing import Iterable, Optional
from dataclasses import dataclass, field


__all__ = ("Tableau",)


@dataclass
class Tableau:
    formulas: Iterable[Formula]
    entities: Iterable[Term] = field(default_factory=list)
    parent: Optional["Tableau"] = None
    closing: bool = False

    @property
    def branch_formulas(self) -> Iterable[Formula]:
        if self.parent is None:
            return self.formulas
        return *self.formulas, *self.parent.branch_formulas

    @property
    def branch_model(self) -> Iterable[Formula]:
        return (f for f in self.branch_formulas if type(f) in (AppliedPredicate,))

    @property
    def branch_entities(self) -> Iterable[Term]:
        if self.parent is None:
            return self.entities
        return *self.entities, *self.parent.branch_entities

    def get_model(self) -> Iterable[Formula]:
        return filter(is_literal, self.branch_formulas)

    @staticmethod
    def merge(*tableaus: Iterable["Tableau"], parent: "Tableau" = None) -> "Tableau":
        formulas = []
        entities = []
        closing = False
        for t in tableaus:
            [formulas.append(f) for f in t.formulas if f not in formulas]
            [entities.append(c) for c in t.entities if c not in entities]
            if t.closing:
                closing = True
        return Tableau(formulas, entities, closing=closing, parent=parent)
    
    def copy(self) -> "Tableau":
        copy = Tableau.merge(self)
        copy.parent = self.parent
        return copy

    def __str__(self) -> str:
        return str(
            {
                "formulas": [str(x) for x in self.formulas],
                "entities": [str(x) for x in self.entities],
            }
        )
