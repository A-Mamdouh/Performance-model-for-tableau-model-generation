"""Implementation of the Tableau node data structure and helping classes"""

# pylint: disable=invalid-name
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Tuple
import itertools

import src.logic.base.syntax as S


@dataclass
class Tableau:
    """This class represents a tableau node.
    It contains formulas, entities and a reference to the node's parent
    """

    #: list of formulas that exist in this node
    formulas: Iterable[S.Formula] = field(default_factory=list)
    #: List of entities that exist in this node
    entities: Iterable[S.Term] = field(default_factory=list)
    #: Parent node. None if this is the root node
    parent: Optional["Tableau"] = None
    #: Current node substitution. This is important with non unique names
    substitution: Optional[S.Substitution] = None
    #: Set of branch formulas. Contains formulas that were dispatched by calculus.
    # Is only used by calculus
    dispatched_formulas: Iterable[S.Formula] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.substitution is None:
            if self.parent is not None:
                # Copy substitution from parent
                self.substitution = self.parent.substitution.copy()
            else:
                # Create a new empty substitution
                self.substitution = S.Substitution()
        # Make sure only unique dispatched formulas are included
        self.dispatched_formulas = list(self.dispatched_formulas)
        unique_dispatched_formulas = []
        for i, x in enumerate(self.dispatched_formulas):
            if x in self.dispatched_formulas[i + 1 :]:
                continue
            if self.parent and x in self.parent.branch_dispatched_formulas:
                continue
            unique_dispatched_formulas.append(x)
        self.dispatched_formulas = unique_dispatched_formulas

    @property
    def branch_dispatched_formulas(self) -> Iterable[S.Formula]:
        """Return the set of dispatched formulas of the entire branch"""
        if self.parent is None:
            return self.dispatched_formulas
        return itertools.chain(
            self.dispatched_formulas, self.parent.branch_dispatched_formulas
        )

    @property
    def undispatched_formulas(self) -> Iterable[S.Formula]:
        """return a set of node formulas that are not dispatched from this branch"""
        return [f for f in self.formulas if f not in self.branch_dispatched_formulas]

    @property
    def branch_undispatched_formulas(self) -> Iterable[S.Formula]:
        """return a set of branch formulas that are not dispatched from this branch"""
        return [
            f for f in self.branch_formulas if f not in self.branch_dispatched_formulas
        ]

    @property
    def branch_formulas(self) -> Iterable[S.Formula]:
        """All formulas from this node up to the root. Starting with this node"""
        if self.parent is None:
            return self.formulas
        return itertools.chain(self.formulas, self.parent.branch_formulas)

    @property
    def branch_entities(self) -> Iterable[S.Term]:
        """All entities (terms) from this node up to the root. Starting with this node"""
        if self.parent is None:
            return self.entities
        return itertools.chain(self.entities, self.parent.branch_entities)

    @property
    def literals(self) -> Iterable[S.Literal]:
        """All literals in the current node"""
        return list(filter(S.is_literal, self.formulas))

    @property
    def branch_literals(self) -> Iterable[S.Literal]:
        """All literals from this node up to the root. Starting with this node"""
        if self.parent is None:
            return self.literals
        return itertools.chain(self.literals, self.parent.branch_literals)

    @classmethod
    def merge(
        cls, *tableaus: "Tableau", parent: Optional["Tableau"] = None
    ) -> "Tableau":
        """Merge given tableaus into one tableau containing all unique formulas and entities"""
        if len(tableaus) == 0:
            return Tableau(parent=parent)
        # Collect all unique formulas
        formulas = set(
            itertools.chain(*map(lambda tableau: tableau.formulas, tableaus))
        )
        # Collect all entities
        entities = set(
            itertools.chain(*map(lambda tableau: tableau.entities, tableaus))
        )
        # Merge substitutions
        merged_substitution = S.Substitution.merge(
            *map(lambda tableau: tableau.substitution, tableaus)
        )
        if merged_substitution is None:
            formulas.add(S.False_)
            merged_substitution = S.Substitution()
        # Create a union of all dispatched formulas
        dispatched_formulas = [f for t in tableaus for f in t.dispatched_formulas]
        # Put together everything into new tableau
        merged_tableau = Tableau(
            formulas=formulas,
            entities=entities,
            parent=None,
            substitution=merged_substitution,
            dispatched_formulas=dispatched_formulas,
        )
        # Set parent if one exists
        if parent:
            merged_tableau = parent.get_unique_tableau(merged_tableau)
            merged_tableau.parent = parent
        return merged_tableau

    def copy(self) -> "Tableau":
        """Return a shallow copy of this tableau node"""
        return Tableau.merge(self, parent=self.parent)

    def __eq__(self, other: "Tableau") -> bool:
        if not isinstance(other, Tableau):
            return False
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        """The hash of a tableau should be the ordered formulas, followed by ordered entities"""
        ordered_formulas: Tuple[S.Formula] = tuple(
            sorted(self.formulas, key=S.Formula.__str__)
        )
        ordered_entities: Tuple[S.Term] = tuple(sorted(map(str, self.entities)))
        return hash((ordered_formulas, ordered_entities))

    def get_unique_tableau(self, leaf: "Tableau") -> "Tableau":
        """Given a leaf without a parent, create a leaf that only contains
        new formulas and entities to this branch
        """
        formulas = set(leaf.formulas)
        entities = set(leaf.entities)
        formulas.difference_update(self.branch_formulas)
        entities.difference_update(self.branch_entities)
        return Tableau(
            list(formulas),
            list(entities),
            parent=leaf.parent,
            dispatched_formulas=leaf.dispatched_formulas,
        )


Axiom = Callable[[Tableau], Optional[Tableau]]
