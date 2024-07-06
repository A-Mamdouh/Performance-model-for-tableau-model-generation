"""Implementation of the Tableau node data structure and helping classes"""

# pylint: disable=invalid-name
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple
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
    def words(self) -> Iterable[str]:
        """A list of words used in the current node"""
        return S.get_words(self.literals)

    @property
    def branch_words(self) -> Iterable[str]:
        """A list of words used in the whole branch starting with the current node"""
        if self.parent is None:
            return self.words
        return itertools.chain(self.words, self.parent.branch_words)

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
        # Collect all unique formulas
        formulas = set(
            itertools.chain(*map(lambda tableau: tableau.formulas, tableaus))
        )
        # Collect all entities
        entities = set(
            itertools.chain(*map(lambda tableau: tableau.entities, tableaus))
        )
        if parent:
            formulas.difference_update(parent.branch_formulas)
            entities.difference_update(parent.branch_entities)
        # Create a tableau from the merged formulas and entities, with the passed parent
        merged_tableau = Tableau(formulas=formulas, entities=entities, parent=parent)
        return merged_tableau

    def copy(self) -> "Tableau":
        """Return a shallow copy of this tableau node"""
        return self.merge(self, parent=self.parent)

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
