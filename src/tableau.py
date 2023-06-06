from language import Formula, Atom
from typing import Optional, FrozenSet, List
from dataclasses import dataclass, field


## Declarations
class TaggedFormula:
    pass


class Tableau:
    pass


class BottomNode(Tableau):
    pass


## Definitions


@dataclass(frozen=True)
class TaggedFormula:
    form: Formula
    tag: bool

    def __eq__(self, f2: TaggedFormula) -> bool:
        if not isinstance(f2, TaggedFormula):
            return False
        return self.tag == f2.tag and self.form == f2.form


def tagTrue(f: Formula) -> TaggedFormula:
    return TaggedFormula(f, True)


def tagFalse(f: Formula) -> TaggedFormula:
    return TaggedFormula(f, False)


@dataclass(frozen=True)
class Tableau:
    # Set of formulas currently held in the tableau
    node_formulas: FrozenSet[TaggedFormula]
    # Production rule that produced the branch (default to initial if parent is not provided)
    rule: Optional[str] = "Initial"
    # Parent node of the tableau (None if it's a root node)
    parent: Optional[Tableau] = None
    # List of children of the node to keep track of the entire tree struture
    children: List[Tableau] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        # Cannot provide a parent without a rule
        assert not ((not self.parent is None) and self.rule == "Initial")
        # Type assertions
        assert self.rule is None or isinstance(self.rule, str)
        assert all(map(lambda x: isinstance(x, TaggedFormula), self.node_formulas))
        # If parent is provided, add child to the tree
        if not self.parent is None:
            self.parent._add(self)

    def _add(self, t: Tableau) -> Tableau:
        # Inner method. Adds tableau to children
        assert isinstance(t, Tableau)
        self.children.append(t)
        return t

    def __contains__(self, item: TaggedFormula) -> bool:
        """Check if a tagged formula is in the tableau and all parents"""
        if item in self.node_formulas:
            return True
        if self.parent is None:
            return False
        return item in self.parent

    def contains_all(self, *items: TaggedFormula) -> bool:
        return self._contains_all(frozenset(items))

    def _contains_all(self, items: FrozenSet[TaggedFormula]) -> bool:
        """Checks if a set of elements exist in the entire tableau tree"""
        diff = items - self.node_formulas
        if len(diff) == 0:
            return True
        if self.parent is None:
            return False
        return self.parent._contains_all(diff)

    def get_model(self) -> FrozenSet[TaggedFormula]:
        """Returns a model of tagged atoms"""
        model = {x for x in self.formulas if isinstance(x, Atom)}
        if not self.parent is None:
            model.union(self.parent.get_model())
        return frozenset(model)

    @property
    def formulas(self) -> List[TaggedFormula]:
        """Returns a list of all tagged formula in the tableau tree"""
        if self._formulas is None:
            return self._formulas
        return [*self._formulas, *self._parent.formulas]

    def __eq__(self, t2: Tableau) -> bool:
        """
        Equality between tableau nodes.
        *NOTE: This does not compare entire trees*
        """
        if not isinstance(t2, Tableau):
            return False
        return self.node_formulas == t2.node_formulas


class BottomNode(Tableau):
    """Bottom node tableau implementation as a special case of the tableau"""

    def __init__(self, rule: str, parent: Tableau) -> None:
        super().__init__(frozenset(), rule, parent)

    def add(self, rule: str, output: FrozenSet[TaggedFormula]) -> Tableau:
        raise RuntimeError("Cannot add to a closed branch")

    def get_model(self) -> FrozenSet[TaggedFormula]:
        raise RuntimeError("Cannot generate model from a closed branch")

"""
TODO:
    - Add rules for quantifiers
"""