"""Implementation of the Tableau node data structure and helping classes"""

# pylint: disable=invalid-name
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple
import itertools

import src.logic.syntax as S

__all__ = ("Tableau",)


@dataclass
class EventInformation:
    """Information about an event in a tableau branch"""

    event: S.Term
    positive_types: Iterable[S.Type_] = field(default_factory=list)
    positive_agents: Iterable[S.Agent] = field(default_factory=list)
    negative_types: Iterable[S.Not] = field(default_factory=list)
    negative_agents: Iterable[S.Not] = field(default_factory=list)

    @property
    def all_literals(self) -> Iterable[S.Literal]:
        """All literals containing the event"""
        return (
            *self.positive_types,
            *self.positive_agents,
            *self.negative_types,
            *self.negative_agents,
        )


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
    #: True if this node makes the current branch a closed branch
    closing: bool = False

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
    def events(self) -> Iterable[S.Term]:
        """All events in the current node"""
        return filter(lambda entity: entity.sort == S.Term.Sort.EVENT, self.entities)

    @property
    def branch_events(self) -> Iterable[S.Term]:
        """All events from this node up to the root. Starting with this node"""
        if self.parent is None:
            return filter(
                lambda entity: entity.sort == S.Term.Sort.EVENT, self.entities
            )
        return filter(
            lambda entity: entity.sort == S.Term.Sort.EVENT,
            itertools.chain(self.entities, self.parent.branch_entities),
        )

    @property
    def literals(self) -> Iterable[S.Literal]:
        """All literals in the current node"""
        return filter(S.is_literal, self.formulas)

    @property
    def branch_literals(self) -> Iterable[S.Literal]:
        """All literals from this node up to the root. Starting with this node"""
        return filter(S.is_literal, self.branch_formulas)

    @property
    def event_info(self) -> Iterable[EventInformation]:
        """List of event info in the current node"""
        return self._get_event_information_from_literals_list(self.literals)

    @property
    def branch_event_info(self) -> Iterable[EventInformation]:
        """List of event info from this node up to the root."""
        return self._get_event_information_from_literals_list(self.branch_literals)

    @staticmethod
    def _get_event_information_from_literals_list(
        literals_list: Iterable[S.Literal],
    ) -> Iterable[EventInformation]:
        """Return event information from the input list of literals"""
        event_information: Dict[S.Term, EventInformation] = {}
        for literal in literals_list:
            # Add Event Agent
            if isinstance(literal, S.Agent):
                event = literal.event
                info_record = event_information.get(event)
                if info_record is None:
                    info_record = EventInformation(event=event)
                    event_information[event] = info_record
                info_record.positive_agents.append(literal)
            # Add Event Type
            elif isinstance(literal, S.Type_):
                event = literal.event
                info_record = event_information.get(event)
                if info_record is None:
                    info_record = EventInformation(event=event)
                    event_information[event] = info_record
                info_record.positive_types.append(literal)
            elif isinstance(literal, S.Not):
                # Add negative Event Agent
                if isinstance(literal.formula, S.Agent):
                    event = literal.formula.event
                    info_record = event_information.get(event)
                    if info_record is None:
                        info_record = EventInformation(event=event)
                        event_information[event] = info_record
                    info_record.negative_agents.append(literal)
                # Add negative Event Type
                elif isinstance(literal.formula, S.Type_):
                    event = literal.formula.event
                    info_record = event_information.get(event)
                    if info_record is None:
                        info_record = EventInformation(event=event)
                        event_information[event] = info_record
                    info_record.negative_types.append(literal)
        return event_information.values()

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
        merged_tableau = Tableau(formulas, entities, parent, False)
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
