"""This module extends the base Tableau implementation for event semantics logic"""

from dataclasses import dataclass, field
from typing import Dict, Iterable

from src.logic.simple_event_semantics.syntax import Filters
from src.logic.base.syntax import AppliedPredicate, Literal, Not, Term
from src.logic.base.tableau import Tableau as BaseTableau


@dataclass
class EventInformation:
    """Information about an event in a tableau branch"""

    event: Term
    positive_types: Iterable[AppliedPredicate] = field(default_factory=list)
    positive_agents: Iterable[AppliedPredicate] = field(default_factory=list)
    negative_types: Iterable[Not] = field(default_factory=list)
    negative_agents: Iterable[Not] = field(default_factory=list)

    @property
    def all_literals(self) -> Iterable[Literal]:
        """All literals containing the event"""
        return (
            *self.positive_types,
            *self.positive_agents,
            *self.negative_types,
            *self.negative_agents,
        )


class Tableau(BaseTableau):
    """Extension of the base tableau to add event data"""

    @property
    def events(self) -> Iterable[Term]:
        """A list of all event terms in the current node"""
        return filter(Filters.is_event_term, self.entities)

    @property
    def branch_events(self) -> Iterable[Term]:
        """All events from this node up to the root. Starting with this node"""
        if self.parent is None:
            return self.events
        return *self.events, *self.parent.branch_events

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
        literals_list: Iterable[Literal],
    ) -> Iterable[EventInformation]:
        """Return event information from the input list of literals"""
        event_information: Dict[Term, EventInformation] = {}
        for literal in literals_list:
            # Add Event Agent
            if Filters.is_agent_formula(literal):
                event = literal.args[0]
                info_record = event_information.get(event)
                if info_record is None:
                    info_record = EventInformation(event=event)
                    event_information[event] = info_record
                info_record.positive_agents.append(literal)
            # Add Event Type
            elif Filters.is_type_formula(literal):
                event = literal.args[0]
                info_record = event_information.get(event)
                if info_record is None:
                    info_record = EventInformation(event=event)
                    event_information[event] = info_record
                info_record.positive_types.append(literal)
            elif isinstance(literal, Not):
                # Add negative Event Agent
                if Filters.is_agent_formula(literal.formula):
                    event = literal.formula.args[0]
                    info_record = event_information.get(event)
                    if info_record is None:
                        info_record = EventInformation(event=event)
                        event_information[event] = info_record
                    info_record.negative_agents.append(literal)
                # Add negative Event Type
                elif Filters.is_type_formula(literal.formula):
                    event = literal.formula.event
                    info_record = event_information.get(event)
                    if info_record is None:
                        info_record = EventInformation(event=event)
                        event_information[event] = info_record
                    info_record.negative_types.append(literal)
        return event_information.values()
