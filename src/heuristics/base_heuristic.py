"""Base classes and types for heuristics"""

from typing import Any, Tuple

from src.search.search_node import TableauSearchNode

ContextObject = Any


class Heuristic:
    """Base class of heuristics for the search agent"""

    # pylint: disable=W0613:unused-argument
    def score_node(
        self, previous_context, search_node: TableauSearchNode
    ) -> Tuple[ContextObject, float]:
        """Takes the previous context object and the current search node,
        then returns the new context object and a numerical score of the search node"""
        return None, 0.0

    def get_empty_context(self) -> ContextObject:
        """Returns a context object that corresponds to the start of a dialogue"""
        return None
