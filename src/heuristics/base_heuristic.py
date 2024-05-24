"""Base classes and types for heuristics"""

from typing import Any, Iterable, Tuple

import src.logic.tableau as T

ContextObject = Any


class Heuristic:
    """Base class of heuristics for the search agent"""

    # pylint: disable=W0613:unused-argument
    def score_branch(
        self, previous_context, branch_embeddings: Iterable[T.EventInformation]
    ) -> Tuple[ContextObject, float]:
        """Takes the previous context object and the current branch embedding,
        then returns the new context object and a numerical score of the branch"""
        return None, 0.0

    def get_empty_context(self) -> ContextObject:
        """Returns a context object that corresponds to the start of a dialogue"""
        return None