"""This module implements the minimum events heuristic.
This heuristics favors models with less events.
"""

from typing import Iterable, Tuple

from src.heuristics.base_heuristic import Heuristic, ContextObject
import src.logic.tableau as T


class MinEvents(Heuristic):
    """This heuristic prefers models with less entities"""

    def score_branch(
        self, previous_context, branch_embeddings: Iterable[T.EventInformation]
    ) -> Tuple[ContextObject, float]:
        context = super().score_branch(previous_context, branch_embeddings)[0]
        return context, len(list(branch_embeddings))
