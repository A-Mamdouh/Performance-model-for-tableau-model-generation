import src.tableau as T
from typing import Any, Iterable, Tuple, override


ContextObject = Any


class Heuristic:

    def score_branch(
        self, previous_context, branch_embeddings: Iterable[T.EventInformation]
    ) -> Tuple[ContextObject, float]:
        return None, 0.0

    def get_empty_context(self) -> ContextObject:
        return None


class MinEvents(Heuristic):
    @override
    def score_branch(self, previous_context, branch_embeddings: Iterable[T.EventInformation]) -> Tuple[ContextObject, float]:
        context = super().score_branch(previous_context, branch_embeddings)[0]
        return context, len(list(branch_embeddings))
