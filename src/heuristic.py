from . import syntax
from dataclasses import dataclass
from typing import Iterable, Any, Tuple


ContextObject = Any


@dataclass
class EventEmbedding:
    event: syntax.Term
    event_literals: Iterable[syntax.Formula]


class Heuristic:

    def score_branch(
        self, previous_context, event_embeddings: Iterable[EventEmbedding]
    ) -> Tuple[ContextObject, float]:
        event_embeddings_strs = [
            (str(e.event), [str(x) for x in e.event_literals])
            for e in event_embeddings
        ]
        return None, 0.0

    def get_empty_context(self) -> ContextObject:
        return None
