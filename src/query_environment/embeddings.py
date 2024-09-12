"""Handles embedding of tableau models"""

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional, Tuple


from torch import Tensor

from src.logic.base.syntax import Term


@dataclass
class EventInformation:
    _info_mgr: "EventInformationManager"
    event: Term
    subject: Term | None = None
    verb: Term | None = None
    object_: Term | None = None

    def __setattr__(self, attr_name: str, value: Any) -> None:
        super().__setattr__(attr_name, value)
        # Call update function if a public attribute is set
        if not attr_name.startswith("_"):
            self._info_mgr.update_event_info(self)

    def copy(self, **changes) -> "EventInformation":
        """Return a deep copy of this class"""
        return replace(self, changes=changes)


@dataclass
class EventEmbedding:
    """Wrapper for event information embeddings"""

    _info_mgr: "EventInformationManager"
    context_embedding: Tensor
    event: Tensor
    subject: Tensor
    verb: Tensor
    object_: Tensor | None

    def copy(self, **changes) -> "EventEmbedding":
        """Return a deep copy of this class"""
        return replace(self, changes=changes)


@dataclass
class EventInformationManager:
    """Container for event information and its embedings"""

    events_information: Dict[Term, EventInformation] = field(default_factory=dict)
    events_embeddings: Dict[Term, EventEmbedding] = field(default_factory=dict)
    context_vector: Tensor = 0
    encoding_model: Callable[[Tensor], Tensor] = None
    # TODO: Add a reference to the encoding model

    def __getitem__(self, key: Term) -> Tuple[EventInformation, EventEmbedding]:
        return self.events_information[key], self.events_embeddings[key]

    def get(self, key: Term) -> Optional[Tuple[EventInformation, EventEmbedding]]:
        if self.events_information.get(key) is not None:
            return self[key]
        info = EventInformation(self, event=key)
        embd = EventEmbedding(
            self, None, None, None, None, None
        )  # TODO: Calculate these values
        self.events_information[key] = info
        self.events_embeddings[key] = embd
        return self[key]

    @property
    def embeddings(self) -> Dict[Term, EventEmbedding]:
        return self.events_embeddings

    @property
    def info(self) -> Dict[Term, EventInformation]:
        return self.events_information

    def update_event_info(self, event_info: EventInformation) -> None:
        raise NotImplementedError()

    def update_event_embedding(self, event_embedding: EventEmbedding) -> None:
        raise NotImplementedError()

    def copy(self, **changes) -> "EventInformationManager":
        """Return a deep copy of this class"""
        copy_ = EventInformationManager(
            events_information={
                event: info.copy() for event, info in self.events_information.items()
            },
            events_embeddings={
                event: embd.copy() for event, embd in self.events_embeddings.items()
            },
        )
        return replace(copy_, changes=changes)
