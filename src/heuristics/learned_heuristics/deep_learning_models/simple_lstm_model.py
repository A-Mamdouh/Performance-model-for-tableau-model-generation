"""This module contains the implementation of a neural network scoring agent for the learned heuristic class"""

from dataclasses import dataclass
from functools import reduce
import operator
from typing import List, Optional, Tuple

import torch

from src.logic.tableau import EventInformation
from src.search.search_node import TableauSearchNode


@dataclass
class InputData:
    """Data needed for the heuristic input. This is an internal utility class"""

    model_depth: int
    events_data: List[EventInformation]

    @classmethod
    def from_search_node(cls, node: TableauSearchNode) -> "InputData":
        """Create input data object from search node"""
        previous = []
        if node.parent:
            previous = list(node.parent.tableau.branch_event_info)
        current_events = list(node.tableau.branch_event_info)
        n_new_events = len(current_events) - len(previous)
        current_events = current_events[:n_new_events]
        return cls(model_depth=node.sentence_depth, events_data=current_events)

    def get_encoding_tensor(self) -> torch.Tensor:
        """Return a tensor E x 3 where E is the number of events.
        Every row has the tuple <agent, type> encoding"""
        output = []
        for event_data in self.events_data:
            agent_encoding = reduce(
                operator.xor,
                map(lambda f: hash(f.formula.agent.name), event_data.negative_agents),
                0,
            )
            positive_agents = list(event_data.positive_agents)
            if positive_agents:
                agent_encoding = hash(positive_agents[0].agent.name)

            type_encoding = reduce(
                operator.xor,
                map(lambda f: hash(f.formula.type_.name), event_data.negative_types),
                0,
            )
            positive_types = list(event_data.positive_types)
            if positive_types:
                type_encoding = hash(positive_types[0].type_.name)
            output.append((self.model_depth, type_encoding, agent_encoding))
        return torch.tensor(output, dtype=torch.float32)


@dataclass
class ModelConfig:
    """Configuration for DL model"""

    input_size: int = 3
    latent_size: int = 64
    hidden_size: int = 100
    output_size: int = 1
    num_layers: int = 2
    dropout: float = 0.0
    bidirectional: bool = False
    device: str | torch.device = "cuda"


class Model(torch.nn.Module):
    """Deep learning model.
    This model uses a linear feature extraction block,
    followed by an LSTM block, then a prediction head for the score output
    """

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self._cfg = cfg
        self.features = torch.nn.Sequential(
            torch.nn.Linear(cfg.input_size, cfg.latent_size),
            torch.nn.ReLU(),
            torch.nn.Dropout1d(cfg.dropout),
        )
        self.memory_unit = torch.nn.GRU(
            input_size=cfg.latent_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
        )
        self.prediction_head = torch.nn.Sequential(
            torch.nn.Linear(cfg.hidden_size, cfg.output_size),
            torch.nn.ReLU(),
        )
        self._module_list = torch.nn.ModuleList(
            (self.features, self.memory_unit, self.prediction_head)
        )
        self.to(self._cfg.device)

    def from_search_node(
        self, node: TableauSearchNode, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return new score and context tuple from search node and previous context.
        This acts as a blackbox API for the heuristic module and the rest of the project
        """
        input_tensor = InputData.from_search_node(node).get_encoding_tensor()
        # If nothing new is in this node, return a copy of the input
        if len(input_tensor) == 0:
            return node.parent.priority, context.clone()
        return self.forward(input_tensor, context)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a tuple of the score and new context (hidden state)"""
        x = x.to(device=self._cfg.device)
        h = h.to(device=self._cfg.device)
        features = self.features(x)
        y, h_new = self.memory_unit(features, h)
        score = self.prediction_head(y)
        return score, h_new

    def get_initial_h(self) -> torch.Tensor:
        """Return a tensor for the initial history"""
        cfg: ModelConfig = self._cfg
        d = 2 if cfg.bidirectional else 1
        return torch.zeros(
            (
                d * cfg.num_layers,
                cfg.hidden_size,
            ),
            dtype=torch.float32,
            device=cfg.device,
        )
