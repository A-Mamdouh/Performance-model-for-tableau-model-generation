"""This module contains the implementation of a neural network scoring agent for the learned heuristic class"""

from dataclasses import dataclass
from functools import reduce
import operator
from typing import ClassVar, List, Optional, Tuple

import torch

from src.heuristics.base_heuristic import Heuristic
from src.heuristics.context_token import ContextToken
from src.logic.base.tableau import EventInformation
from src.search.search_node import TableauSearchNode


@dataclass
class InputData:
    """Data needed for the heuristic input. This is an internal utility class"""

    model_depth: int
    events_data: List[EventInformation]
    embedding_size: ClassVar[int] = 3

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
class GRUModelConfig:
    """Configuration for DL model"""

    feature_extraction_depth: int = 2
    gru_num_layers: int = 2
    prediction_head_depth: int = 1
    latent_size: int = 64
    hidden_size: int = 100
    output_size: int = 1
    dropout: float = 0.0
    bidirectional: bool = False
    accelerated: bool = True
    device: str | torch.device | None = None

    def __post_init__(self) -> None:
        if self.accelerated and self.device is None:
            backends = (torch.cuda, "cuda"), (torch.backends.mps, "mps")
            for backend, name in backends:
                if backend.is_available():
                    self.device = name
                    break
            else:
                self.device = "cpu"
                self.accelerated = False


class GRUModel(torch.nn.Module, Heuristic):
    """Deep learning model.
    This model uses a linear feature extraction block,
    followed by an LSTM block, then a prediction head for the score output
    """

    def __init__(self, cfg: Optional[GRUModelConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = GRUModelConfig()
        self._cfg = cfg
        self.feature_extraction_backbone = self._build_feature_extraction_backbone()
        self.memory_unit = torch.nn.GRU(
            input_size=cfg.latent_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.gru_num_layers,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
        )
        self.prediction_head = self._build_prediction_head()
        self._module_list = torch.nn.ModuleList(
            (self.feature_extraction_backbone, self.memory_unit, self.prediction_head)
        )
        self.to(self._cfg.device)
    
    def _build_feature_extraction_backbone(self) -> torch.nn.Module:
        layers = [
            torch.nn.Linear(InputData.embedding_size, self._cfg.latent_size),
            torch.nn.ReLU(),
            torch.nn.Dropout1d(self._cfg.dropout),
        ]
        for _ in range(self._cfg.feature_extraction_depth):
            layers.extend([
                torch.nn.Linear(self._cfg.latent_size, self._cfg.latent_size),
                torch.nn.ReLU(),
                torch.nn.Dropout1d(self._cfg.dropout),
            ])
        return torch.nn.Sequential(*layers)
    
    def _build_prediction_head(self) -> torch.nn.Module:
        layers = []
        for _ in range(self._cfg.prediction_head_depth):
            layers.extend([
                torch.nn.Linear(self._cfg.hidden_size, self._cfg.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout1d(self._cfg.dropout),
            ])
        layers.append(torch.nn.Linear(self._cfg.hidden_size, self._cfg.output_size))
        return torch.nn.Sequential(*layers)
    
    @staticmethod
    def _make_empty_branch_tensor(input_: InputData) -> torch.Tensor:
        """Create an input tensor that represents a model with no new event information."""
        return torch.zeros(((input_.model_depth, 0, 0),), dtype=torch.float32)

    def _from_search_node(
        self, node: TableauSearchNode, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return new score and context tuple from search node and previous context.
        This acts as a blackbox API for the heuristic module and the rest of the project
        """
        input_data: InputData = InputData.from_search_node(node)
        encoding: torch.Tensor = input_data.get_encoding_tensor()
        # If nothing new is in this node, get the empty representaiton
        if len(encoding) == 0:
            encoding = self._make_empty_branch_tensor(input_data)
        return self.forward(encoding, context)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a tuple of the score and new context (hidden state)"""
        x = x.to(device=self._cfg.device)
        h = h.to(device=self._cfg.device)
        features = self.feature_extraction_backbone(x)
        y, h_new = self.memory_unit(features, h)
        score = self.prediction_head(y)
        return score, h_new
    
    def forward_batch(self, xs: List[torch.Tensor], h0s: torch.Tensor | None = None):
        """ X is a list of sequences, h0s is a tensor with the starting hidden state for each sequence.
        h0s should have the shape (Number of sequences) x (hidden layers) x (hidden state size),
        regardless of the internal batch size first option.
        return the output for each part of the sequences and the hidden states at the end of each sequence.
        The output shapes matches the input shapes
        """
        if h0s is None:
            h0s = torch.stack((self.get_empty_context(),) * len(xs), dim=0)
        h0s = h0s.moveaxis(0, 1).to(self._cfg.device).contiguous()
        # First, get the lstm features input from the linear units
        flat_sequences = torch.concat(xs, dim=0).to(self._cfg.device)
        flat_features: torch.Tensor = self.feature_extraction_backbone(flat_sequences) # Batch Size * Sequence Length x LSTM Input Shape
        # Then, reshape back into the lstm input Length x BatchSize x latent_size and get the lstm output
        reshaped_lstm_input = flat_features.split_with_sizes(list(map(len, xs)))
        packed_feature_tensor = torch.nn.utils.rnn.pack_sequence(reshaped_lstm_input)
        lstm_output_features, lstm_hidden_output = self.memory_unit(packed_feature_tensor, h0s)
        # flatten the lstm output features to pass onto the prediction head
        unpacked_lstm_features = torch.nn.utils.rnn.unpack_sequence(lstm_output_features)
        flattened_lstm_features = torch.concat(unpacked_lstm_features, dim=0)
        prediction_output = self.prediction_head(flattened_lstm_features)
        # Reshape the prediction head output to match the input shape
        output_predictions = prediction_output.split_with_sizes(list(map(len, xs)))
        # Reshape the lstm hidden state outputo the first dimension is the batch
        output_hidden_state_batch_first = lstm_hidden_output.moveaxis(1, 0)
        return output_predictions, output_hidden_state_batch_first        

    def _get_initial_h(self) -> torch.Tensor:
        """Return a tensor for the initial history"""
        cfg: GRUModelConfig = self._cfg
        d = 2 if cfg.bidirectional else 1
        return torch.zeros(
            (
                d * cfg.gru_num_layers,
                cfg.hidden_size,
            ),
            dtype=torch.float32,
            device=cfg.device,
        )
    
    def score_node(
        self, previous_context, search_node: TableauSearchNode
    ) -> Tuple[ContextToken, float]:
        score, new_context = self._from_search_node(search_node, previous_context)
        return new_context, score

    def get_empty_context(self) -> ContextToken:
        return self._get_initial_h()
    
    @staticmethod
    def get_node_encoding(node: TableauSearchNode) -> torch.Tensor:
        """Get the branch encoding tensor. Used for training"""
        input_data: InputData = InputData.from_search_node(node)
        encoding: torch.Tensor = input_data.get_encoding_tensor()
        # If nothing new is in this node, get the empty representaiton
        if len(encoding) == 0:
            encoding = GRUModel._make_empty_branch_tensor(input_data)
        return encoding
