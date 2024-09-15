"""Holds the implementation of the learning agent"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor

from src.heuristics.learned_heuristics.deep_learning_models.word_encoder import (
    WordEncoder,
)
import src.logic.base.syntax as S
from src.query_environment.environment import get_event_from_atom


class WitnessEncoder(nn.Module):
    """This encoder produces encoding encapsulating an even given its order in a dialog and its details"""

    def __init__(self) -> None:
        pass


class HeuristicModel(nn.Module):
    """This model estimates the search heuristic"""

    @dataclass
    class Config:
        """Model config"""

        n_tokens: int = 2  # tokens: subject, object
        latent_size: int = 64
        dropout: float = 0.0
        feature_extraction_depth: int = 2
        prediction_head_depth: int = 2
        hidden_size: int = 100
        gru_num_layers: int = 2
        bidirectional: bool = False
        output_size: int = 1
        accelerated: bool = True
        device: Optional[torch.DeviceObjType] = None

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

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = WordEncoder(device=self.cfg.device)
        self.event_encoding_length = (
            self.encoder.word_encoding_length * self.cfg.n_tokens
        )
        self.feature_extraction_backbone = build_feature_extraction_backbone(
            input_size=self.event_encoding_length,
            latent_size=self.cfg.latent_size,
            dropout=self.cfg.dropout,
            feature_extraction_depth=self.cfg.feature_extraction_depth,
        )
        self.memory_unit = nn.GRU(
            input_size=self.cfg.latent_size,
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.gru_num_layers,
            dropout=self.cfg.dropout,
            bidirectional=self.cfg.bidirectional,
        )
        self.prediction_head = build_prediction_head(
            output_size=self.cfg.output_size,
            hidden_size=self.cfg.hidden_size,
            prediction_head_depth=self.cfg.prediction_head_depth,
            dropout=self.cfg.dropout,
        )
        # pylint: disable=e1121:too-many-function-args
        self._module_list = nn.ModuleList(
            self.feature_extraction_backbone,
            self.memory_unit,
            self.prediction_head,
        )
        self.to(self._cfg.device)

    def get_encoding(self, x: Tensor, h: Tensor) -> Tensor:
        """Return the encoded tokens of x given the context vector h"""
        x = x.to(device=self._cfg.device)
        h = h.to(device=self._cfg.device)
        tokens = self.feature_extraction_backbone(x)
        return tokens

    def forward(
        self, x: Tensor, h: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Return a tuple of the score and new context (hidden state)"""
        x = x.to(device=self._cfg.device)
        h = h.to(device=self._cfg.device)
        features = self.feature_extraction_backbone(x)
        y, h_new = self.memory_unit(features, h)
        score = self.prediction_head(y)
        return score, h_new

    def score_ctx(self, h: Tensor) -> float:
        """Return a numerical score of the likelihood of the context being correct"""
        return self.prediction_head(h)

    def next_context_vector(
        self, literals: List[S.Term], h: Tensor
    ) -> Tensor:
        """
        Get the next context vector
        given the list of uncovered literals and current context vector h
        """
        # Collect event information
        literals_by_event = {}
        for literal in literals:
            # Skip non-event formulas
            event = None
            if isinstance(literal, S.Not):
                event = get_event_from_atom(literal.formula)
            else:
                event = get_event_from_atom(literal)
            record = literals_by_event.get(event)
            if not record:
                literals_by_event[event] = record = []
            record.apend(literal)
        # TODO: Resume here


def build_feature_extraction_backbone(
    input_size: int,
    latent_size: int = 64,
    dropout: float = 0.0,
    feature_extraction_depth: int = 2,
) -> nn.Module:
    layers = [
        nn.Linear(input_size, latent_size),
        nn.ReLU(),
        nn.Dropout1d(dropout),
    ]
    for _ in range(feature_extraction_depth):
        layers.extend(
            [
                nn.Linear(latent_size, latent_size),
                nn.ReLU(),
                nn.Dropout1d(dropout),
            ]
        )
    return nn.Sequential(*layers)


def build_prediction_head(
    output_size: int = 1,
    hidden_size: int = 100,
    prediction_head_depth: int = 2,
    dropout: float = 0.0,
) -> nn.Module:
    layers = []
    for _ in range(prediction_head_depth):
        inner_layers = [
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        ]
        if dropout:
            inner_layers.append(nn.Dropout1d(dropout))
        layers.extend(inner_layers)
    layers.append(nn.Linear(hidden_size, output_size))
    return nn.Sequential(*layers)
