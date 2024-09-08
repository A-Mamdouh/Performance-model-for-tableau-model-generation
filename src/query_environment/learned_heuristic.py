from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from src.heuristics.learned_heuristics.deep_learning_models.word_encoder import WordEncoder


class WitnessEncoder(nn.Module):
    """This encoder produces encoding encapsulating an even given its order in a dialog and its details"""
    def __init__(self) -> None:
        pass


class HeuristicModel(nn.Module):
    """This model estimates the search heuristic"""
    @dataclass
    class Config:
        """Model config"""
        n_tokens: int = 2 # tokens: subject, object
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
        self.event_encoding_length = self.encoder.word_encoding_length * self.cfg.n_tokens
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
        self._module_list = torch.nn.ModuleList(
            self.feature_extraction_backbone,
            self.memory_unit,
            self.prediction_head,
        )
        self.to(self._cfg.device)

    
def build_feature_extraction_backbone(
        input_size: int,
        latent_size: int=64,
        dropout: float=0.0,
        feature_extraction_depth: int=2,
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
    prediction_head_depth: int=2,
    dropout: float = 0.0
) -> nn.Module:
    layers = []
    for _ in range(prediction_head_depth):
        inner_layers = [
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        ]
        if dropout:
            inner_layers.append(torch.nn.Dropout1d(dropout))
        layers.extend(inner_layers)
    layers.append(torch.nn.Linear(hidden_size, output_size))
    return torch.nn.Sequential(*layers)
