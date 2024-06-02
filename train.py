# %%
from typing import List

import matplotlib.pyplot as plt
import torch
import tqdm

from src.heuristics.learned_heuristics.deep_learning_models.simple_gru_model import GRUModel, GRUModelConfig
from src.heuristics.min_events import MinEvents
from src import narration
from src.search.informed_agents import GreedyAgent
from src.search.search_node import HeuristicTableauSearchNode
from src.logic.syntax import Formula

# %%
def make_narrator() -> narration.Narrator:
    run = narration.Verb("run", "ran")
    sleep = narration.Verb("sleep", "slept")
    eat = narration.Verb("eat", "ate")
    john = "john"
    bob = "bob"
    mary = "mary"
    story = [
        narration.NounVerbSentence(john, eat),
        narration.NounVerbSentence(bob, eat),
        narration.NounNotVerbSentence(bob, run),
        narration.NounAlwaysVerbSentence(bob, sleep),
        narration.NounNotVerbSentence(mary, run),
        narration.NounVerbSentence(bob, eat),
    ]
    return narration.Narrator(story)
narrator = make_narrator()

# %%
def get_target_models(agent: GreedyAgent, narrator: narration.Narrator) -> List[HeuristicTableauSearchNode]:
    return list(agent.search(narrator.copy()))

models = get_target_models(GreedyAgent(MinEvents()), narrator)

# %%
def get_training_sequences(agent: GreedyAgent, solved_tree: List[HeuristicTableauSearchNode], device: str | torch.device) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
    # First, collect the model leaves
    leaves: list[HeuristicTableauSearchNode] = [*solved_tree]
    for node in solved_tree:
        if node.parent in leaves:
            leaves.remove(node.parent)
    # Collect sequences
    sequences: List[List[HeuristicTableauSearchNode]] = []
    labels: List[List[float]] = []
    for leaf in leaves:
        sequence: List[HeuristicTableauSearchNode] = []
        sequence_labels: List[float] = []
        current: HeuristicTableauSearchNode = leaf
        while current:
            sequence.append(GRUModel.get_node_encoding(current))
            sequence_labels.append(current.priority)
            current = current.parent
        sequence = torch.concat(sequence[::-1], dim=0)
        sequences.append(sequence)
        sequence_labels = torch.tensor(sequence_labels, dtype=torch.float32, device=device)[:, None]
        labels.append(sequence_labels)
    return sequences, labels
gru_model = GRUModel(GRUModelConfig(
    latent_size = 64,
    hidden_size = 128,
    dropout=0.5,
    bidirectional=False
))
gru_agent = GreedyAgent(heuristic=gru_model)
train_sequences, train_labels = get_training_sequences(gru_agent, models, gru_model._cfg.device)

# %%
def train(model: GRUModel, train_sequences, train_labels, iters: int, lr: float):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    flat_labels = torch.stack(train_labels, dim=0)
    iterator = tqdm.trange(iters, desc="training")
    for _ in iterator:
        optimizer.zero_grad()
        out = model.forward_batch(train_sequences)[0]
        flat_output = torch.stack(out, dim=0)
        loss = loss_fn(flat_output, flat_labels)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        iterator.set_description_str(f"Training loss: {loss_value:.4f}")
        loss_history.append(loss_value)
    return loss_history
loss_history = train(gru_model, train_sequences, train_labels, 30, lr=1e-4)
plt.plot(loss_history)
plt.show()

# %%
gru_model.eval()
with torch.inference_mode():
    trained_models = get_target_models(gru_agent, narrator)

# %%
with open("training_output", "w") as fp:
    for model in trained_models:
        print(
            f"model @ {model.sentence_depth} - {model.priority[0,0]:.6f}: ",
            *(
                Formula.__str__(x)
                for x in reversed(list(model.tableau.branch_formulas))
                if x.annotation
            ),
            sep="\n ",
            end="\n\n",
            file=fp,
        )
        print(
            "Entities:",
            *(str(x) for x in reversed(list(model.tableau.branch_entities))),
            sep="\n ",
            file=fp,
        )
        print("-" * 30, file=fp)


