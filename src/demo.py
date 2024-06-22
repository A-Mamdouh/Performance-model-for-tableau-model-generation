"""Demo showing an agent output"""

from src.heuristics.highest_salience_first import AverageSalience
from src.heuristics.learned_heuristics.deep_learning_models.simple_gru_model import GRUModel
from src.heuristics.min_events import MinEvents
import src.narration as N
from src.logic.syntax import Formula
from src.search.informed_agents import GreedyAgent


def main():
    """Demo entry point"""
    run = N.Verb("run", "ran")
    sleep = N.Verb("sleep", "slept")
    eat = N.Verb("eat", "ate")
    john = "john"
    bob = "bob"
    mary = "mary"
    story = [
        N.NounVerbSentence(john, eat),
        N.NounVerbSentence(bob, eat),
        N.NounNotVerbSentence(bob, run),
        N.NounAlwaysVerbSentence(bob, sleep),
        N.NounNotVerbSentence(mary, run),
        N.NounVerbSentence(bob, eat),
    ]
    narrator = N.Narrator(story)
    # heuristic = LSTMModel()
    # heuristic.eval()
    # heuristic = MinEvents()
    heuristic = AverageSalience()
    inference_agent = GreedyAgent(heuristic=heuristic)
    n_models = 0
    for model in inference_agent.search(narrator):
        # print("model:", *model.get_model(), sep=" ", end="\n")
        print(
            f"model @ {model.sentence_depth} > {model.priority:.6f}: ",
            *(
                Formula.__str__(x)
                for x in reversed(list(model.tableau.branch_formulas))
                if x.annotation
            ),
            sep="\n ",
            end="\n\n",
        )
        print(
            "Entities:",
            *(str(x) for x in reversed(list(model.tableau.branch_entities))),
            sep="\n ",
        )
        print("Saliences:", model.salience_records, sep="\n")
        print("-" * 30)
        n_models += 1
    print(model.salience_records)


if __name__ == "__main__":
    main()
