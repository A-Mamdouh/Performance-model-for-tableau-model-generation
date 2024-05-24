"""Demo showing an agent output"""

import src.narrator as N
import src.heuristics as H
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
        # NounAlwaysVerbSentence(bob, sleep),
        # NounNotVerbSentence(mary, run),
        # NounVerbSentence(bob, eat),
    ]
    narrator = N.Narrator(story)
    heuristic = H.MinEvents()
    inference_agent = GreedyAgent(heuristic=heuristic)
    n_models = 0
    for model in inference_agent.search(narrator):
        # print("model:", *model.get_model(), sep=" ", end="\n")
        print(
            f"model @ {model.sentence_depth}: ",
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
        print("-" * 30)
        n_models += 1
    print(n_models)


if __name__ == "__main__":
    main()
