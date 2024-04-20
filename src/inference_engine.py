from .syntax import *
from .tableau import *
from .calculus import *
from .narrator import *
from .heuristic import Heuristic
from typing import Optional, List, Generator, Tuple


class InferenceAgent:
    def __init__(self, heuristic_agent: Heuristic, tableau: Optional[Tableau] = None):
        self.tableau = tableau
        if tableau is None:
            self.tableau = self._create_axioms()
        self.heuristic_agent = heuristic_agent

    @staticmethod
    def _create_axioms() -> Tableau:
        # First axiom: any event has only one type "A e.[ A t1.[ ty(e, t1) => (A t2.[ty(e, t2) => t1=t2 ]) ] ]"
        axiom1: Formula = Forall(
            lambda e: Forall(
                lambda t1: Implies(
                    Type_(e, t1),
                    Forall(
                        lambda t2: Implies(Type_(e, t2), Eq(t1, t2)), Term.Sort.TYPE
                    ),
                ),
                Term.Sort.TYPE,
            ),
            Term.Sort.EVENT,
        )
        # Second axiom: any event has only one agent "A e.[ A a1.[ ag(e, a1) => (A a2.[ag(e, a2) => a1=a2 ]) ] ]"
        axiom2: Formula = Forall(
            lambda e: Forall(
                lambda a1: Implies(
                    Agent(e, a1),
                    Forall(
                        lambda a2: Implies(Agent(e, a2), Eq(a1, a2)), Term.Sort.AGENT
                    ),
                ),
                Term.Sort.AGENT,
            ),
            Term.Sort.EVENT,
        )
        return Tableau(
            [
                # axiom1,
                # axiom2,
            ]
        )

    def dfs(
        self, narrator: Narrator, tableau: Optional[Tableau] = None
    ) -> Generator[Tableau, None, None]:
        if tableau is None:
            tableau = self.tableau
        try:
            sentence: Sentence = next(iter(narrator))
        except StopIteration:
            return

        # Check for new entities
        noun_constant = Constant(Term.Sort.AGENT, sentence.noun)
        verb_constant = Constant(Term.Sort.TYPE, sentence.verb.inf)
        new_entities: List[Term] = []
        if noun_constant not in tableau.branch_entities:
            new_entities.append(noun_constant)
        if verb_constant not in tableau.branch_entities:
            new_entities.append(verb_constant)
        
        # Order readings by focus score
        scored_focused_formulas: List[Tuple[float, Formula]] = []
        for focus in Sentence.Focus:
            for formula in sentence.get_formulas(focus):
                score = self.heuristic_agent.scoreFormula(formula, tableau)
                scored_focused_formulas.append((score, formula))
        for _, formula in sorted(scored_focused_formulas, key=lambda x: x[0]):
            new_tableau = Tableau([formula], new_entities, tableau)
            # Create model for current sentence
            for model in generate_models(new_tableau):
                yield model
                # Get models for future sentences
                for model2 in self.dfs(narrator.copy(), model):
                    yield model2


def main():
    run = Verb("run", "ran")
    sleep = Verb("sleep", "slept")
    eat = Verb("eat", "ate")
    john = "john"
    bob = "bob"
    mary = "mary"
    story = [
        NounVerbSentence(john, eat),
        NounAlwaysVerbSentence(bob, sleep),
        NounNotVerbSentence(mary, run),
        NounVerbSentence(bob, eat),
    ]
    narrator = Narrator(story)
    heuristic = Heuristic()
    inference_agent = InferenceAgent(heuristic_agent=heuristic)
    n_models = 0
    for model in inference_agent.dfs(narrator):
        # print("model:", *model.get_model(), sep=" ", end="\n")
        print(*(x.annotation for x in model.branch_formulas if x.annotation), sep="\n", end="\n")
        # print("Entities:", *(str(x) for x in model.branch_entities), sep="\n")
        print("-" * 30)
        n_models += 1
    print(n_models)


if __name__ == "__main__":
    main()
