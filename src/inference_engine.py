from .syntax import *
from .tableau import *
from .calculus import *
from .narrator import *
from typing import Optional, List, Generator

class InferenceAgent:
    def __init__(self, tableau: Optional[Tableau]=None):
        self.tableau = tableau
        if tableau is None:
            self.tableau = self._create_axioms()
    
    @staticmethod
    def _create_axioms() -> Tableau:
        # First axiom: any event has only one type "A e.[ A t1.[ ty(e, t1) => (A t2.[ty(e, t2) => t1=t2 ]) ] ]"
        axiom1: Formula = Forall(
            lambda e: Forall(
                lambda t1: Implies(Type_(e, t1), Forall(
                    lambda t2: Implies(Type_(e, t2), Eq(t1, t2)),
                    Term.Sort.TYPE
                )),
                Term.Sort.TYPE
            ), Term.Sort.EVENT
        )
        # Second axiom: any event has only one agent "A e.[ A a1.[ ag(e, a1) => (A a2.[ag(e, a2) => a1=a2 ]) ] ]"
        axiom2: Formula = Forall(
            lambda e: Forall(
                lambda a1: Implies(Agent(e, a1), Forall(
                    lambda a2: Implies(Agent(e, a2), Eq(a1, a2)),
                    Term.Sort.AGENT
                )),
                Term.Sort.AGENT
            ),
            Term.Sort.EVENT
        )
        return Tableau([axiom1, axiom2])
    
    
    def dfs(self, narrator: Narrator, tableau: Optional[Tableau] = None) -> Generator[Tableau, None, None]:
        if tableau is None:
            tableau = self.tableau
        try:
            sentence: Sentence = next(iter(narrator))
        except StopIteration:
            return
        formula = sentence.get_formula(Sentence.Focus.FULL)
        # Check for new entities
        noun_constant = Constant(Term.Sort.AGENT, sentence.noun)
        verb_constant = Constant(Term.Sort.TYPE, sentence.verb.inf)
        new_entities: List[Term] = []
        if noun_constant not in tableau.branch_entities:
            new_entities.append(noun_constant)
        if verb_constant not in tableau.branch_entities:
            new_entities.append(verb_constant)
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
        NounNotVerbSentence(mary, run)
    ]
    narrator = Narrator(story)
    inference_agent = InferenceAgent()
    for model in inference_agent.dfs(narrator):
        print("model:", *model.get_model(), sep=" ")
        print(*(str(x) for x in model.branch_formulas), sep="\n")
        print("Entities:", *(str(x) for x in model.branch_entities), sep="\n")
        print("-" * 30)


if __name__ == '__main__':
    main()
