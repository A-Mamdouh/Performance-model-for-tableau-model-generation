from abc import abstractmethod
from functools import reduce
from operator import add
from .syntax import *
from .tableau import *
from .calculus import *
from .narrator import *
from .heuristic import Heuristic, ContextObject, EventEmbedding
from typing import Optional, List, Generator, Iterable
from dataclasses import dataclass, field
from queue import PriorityQueue


@dataclass(order=True)
class TableauSearchNode:
    priority: int
    sentence_depth: int = field(compare=False)
    # Maximal tableau
    tableau: Tableau = field(compare=False)
    parent: Optional["TableauSearchNode"] = field(compare=False, default=None)


class InferenceAgent:
    def __init__(self):
        self.sentences: List[Sentence] = []
        self.nodes: PriorityQueue[TableauSearchNode] = PriorityQueue()
        self.nodes.put(self._create_initial_node())

    def _create_initial_node(self) -> TableauSearchNode:
        return TableauSearchNode(
            priority=0, sentence_depth=0, tableau=self._create_axioms()
        )

    @staticmethod
    def _create_axioms() -> Tableau:
        return Tableau([])

    def search(self, narrator: Narrator) -> Generator[TableauSearchNode, None, bool]:
        for sentence in narrator:
            self.sentences.append(sentence)
            if self.nodes.empty():
                return False
            current_model = self.nodes.get()
            yield current_model
            self._extend_model(current_model)
        while not self.nodes.empty():
            current_model = self.nodes.get()
            yield current_model
            self._extend_model(current_model)
        return True

    def _extend_model(self, model: TableauSearchNode) -> None:
        if model.sentence_depth == len(self.sentences):
            return
        # Get next sentence
        sentence = self.sentences[model.sentence_depth]
        # Check for new entities
        noun_constant = Constant(Term.Sort.AGENT, sentence.noun)
        verb_constant = Constant(Term.Sort.TYPE, sentence.verb.inf)
        new_entities: List[Term] = []
        if noun_constant not in model.tableau.branch_entities:
            new_entities.append(noun_constant)
        if verb_constant not in model.tableau.branch_entities:
            new_entities.append(verb_constant)
        # Get focused formulas
        focused_formulas = reduce(
            add, map(sentence.get_formulas, sentence.get_focuses()), []
        )
        # Calculate next round of maximal tableaus
        for formula in focused_formulas:
            new_tableau = Tableau([formula], new_entities, model.tableau)
            for model_tableau in generate_models(new_tableau):
                self.nodes.put(
                    self.make_search_node(
                        model.sentence_depth + 1, model_tableau, model
                    )
                )

    @abstractmethod
    def make_search_node(
        self, sentence_depth: int, model_tableau: Tableau, parent: TableauSearchNode
    ) -> TableauSearchNode:
        pass


class DFSAgent(InferenceAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_id = 0

    def make_search_node(
        self, sentence_depth: int, model_tableau: Tableau, parent: TableauSearchNode
    ) -> TableauSearchNode:
        self.node_id -= 1
        return TableauSearchNode(
            priority=self.node_id,
            sentence_depth=sentence_depth,
            tableau=model_tableau,
            parent=parent,
        )


class BFSAgent(InferenceAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_id = 0

    def make_search_node(
        self, sentence_depth: int, model_tableau: Tableau, parent: TableauSearchNode
    ) -> TableauSearchNode:
        self.node_id += 1
        return TableauSearchNode(
            priority=self.node_id,
            sentence_depth=sentence_depth,
            tableau=model_tableau,
            parent=parent,
        )


@dataclass
class HeuristicTableauSearchNode(TableauSearchNode):
    context_object: ContextObject = field(compare=False, default=None)


class HeuristicAgent(InferenceAgent):
    def __init__(self, heuristic: Heuristic, *args, **kwargs):
        self._heuristic = heuristic
        super().__init__(*args, **kwargs)

    def _create_initial_node(self) -> HeuristicTableauSearchNode:
        return HeuristicTableauSearchNode(
            priority=0,
            sentence_depth=0,
            tableau=self._create_axioms(),
            context_object=self._heuristic.get_empty_context(),
        )

    def make_search_node(
        self,
        sentence_depth: int,
        model_tableau: Tableau,
        parent: HeuristicTableauSearchNode,
    ) -> HeuristicTableauSearchNode:
        # Get embedding of the current branch
        # Embedding will be all literals about events that were relevant between the current model and previous model
        parent_event_literals = [str(x) for x in parent.tableau.branch_event_literals]
        new_event_literals = [literal for literal in model_tableau.branch_event_literals if str(literal) not in parent_event_literals]
        # Group literals by events
        grouped_literals = {}
        for literal in new_event_literals:
            l = literal
            if isinstance(literal, Not):
                l = literal.formula
            event = l.args[0]
            found = grouped_literals.get(str(event))
            if found is None:
                found = (event, [])
                grouped_literals[str(event)] = found
            literals_list = found[1]
            literals_list.append(literal)
        # Create event embeddings
        event_embeddings = map(
            lambda x: EventEmbedding(x[0], x[1]), grouped_literals.values()
        )
        # Pass to heuristic for scoring
        new_context, branch_score = self._heuristic.score_branch(previous_context=parent.context_object, event_embeddings=event_embeddings)
        # Return new node
        return HeuristicTableauSearchNode(priority=branch_score, sentence_depth=sentence_depth, tableau=model_tableau, parent=parent, context_object=new_context)


def main():
    run = Verb("run", "ran")
    sleep = Verb("sleep", "slept")
    eat = Verb("eat", "ate")
    john = "john"
    bob = "bob"
    mary = "mary"
    story = [
        NounVerbSentence(john, eat),
        NounVerbSentence(bob, eat),
        # NounAlwaysVerbSentence(bob, sleep),
        # NounNotVerbSentence(mary, run),
        # NounVerbSentence(bob, eat),
    ]
    narrator = Narrator(story)
    heuristic = Heuristic()
    inference_agent = HeuristicAgent(heuristic=heuristic)
    n_models = 0
    for model in inference_agent.search(narrator):
        # print("model:", *model.get_model(), sep=" ", end="\n")
        print(
            f"model @ {model.sentence_depth}: ",
            *(
                x.annotation
                for x in reversed(model.tableau.branch_formulas)
                if x.annotation
            ),
            sep="\n ",
            end="\n\n",
        )
        print(
            "Entities:",
            *(str(x) for x in reversed(model.tableau.branch_entities)),
            sep="\n ",
        )
        print("-" * 30)
        n_models += 1
    print(n_models)


if __name__ == "__main__":
    main()
