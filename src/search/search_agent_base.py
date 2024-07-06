"""Implementation of the base search agent. All of the queuing and logic is done here"""

from abc import abstractmethod
from functools import reduce
from operator import add
from queue import PriorityQueue
from typing import Generator, List

import src.logic.base.calculus as C
import src.logic.base.syntax as S
import src.logic.base.tableau as T
import src.narration as N
from src.search.search_node import TableauSearchNode


class InferenceAgent:
    """This class implements a basic search agent which adds new nodes to a priority queue.
    Child classes only need to implement the function for making a new search node to
    adjust the order of node exploration
    """

    def __init__(self):
        self.sentences: List[N.Sentence] = []
        self.nodes: PriorityQueue[TableauSearchNode] = PriorityQueue()
        self.root = self._create_initial_node()
        self.nodes.put(self.root)

    def _create_initial_node(self) -> TableauSearchNode:
        return TableauSearchNode(
            priority=0, sentence_depth=0, tableau=self._create_axioms()
        )

    @staticmethod
    def _create_axioms() -> T.Tableau:
        return T.Tableau([])

    def search(self, narrator: N.Narrator) -> Generator[TableauSearchNode, None, bool]:
        """Run complete search to find all goal models"""
        for sentence in narrator:
            self.sentences.append(sentence)
            if self.nodes.empty():
                return False
            current_model = self.nodes.get()
            if current_model is not self.root:
                yield current_model
            self._extend_model(current_model)
        while not self.nodes.empty():
            current_model = self.nodes.get()
            if current_model is not self.root:
                yield current_model
            self._extend_model(current_model)
        return True

    def _extend_model(self, model: TableauSearchNode) -> None:
        if model.sentence_depth == len(self.sentences):
            return
        # Get next sentence
        sentence = self.sentences[model.sentence_depth]
        # Check for new entities
        noun_constant = S.Constant(S.Term.Sort.AGENT, sentence.noun)
        verb_constant = S.Constant(S.Term.Sort.TYPE, sentence.verb.inf)
        new_entities: List[S.Term] = []
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
            new_tableau = T.Tableau([formula], new_entities, model.tableau)
            for model_tableau in C.generate_models(new_tableau):
                self.nodes.put(
                    self.make_search_node(
                        model.sentence_depth + 1, model_tableau, model
                    )
                )

    @abstractmethod
    def make_search_node(
        self, sentence_depth: int, model_tableau: T.Tableau, parent: TableauSearchNode
    ) -> TableauSearchNode:
        """Create a sortable search node from a model tableau."""
        raise NotImplementedError()
