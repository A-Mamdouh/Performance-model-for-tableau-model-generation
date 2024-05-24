"""Implementation of uninformed search agents"""

from src.search.search_agent_base import InferenceAgent
from src.search.search_node import TableauSearchNode


class DFSAgent(InferenceAgent):
    """Depth first search agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_id = 0

    def make_search_node(
        self, sentence_depth: int, model_tableau: T.Tableau, parent: TableauSearchNode
    ) -> TableauSearchNode:
        self.node_id -= 1
        return TableauSearchNode(
            priority=self.node_id,
            sentence_depth=sentence_depth,
            tableau=model_tableau,
            parent=parent,
        )


class BFSAgent(InferenceAgent):
    """Breadth first search agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_id = 0

    def make_search_node(
        self, sentence_depth: int, model_tableau: T.Tableau, parent: TableauSearchNode
    ) -> TableauSearchNode:
        self.node_id += 1
        return TableauSearchNode(
            priority=self.node_id,
            sentence_depth=sentence_depth,
            tableau=model_tableau,
            parent=parent,
        )
