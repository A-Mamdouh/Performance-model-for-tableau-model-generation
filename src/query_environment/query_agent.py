"""Simple knowledge base agent"""

from dataclasses import dataclass, field
from typing import List, Optional

from src.logic.base.calculus import generate_models
from src.logic.base.syntax import Exists, is_literal
from src.logic.base.tableau import Tableau
from src.query_environment.environment import (
    Axioms,
    Constants,
    Predicates,
    Sorts
)
from src.query_environment.embeddings import EventInformationManager
from src.query_environment.learned_heuristic import HeuristicModel


@dataclass
class AgentNode:
    """Search tree node"""

    tableau: Tableau
    depth: int
    parent: Optional["AgentNode"] = None
    event_info_manager: EventInformationManager = field(
        default_factory=EventInformationManager
    )

    @classmethod
    def make_initial(cls) -> "AgentNode":
        """Create an empty node"""
        return AgentNode(Tableau(), 0, None)


class Agent:
    """Logical agent"""

    def __init__(self, heuristic_model: Optional[HeuristicModel] = None):
        self.knowledge_base: List[Tableau] = []
        self._models: List[AgentNode] = [AgentNode.make_initial()]
        self.heuristic_model = heuristic_model or HeuristicModel(None)

    def add_information(self, tableau: Tableau) -> Optional[Tableau]:
        """Add world knowledge to the agent"""
        self.knowledge_base.append(tableau)
        # Run DFS to find the next model
        while self._models:
            # If the current node is a model for the current knowledge base, return its tableau
            if self._models[-1].depth == len(self.knowledge_base):
                return self._models[-1].tableau
            # Pop the last node on the stack as the current model
            current_model = self._models.pop()
            # Get the chronologically next piece of knowledge
            next_tableau = self.knowledge_base[current_model.depth]
            # Combine current model and the next piece of knowledge
            combined_tableau = Tableau.merge(next_tableau, parent=current_model.tableau)
            # Run calculus to general models
            for new_model_tableau in generate_models(
                combined_tableau, Axioms.get_axioms()
            ):
                # Apply salience decay to the new model
                for entity, salience in list(new_model_tableau.saliences.items()):
                    new_model_tableau.saliences[entity] = salience * new_model_tableau.salience_decay
                # Create new DFS node and push onto the stack
                new_model = Agent.create_next_agent_node(current_model, new_model_tableau, self.heuristic_model)
                self._models.append(new_model)
        return None

    def query(self, query_tableau: Tableau) -> Optional[Tableau]:
        """Query the agent for some information"""
        return self.add_information(query_tableau)

    @staticmethod
    def create_next_agent_node(
        parent_node: AgentNode,
        next_tableau: Tableau,
        heuristic_model: HeuristicModel
    ) -> AgentNode:
        """Create an agent node from the current node and information about the next node"""
        # Collect unique literals of the new node
        next_literals = filter(is_literal, next_tableau.branch_literals)
        next_literals = filter(
            lambda literal: literal not in parent_node.tableau.branch_literals,
            next_literals
        )
        next_literals = list(next_literals)
        # Calculate new context vector
        next_ctx = heuristic_model.next_context_vector(next_literals, parent_node.event_info_manager.context_vector)
        # Return new node
        return AgentNode(
            next_tableau,
            parent_node.depth+1,
            parent_node,
            EventInformationManager(context_vector=next_ctx),
        )


def test_agent():
    """Module entry point"""
    agent = Agent()
    alex_pet_fido = Exists(
        lambda e: Predicates.subject(e, Constants.alex)
        & Predicates.action(e, Constants.pet)
        & Predicates.object(e, Constants.fido),
        sort=Sorts.event,
    )
    fido_bit_alex = Exists(
        lambda e: Predicates.subject(e, Constants.fido)
        & Predicates.action(e, Constants.bite)
        & Predicates.object(e, Constants.alex),
        sort=Sorts.event,
    )
    fido_bit_him = Exists(
        partial_formula=lambda e: Exists(
            partial_formula=lambda o: Predicates.subject(e, Constants.fido)
            & Predicates.action(e, Constants.bite)
            & Predicates.object(e, o),
            sort=Sorts.agent,
        ),
        sort=Sorts.event,
    )
    bob_pet_fido = Exists(
        lambda e: Predicates.subject(e, Constants.bob)
        & Predicates.action(e, Constants.pet)
        & Predicates.object(e, Constants.fido),
        sort=Sorts.event,
    )
    he_bit_him = Exists(
        partial_formula=lambda e: Exists(
            partial_formula=lambda s: Exists(
                partial_formula=lambda o: Predicates.subject(e, s)
                & Predicates.action(e, Constants.bite)
                & Predicates.object(e, o),
                sort=Sorts.agent,
            ),
            sort=Sorts.agent,
        ),
        sort=Sorts.event,
    )

    alex_pet_fido_t = Tableau(
        formulas=[alex_pet_fido],
        entities=[Constants.alex, Constants.pet, Constants.fido],
        saliences={
            Constants.alex: Tableau.recall_salience,
            Constants.pet: Tableau.recall_salience,
            Constants.fido: Tableau.recall_salience,
        }
    )
    bob_pet_fido_t = Tableau(
        formulas=[bob_pet_fido],
        entities=[Constants.bob],
        saliences={
            Constants.bob: Tableau.recall_salience,
            Constants.pet: Tableau.recall_salience,
            Constants.fido: Tableau.recall_salience,
        }
    )
    fido_bit_alex_t = Tableau(
        formulas=[fido_bit_alex],
        entities=[Constants.bite],
        saliences={
            Constants.alex: Tableau.recall_salience,
            Constants.bite: Tableau.recall_salience,
            Constants.fido: Tableau.recall_salience,
        } 
    )

    fido_bit_him_t = Tableau(
        formulas=[fido_bit_him],
        saliences={
            Constants.bite: Tableau.recall_salience,
            Constants.fido: Tableau.recall_salience,
        } 
    )

    story = [
        alex_pet_fido_t,
        bob_pet_fido_t,
        fido_bit_alex_t,
        fido_bit_him_t,
    ]

    for t in story:
        out = agent.add_information(t)
        if not out:
            print("No valid model")
            break
        for formula in sorted(
            out.branch_literals, key=lambda l: (str(l.args[0]), str(l))
        ):
            print(" ", formula)
        print()
    print("Done")


if __name__ == "__main__":
    test_agent()
