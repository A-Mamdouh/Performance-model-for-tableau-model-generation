from src.logic.base.calculus import generate_models
from src.logic.base.syntax import Exists, Forall, Formula, Predicate, Sort, Variable
from src.logic.base.tableau import Tableau
from src.query_environment.environment import AxiomsBase, AxiomUtils

from copy import deepcopy
from dataclasses import dataclass, field
import functools
import operator as op
from typing import Any, Dict, List, Optional, Tuple


class Sorts:
    individual: Sort = Sort("individual")
    verb: Sort = Sort("verb")
    event: Sort = Sort("event")
    adjective: Sort = Sort("adjective")


class Predicates:
    subject = Predicate("subject", 2, [Sorts.event, Sorts.individual])
    verb = Predicate("verb", 2, [Sorts.event, Sorts.verb])
    object_ = Predicate("object", 2, [Sorts.event, Sorts.individual])
    adjective = Predicate("adjective", 2, [Sorts.adjective, Sorts.individual])


@dataclass
class Noun:
    name: str
    referent: Optional[str]
    is_reference: bool

    def __post_init__(self) -> None:
        self.name = self.name.lower()
        if self.referent:
            self.referent = self.referent.lower()

    def __str__(self) -> str:
        return f"{self.name} ({self.referent})"


class Axioms(AxiomsBase):
    @staticmethod
    @AxiomUtils.only_one_kind_per_event(Predicates.subject)
    def axiom_only_one_subject(tableau: Tableau) -> Optional[Tableau]:
        """Only only subject per event"""

    @staticmethod
    @AxiomUtils.only_one_kind_per_event(Predicates.object_)
    def axiom_only_one_object(tableau: Tableau) -> Optional[Tableau]:
        """Only only object per event"""

    @staticmethod
    @AxiomUtils.only_one_kind_per_event(Predicates.verb)
    def axiom_only_one_verb(tableau: Tableau) -> Optional[Tableau]:
        """Only only verb per event"""





@dataclass
class Sentence:
    """This class is used for sentence initial annotation to help with translation to logic."""
    sentence: str
    subject: Noun
    verb: str
    object_: Optional[Noun]
    adjectives: List[Tuple[str, str]]
    is_negated: bool
    is_always: bool

    def __post_init__(self) -> None:
        self.sentence = self.sentence.lower()
        self.verb = self.verb.lower()
        self.adjectives = [(first.lower(), second.lower()) for first, second in self.adjectives]

    @classmethod
    def from_dict(cls, sentence_dict: Dict[str, any]) -> "Sentence":
        """Create and return a new sentence from a dictionary"""
        dict_ = deepcopy(sentence_dict)
        dict_["subject"] = Noun(**dict_["subject"])
        # Create proper noun objects
        if dict_.get("object_"):
            dict_["object_"] = Noun(**dict_["object_"])
        # Make sure adjectives have the correct tuple type
        adjectives = []
        for adjective, noun in dict_["adjectives"]:
            adjectives.append((adjective, noun))
        dict_["adjectives"] = adjectives
        # Create and return the new sentence object
        return cls(**dict_)

    def get_tableaus(self, parent: Optional[Tableau] = None) -> List[Tableau]:
        """Create all possible logical formulas from the sentence based on different readings."""
        entities = []

        # Find verb
        v_const = Sorts.verb.make_constant(self.verb)
        entities.append(v_const)
        verb = lambda e: Predicates.verb(e, v_const)

        # Find subject
        if self.subject.is_reference:
            s = Variable(Sorts.individual, self.subject.name)
            subject = lambda e: Exists(lambda v: Predicates.subject(e, v), variable=s)
        else:
            s_const = Sorts.individual.make_constant(self.subject.name)
            entities.append(s_const)
            subject = lambda e: Predicates.subject(e, s_const)

        # Find object
        object_ = None
        if self.object_:
            if self.object_.is_reference:
                o = Variable(Sorts.individual)
                object_ = lambda e: Exists(
                    lambda v: Predicates.object_(e, v), variable=o
                )
            else:
                o_const = Sorts.individual.make_constant(self.object_.name)
                entities.append(o_const)
                object_ = lambda e: Predicates.object_(
                    e, o_const
                )

        # Find adjectives
        adj_forms: List[Formula] = []
        for adjective, noun in self.adjectives:
            adj_const = Sorts.adjective.make_constant(adjective)
            entities.append(adj_const)
            noun_const = Sorts.individual.make_constant(noun)
            entities.append(noun_const)
            adj_forms.append(Predicates.adjective(adj_const, noun_const))
        adj_form = None
        if adj_forms:
            adj_form = functools.reduce(op.and_, adj_forms)

        # Collect formulas
        formulas = []
        # Handle the case where the sentence is negated
        if self.is_negated:
            formulas.extend(self._get_negated_readings(subject, verb, object_, adj_form))
        else:
            # Handle the case where the sentence is not negated and has the modifier always
            if self.is_always:
                formulas.extend(self._get_always_readings(subject, verb, object_, adj_form))
            # Handle the case where the sentence is not negated and doesn't have the modifier always
            else:
                # In this case, there is only one simple reading of Ee. subject(e) & object(e) & adjectives
                formulas.extend(self._get_simple_readings(subject, verb, object_, adj_form))

        # Create a tableau for each formula / reading
        tableaus: List[Tableau] = [Tableau([formula], entities) for formula in formulas]
        # If a parent is provided, use the merge function to assert the uniqueness properties
        if parent:
            tableaus = [tableau.merge(parent=parent) for tableau in tableaus]
        # Return the created tableaus
        return tableaus

    def _get_negated_readings(self, subject, verb, object_, adjectives: Optional[Formula]) -> List[Formula]:
        formulas = []
        if self.is_always:
            raise NotImplementedError("Cannot process sentences with both always and negation.")
        # Technically, the powerset of the components of the sentence should be used,
        # but this is sufficient for the time being.
        # First case: negated subject formula
        formulas.append(Exists(lambda e: ~subject(e) & verb(e) & object_(e) & adjectives))
        # Second case: negated object formula
        if object_:
            formulas.append(Exists(lambda e: ~object_(e) & subject(e) & verb(e) & adjectives))
        # Third case: negated verg
            formulas.append(Exists(lambda e: ~verb(e) & subject(e) & object_(e) & adjectives))
        # Fourth case: negated event
        formulas.append(~Exists(lambda e: subject(e) & verb(e) & object_(e) & adjectives))
        return formulas

    @staticmethod
    def _get_always_readings(subject, verb, object_, adjectives: Optional[Formula]) -> List[Formula]:
        formulas: List[Formula] = []
        if not object_:
            raise ValueError("No object found in an always statement.")
        # First formula is subject -> (object & adjectives & verb)
        first_formula: Predicate = lambda e: object_(e) & verb(e)
        if adjectives:
            first_formula1 = lambda e: first_formula(e) & adjectives
        else:
            first_formula1 = first_formula
        formulas.append(
    Forall(lambda e: subject(e) >> first_formula1(e))
        )
        # Second formula is object -> (subject & adjectives & verb)
        second_formula: Predicate = lambda e: subject(e) & verb(e)
        if adjectives:
            second_formula1 = lambda e: second_formula(e) & adjectives
        else:
            second_formula1 = second_formula
        formulas.append(Forall(lambda e: object_(e) >> second_formula1(e)))
        # Third formula is verb -> (subject & adjectives & object)
        third_formula: Predicate = lambda e: subject(e) & object_(e)
        if adjectives:
            third_formula1 = lambda e: third_formula(e) & adjectives
        else:
            third_formula1 = third_formula

        formulas.append(Forall(lambda e: verb(e) >> third_formula1(e)))
        return formulas

    @staticmethod
    def _get_simple_readings(subject: Predicate, verb: Predicate, object_: Optional[Predicate], adjectives: Optional[Formula]) -> List[Formula]:
        formula: Predicate = lambda e: verb(e) & subject(e)
        if object_:
            formula1 = lambda e: formula(e) & object_(e)
        else:
            formula1 = formula
        
        if adjectives:
            formula2 = lambda e: formula1(e) & adjectives
        else:
            formula2 = formula1
        return [Exists(formula2, Sorts.event)]


@dataclass
class ModelTreeNode:
    reading_model: Tableau
    previous_sentence_reading: Optional["DialogTreeNode"] = None
    next_sentence_readings: List["DialogTreeNode"] = field(default_factory=list)

    def extend_tree(self, *sentences: Sentence) -> None:
        if not sentences:
            return
        next_sentence, *rest_sentences = sentences
        for reading in next_sentence.get_tableaus():
            node = DialogTreeNode(next_sentence, reading, parent_reading_model=self)
            node.extend_tree(*rest_sentences)
            self.next_sentence_readings.append(node)
    
    @classmethod
    def create_tree(cls, *sentences: Sentence) -> "ModelTreeNode":
        node = ModelTreeNode(Tableau())
        node.extend_tree(*sentences)
        return node


@dataclass
class DialogTreeNode:
    sentence: Sentence
    sentence_reading: Tableau
    parent_reading_model: ModelTreeNode
    models: List[ModelTreeNode] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.models:
            self._gen_models()

    def _gen_models(self) -> None:
        extended_tableau = Tableau.merge(self.sentence_reading, parent=self.parent_reading_model.reading_model)
        for model in generate_models(extended_tableau, axioms=Axioms.get_axioms()):
            self.models.append(ModelTreeNode(model, self))

    def extend_tree(self, *sentences: Sentence) -> None:
        for model in self.models:
            model.extend_tree(*sentences)


@dataclass
class Dialog:
    sentences: List[Sentence]
    model_root: ModelTreeNode = field(init=False)

    def __len__(self) -> int:
        return len(self.sentences)
    
    def __item__(self, depth: int) -> None:
        return self.get_models(depth)

    def __post_init__(self) -> None:
         self.model_root = ModelTreeNode.create_tree(*self.sentences)

    def get_models(self, sentence_depth: Optional[int] = None) -> List[ModelTreeNode]:
        """Get all the models of the dialog at given sentence depth"""
        if sentence_depth is None:
            sentence_depth = len(self) - 1
        if sentence_depth >= len(self):
            raise KeyError(f"Cannot get model at depth {sentence_depth} from a dialog of depth {len(self)}")
        current_models = [self.model_root]
        for _ in range(sentence_depth+1):
            next_models = []
            for model in current_models:
                for reading in model.next_sentence_readings:
                    next_models.extend(reading.models)
            current_models = next_models
        return current_models
    
    @classmethod
    def from_dict(cls, raw_dialog: List[Dict[str, Any]]) -> "Dialog":
        return cls(sentences=list(map(Sentence.from_dict, raw_dialog)))