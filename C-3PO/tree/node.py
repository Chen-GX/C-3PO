import copy

from typing import List, Dict
from metrics import get_item_metrics

class State():
    """
    This class represents the state of a node
    param text: new generation text in this node
    param is_terminal: whether stopping
    
    """
    def __init__(
        self,
        input_text="",
        output_text="",
        thought=None,
        action=None,
        action_input=None,
        observation=None,
        is_terminal=False,
        terminal_reason=None,
        retrieve_query=None,
        retrieved_all_documents=None,
        ):
        self.input_text = input_text
        self.output_text = output_text
        self.is_terminal = is_terminal
        self.terminal_reason = terminal_reason

        self.thought = thought
        self.action = action
        self.action_input = action_input
        self.observation = observation

        self.retrieve_query = retrieve_query
        self.retrieved_all_documents = retrieved_all_documents

        self.final_answer = None  # extract key word from LLM
        self.final_status = None

        self.key_answer = None
        self.key_status = None

    def to_dict(self):
        return self.__dict__


class BaseNode():
    """
    This class defines a node of the tree
    param parent: parent node
    param state: state of current node
    param P: prior probability
    param total_value: the expected probability of solving this problem
    """
    def __init__(self, node_id=None, parent=None, state=None, P=None, depth=0, role=None, plan=None, documents=[], query_history={}):
        self.node_id = node_id
        self.parent = parent
        self.state = state
        self.depth = depth
        self.P = P
        self.role = role
        
        self.children = []

        self.documents = copy.deepcopy(documents)  # document {"title": str, "text": str, ...}
        self.plan = plan
        self.query_history = copy.deepcopy(query_history)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def add_child(self, child):
        self.children.append(child)

    def extend_documents(self, documents: List[Dict]):
        for document in documents:
            if document not in self.documents:
                self.documents.append(copy.deepcopy(document))
    
    def remove_noinfo_documents(self, query_text):
        new_documents = [doc for doc in self.documents if query_text != doc["text"]]
        self.documents = new_documents

    def update_plan(self):
        self.plan = self.state.output_text

    def document_is_empty(self) -> bool:
        return len(self.documents) == 0

    def check_answer(self, qa_pairs: List[Dict], answers: List[str], is_asqa=False):
        self.state.final_answer = self.state.output_text
        item = {"qa_pairs": qa_pairs, "answers": answers, "response": self.state.final_answer}
        self.state.final_status = get_item_metrics(item, is_asqa=is_asqa)

    def return_solution_state(self, node_id: str):
        return {
            "node_id": node_id,
            "final_status": self.state.final_status,
            "key_status": self.state.key_status,
        }

    def return_state(self):
        # return self.state
        infos = {
            "role": self.role,
            "plan": self.plan,
            "documents": self.documents,
            "query_history": self.query_history,
            "state": self.state.to_dict(),
        }
        return infos
