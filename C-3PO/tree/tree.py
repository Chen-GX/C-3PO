

from tree.node import BaseNode, State


class Tree():
    def __init__(self, args, data_item, root=None, server=False):
        self.args = args
        self.server = server
        if not server:
            self.question_id = data_item['id']
            self.question = data_item['question']
            self.qa_pairs = data_item['qa_pairs']
            self.answers = data_item['answers']  # list
            self.is_asqa = True if self.args.dataname == "ASQA" else False
            assert isinstance(self.answers, list), "answers should be a list"
        else:
            self.question_id = None
            self.question = data_item
            self.qa_pairs = None
            self.answers = [""]
            self.is_asqa = False

        if root is None:
            self.root = BaseNode(node_id='0', parent=None, state=State(), P=1)
        else:
            self.root = root
        
        self.solution_nodes = []

    def add_solution_nodes(self, node_id: str):
        self.solution_nodes.append(node_id)

    def save_tree(self):

        candidates = [self.root]
        states = {}
        while candidates:
            node = candidates.pop(0)
            states[node.node_id] = node.return_state()
            if not node.is_leaf():
                candidates.extend(node.children)
        
        return_infos = {
            "question_id": self.question_id,
            "question": self.question,
            "qa_pairs": self.qa_pairs,
            "answers": self.answers,
            "tree": states,
            "solution_nodes": self.solution_nodes,
        }
        return return_infos

    