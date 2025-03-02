import re, json, string
from tqdm import tqdm
import numpy as np

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_presence(answers, context):
    """Verify if any of the answers is present in the given context."""

    answers = [normalize_answer(ans) for ans in answers]
    context = " ".join(context.split("#@"))
    context = normalize_answer(context)

    for ans in answers:
        if ans in context:
            return 1

    return 0

def compute_str_em(item, _key):
    """Compute STR-EM metric (only for ASQA item)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """

    if 'qa_pairs' not in item or item['qa_pairs'] is None:
        return 0, 0

    
    loc_acc = []
    for qa_pair in item['qa_pairs']:
        loc_acc.append(exact_presence(qa_pair['answers'], item[_key]))

    acc = np.mean(loc_acc)
    hit = int(np.mean(loc_acc) == 1)

    return acc, hit


def get_item_metrics(item, _key='response', is_asqa=False):
    
    if is_asqa:
        acc, _ = compute_str_em(item, _key)
    else:
        acc = exact_presence(item['answers'], item[_key]) 

    return acc

def directly_get_metrics(answers, response):
    acc = exact_presence(answers, response)
    return acc


if __name__ == "__main__":
    item = {
        "answers": ["december – 7th"],
        # "response": "March 19, 1848",
        "response": "December #@ December 7th #@ December – 7th",    # 'december – 7th'
    }
    print(get_item_metrics(item, is_asqa=False))