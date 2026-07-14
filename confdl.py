import numpy as np
import json
from collections import OrderedDict, defaultdict


def reformat_json(
    path: str, 
    reverse=set(),
    rename=dict(),
    keep=None,
):
    """    
    parses the records in a json file (formatted by Evgenia)
    returns a list of records for `judges` (human, NLG eval metric) and another for `systems` (uncertainty quantifiers)

    reverse: name of the quantifiers that need to multiplied by -1 (that is, to turn them into expressions of certainty/confidence)
    rename: key (old name) -> value (new name)
    keep: use this to keep only some judges/systems
    """    

    with open(path, 'r') as f:
        data = json.load(f)

    sys_rows = defaultdict(list)
    judge_rows = defaultdict(list)
    for record in data:


        # Evgenia stores uncertainty quantifiers under 
        for rater, judgment in record['uncertainty']['uncertainty_scores'].items():             
            if keep is not None and rater not in keep:
                continue
            if isinstance(judgment, str) and judgment.startswith("Confidence:"):
                judgment = judgment.strip("Confidence:")
            score = float(judgment)
            if rater in reverse:
                score = -score
                rater_name = f"neg_{rename.get(rater, rater)}"
            else:
                rater_name = rename.get(rater, rater)
            sys_rows[record['id']].append({'id': record['id'], 'rater': rater_name, 'label': None, 'score': score})
        
        # Evgenia stores human/oracle judgments as key-value pairs in the record's root
        for rater in record.keys():
            if keep is not None and rater not in keep:
                continue
            if rater.startswith('human_judge'):
                label = record[rater].lower() == 'correct'
                judge_rows[record['id']].append({'id': record['id'], 'rater': rename.get(rater, rater), 'label': label, 'score': float(label)})

        # Evgenia stores automated judgments as key-value pairs under 'greedy_correctness'
        for rater, judgment in record['greedy_correctness'].items():                            
            # Some judgments depend on a threshold, Evgenia stores those (and the respective labels) in a dictionary
            if isinstance(judgment, dict):
                for key, value in judgment.items():
                    rater_name = f"{rater}/{key}"
                    if keep is not None and rater_name not in keep:
                        continue
                    rater_name = f"{rename.get(rater, rater)}/{key}"
                    label = bool(value)
                    judge_rows[record['id']].append({'id': record['id'], 'rater': rater_name, 'label': label, 'score': float(label)})
            else:
                if keep is not None and rater not in keep:
                    continue
                label = bool(judgment)
                judge_rows[record['id']].append({'id': record['id'], 'rater': rename.get(rater, rater), 'label': label, 'score': float(label)})
        
    return judge_rows, sys_rows


class Vocabulary:
    def __init__(self):
        self.obj2int = dict()
        self.objects = []

    def __len__(self):
        return len(self.objects)

    def __call__(self, obj):
        """map obj to id"""
        idx = self.obj2int.get(obj, None)
        if idx is None:
            idx = len(self.objects)
            self.objects.append(obj)
            self.obj2int[obj] = idx
        return idx
        
    def __getitem__(self, idx):
        """retrieve object from id"""
        if not isinstance(idx, int):
            raise KeyError("idx must be int")
        return self.objects[idx]

    def items(self):
        return self.obj2int.items()

    def __iter__(self):
        return iter(self.objects)




class Dataset:
    def __init__(self, system_rows, judge_rows, standardize=True):
        """
        Construct "tall" representation for
        - item id
        - system id
        - confidence score (by system)
        - judge id
        - label (by judge)

        These are np arrays with the same length.

        Use 'standardize' to transform the confidence scores (per system) using (x-loc)/(2*scale), see Gelman and Hill for explanation.
        Set 'oracle' to a string to identify a reliable judge.
        """
        self.item2id = Vocabulary()
        self.judge2id = Vocabulary()
        self.sys2id = Vocabulary()
        I, S, X, J, Y = [], [], [], [], []
        score_by_sys = dict()
        for i, sys_records in system_rows.items():
            judge_records = judge_rows[i]              
            for srec in sys_records:       
                if srec['rater'] not in score_by_sys:
                    score_by_sys[srec['rater']] = defaultdict(list)
                score_by_sys[srec['rater']][srec['id']] = srec['score']            
                for jrec in judge_records:
                    I.append(self.item2id(srec['id']))            
                    S.append(self.sys2id(srec['rater']))
                    X.append(srec['score'])
                    J.append(self.judge2id(jrec['rater']))
                    Y.append(jrec['label'])
        self.I = np.array(I) # I is the item id
        self.S = np.array(S) # S is the system id
        self.X = np.array(X) # X is the confidence score (by system)
        self.J = np.array(J) # J is the judge id
        self.Y = np.array(Y) # Y is the label      
        
        params = dict()
        for system, measurements in score_by_sys.items():
            values = np.array([m for m in measurements.values()])
            loc = np.mean(values)
            scale = np.std(values)
            params[self.sys2id(system)] = (loc, scale)
        self.transformation_parameters = params

        if standardize:
            for i in range(len(self.I)):
                loc, scale = params[self.S[i]]
                stderr = 2 * scale
                if stderr == 0.:
                    stderr = 1.
                self.X[i] = (self.X[i] - loc) / stderr



