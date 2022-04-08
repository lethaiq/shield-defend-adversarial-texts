import OpenAttack as oa
import pandas as pd
from collections import deque
import requests
import re, string
import time
from time import sleep
from multiprocessing.dummy import Pool as ThreadPool 
from datasets import Dataset
import datasets # use the Hugging Face's datasets library
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import sys
import pickle


class MyClassifier(oa.Classifier):
    def __init__(self):
        pass

    def get_pred(self, input_):
        probs = self.get_prob(input_)
        return probs.argmax(axis=1)

    def get_prob(self, input_):
        ret = []
        for sent in input_:
            pred = score_comment(sent)
            ret.append([1-pred, pred])
        ret = np.array(ret)
        return ret


def load_attacker(name):
    attacker = None
    elif name == 'TextBugger':
        attacker = oa.attackers.TextBuggerAttacker()
    elif name == 'DeepWordBug':
        attacker = oa.attackers.DeepWordBugAttacker()
    elif name == 'VIPER':
        attacker = oa.attackers.VIPERAttacker()
    return attacker


victim = MyClassifier()
attacker = load_attacker('TextBugger')
attack_eval = oa.AttackEval(attacker, victim)
dataset = load_attack_dataset('KaggleToxicComment')
adversarials, result = attack_eval.eval(dataset, visualize=True)
