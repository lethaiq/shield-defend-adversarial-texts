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
from model import *
from dataset import *
from utils import *

load_path = './model.pt'
batch_size=256
max_len=64

class MyClassifier(oa.Classifier):
    def __init__(self, model, batch_size=1, max_len=64):
        self.model = model
        self.batch_size = batch_size
        self.max_len = max_len

    def get_pred(self, texts):
        probs = self.get_prob(texts)
        return probs.argmax(axis=1)

    def get_prob(self, texts):
        data_iter = prepare_single_bert(texts, [None]*len(texts), 
            batch_size=self.batch_size, max_len=self.max_len)
        _, preds = evaluate_without_attack(model, data_iter)
        return preds


def load_attacker(name):
    attacker = None
    elif name == 'TextBugger':
        attacker = oa.attackers.TextBuggerAttacker()
    elif name == 'DeepWordBug':
        attacker = oa.attackers.DeepWordBugAttacker()
    elif name == 'VIPER':
        attacker = oa.attackers.VIPERAttacker()
    return attacker


model = BertClassifierDARTS(model_type='bert-base-uncased', 
                            freeze_bert=False, 
                            output_dim=2, 
                            ensemble=0, 
                            device=device)
model.load_state_dict(torch.load(load_path))


victim = MyClassifier(model, batch_size=batch_size, max_len=max_len)
attacker = load_attacker('TextBugger')
attack_eval = oa.AttackEval(attacker, victim)
dataset = load_attack_dataset('KaggleToxicComment')
adversarials, result = attack_eval.eval(dataset, visualize=True)
