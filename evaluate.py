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
batch_size=1
max_len=64
model = 'bert-base-uncased'
dataset_name = 'spam'
device = 'cuda:0'

class MyClassifier(oa.Classifier):
    def __init__(self, model, tokenizer, batch_size=1, max_len=64, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.device = device

    def get_pred(self, texts):
        probs = self.get_prob(texts)
        return probs.argmax(axis=1)

    def get_prob(self, texts):
        data_iter = prepare_single_bert(texts, [None]*len(texts), 
                                        tokenizer=self.tokenizer, 
                                        batch_size=self.batch_size, 
                                        max_len=self.max_len,
                                        device=self.device)
        _, preds = evaluate_without_attack(model, data_iter)
        return preds

def load_attacker(name):
    attacker = None
    if name == 'TextBugger':
        attacker = oa.attackers.TextBuggerAttacker()
    elif name == 'DeepWordBug':
        attacker = oa.attackers.DeepWordBugAttacker()
    elif name == 'VIPER':
        attacker = oa.attackers.VIPERAttacker()
    return attacker


model = BertClassifierDARTS(model_type=model, 
                            freeze_bert=False, 
                            output_dim=2, 
                            ensemble=0, 
                            device=device)
model.load_state_dict(torch.load(load_path))
tokenizer = AutoTokenizer.from_pretrained(model)

victim = MyClassifier(model, tokenizer, batch_size=batch_size, max_len=max_len)
attacker = load_attacker('TextBugger')
attack_eval = oa.AttackEval(attacker, victim)
_, _, test_dataset = load_nlp_dataset(dataset_name)
adversarials, result = attack_eval.eval(dataset, visualize=True)
