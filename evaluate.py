from datasets.utils.logging import set_verbosity_error
set_verbosity_error()

import OpenAttack as oa
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import pickle
from model import *
from dataset import *
from utils import *
from sklearn.metrics import f1_score

load_path = './model.pt'
batch_size=1
max_len=64
model_type = 'bert-base-uncased'
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
        data_iter = prepare_single_bert(texts, 
                                        tokenizer=self.tokenizer, 
                                        batch_size=self.batch_size, 
                                        max_len=self.max_len,
                                        device=self.device)
        preds = get_preds(model, data_iter)
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

def dataset_mapping(x):
    return {
        "x": x["text"],
        "y": x["label"],
        'target': 0 if x["label"] == 1 else 1
    }

model = BertClassifierDARTS(model_type=model_type, 
                            freeze_bert=False, 
                            output_dim=2, 
                            ensemble=0, 
                            device=device)
model.load_state_dict(torch.load(load_path))
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_type)

victim = MyClassifier(model, tokenizer, batch_size=batch_size, max_len=max_len, device=device)
attacker = load_attacker('DeepWordBug')
attack_eval = oa.AttackEval(attacker, victim)
_, _, test_dataset = load_nlp_dataset(dataset_name)
test_dataset = test_dataset.map(dataset_mapping)


_, _, test_iter, _ = prepare_dataset_bert('bert-base-uncased', 
                                        dataset_name, 
                                        batch_size=32,
                                        max_len=max_len,
                                        device=device)

preds, loss, acc = evaluate_without_attack(model, test_iter)
labels = [a['label'] for a in dataset]
f1 = f1_score(labels, preds)


# adversarials, result = attack_eval.eval(test_dataset, visualize=True)
