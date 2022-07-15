from datasets.utils.logging import set_verbosity_error
set_verbosity_error()

import OpenAttack as oa
from transformers import AutoTokenizer
from model import *
from dataset import *
from utils import *
import random
import numpy
import torch
import tensorflow as tf

random.seed(12)
torch.manual_seed(12)
tf.random.set_seed(12)
np.random.seed(12)


load_path = './shield.pt'
max_len=128
model_type = 'bert-base-uncased'
dataset_name = 'sst'
device = 'cuda:0'
attacker_name = 'TextFooler'
inference_temp = 0.1
rng = np.random.default_rng(12)

model = BertClassifierDARTS(model_type=model_type, 
                                    freeze_bert=True,
                                    is_training=False,
                                    inference=True,
                                    output_dim=2, 
                                    ensemble=1, 
                                    N=5, 
                                    temperature=inference_temp,
                                    gumbel=1,
                                    scaler=1,
                                    darts=True,
                                    device=device)
model.load_state_dict(torch.load(load_path))
model = model.to(device)
model.eval()

_, _, test_iter, _ = prepare_dataset_bert(model_type, 
                                        dataset_name, 
                                        batch_size=32,
                                        max_len=max_len,
                                        device=device)
preds = get_preds(model, test_iter)
preds = np.argmax(preds, axis=1)
labels = [a['label'] for a in test_iter.dataset]
f1 = f1_score(labels, preds)
acc = accuracy_score(labels, preds)
print("ACC:", acc)
print("F1:", f1)


tokenizer = AutoTokenizer.from_pretrained(model_type)
victim = MyClassifier(model, tokenizer, batch_size=1, max_len=max_len, device=device)
attacker = load_attacker(attacker_name)
attack_eval = oa.AttackEval(attacker, victim)
_, _, test_dataset = load_nlp_dataset(dataset_name)
test_dataset = test_dataset.select(rng.choice(len(test_dataset), 100))
test_dataset = test_dataset.map(dataset_mapping)
adversarials, result = attack_eval.eval(test_dataset, visualize=True)

cal_true_success_rate(adversarials, test_dataset)
