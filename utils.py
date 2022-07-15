import torch
import numpy as np
from sklearn.metrics import f1_score
import OpenAttack as oa
from sklearn.metrics import f1_score, accuracy_score
from datasets import Dataset
from dataset import *

from contextlib import contextmanager

@contextmanager
def no_ssl_verify():
    import ssl
    from urllib import request

    try:
        request.urlopen.__kwdefaults__.update({'context': ssl.SSLContext()})
        yield
    finally:
        request.urlopen.__kwdefaults__.update({'context': None})


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
                                        batch_size=1, 
                                        max_len=self.max_len,
                                        device=self.device)
        preds = get_preds(self.model, data_iter)
        return preds

def load_attacker(name):
    attacker = None
    with no_ssl_verify():
        if name == 'TextBugger':
            attacker = oa.attackers.TextBuggerAttacker()
        elif name == 'DeepWordBug':
            attacker = oa.attackers.DeepWordBugAttacker()
        elif name == 'TextFooler':
            attacker = oa.attackers.TextFoolerAttacker()
        elif name == 'BertAttack':
            attacker = oa.attackers.BERTAttacker()
    return attacker

def dataset_mapping(x):
    return {
        "x": x["text"],
        "y": x["label"],
    }

def cal_true_success_rate(advs, dataset):
    success = []
    labels = []
    pred_orgs = []
    pred_gens = []

    for i, adv in enumerate(advs):
        labels.append(dataset[i]['label'])

        pred_org = np.argmax(adv[1])
        pred_gen = adv[3]
        gen = adv[2]

        pred_orgs.append(pred_org)
        pred_gens.append(np.argmax(pred_gen))

        if pred_org == dataset[i]['label']:
            if gen != None:
                success.append(1)
            else:
                success.append(0)

    print(np.unique(labels, return_counts=True))
    print("Origin accuracy", accuracy_score(labels, pred_orgs))
    print("Adversarial accuracy", accuracy_score(labels, pred_gens))
    print("Attack Success Rate", np.mean(success))

def get_diversity_training_term(model, batch, optimize=True, logsumexp=False):
    l2_distance = torch.nn.MSELoss()
    loss_func = torch.nn.CrossEntropyLoss()
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)

    label = batch['labels']
    grads = []
    for i in  range(model.N):
        loss = loss_func(model.pred_heads[i]*model.random_key[:,i].unsqueeze(1), label.cuda()) # probs on individual
        grad = torch.autograd.grad(loss, model.emb1, create_graph=True)[0]
        grads.append(grad)

    total_cost = []
    total_cost_l2 = []
    for i in range(len(grads)):
        for j in range(len(grads)):
            if j > i:
                cost = cos(grads[i].contiguous().view(-1), grads[j].contiguous().view(-1))
                cost_l2 = l2_distance(grads[i].contiguous().view(-1), grads[j].contiguous().view(-1))
                total_cost.append(cost.unsqueeze(0))
                total_cost_l2.append(cost_l2.unsqueeze(0))

    total_cost = torch.cat(total_cost)
    total_cost_l2 = torch.cat(total_cost_l2)

    if logsumexp:
        out = torch.logsumexp(total_cost, 0)
    else:
        out = torch.mean(total_cost, 0)
    out_l2 = torch.mean(total_cost_l2, 0)
    
    return out, out_l2


def evaluate_batch_single(model, batch, allow_grad=False, preds_only=False):
    loss_func = torch.nn.CrossEntropyLoss()
    label = []
    with torch.set_grad_enabled(allow_grad):
        preds_prob = []
        seq = batch['input_ids']
        attn_masks = batch['attention_mask']
        if "labels" in batch:
            label = batch['labels']
        preds = model(seq, attn_masks)
        if len(preds_prob) == 0:
            preds_prob = torch.nn.functional.softmax(preds, dim=-1)
        loss = None
        acc = None
        if not preds_only:
            if len(label)>0:
                loss = loss_func(preds, label)
                acc = torch.sum(preds_prob.argmax(dim=-1) == label).item()
    return preds_prob, loss, acc

def get_preds(model, val_iter):
    model.eval()
    preds = []
    for batch in val_iter:
        preds_prob, loss, acc = evaluate_batch_single(model, batch)
        preds.extend(preds_prob.data.cpu().numpy())
    preds = np.array(preds)
    return preds   

def evaluate_without_attack(model, val_iter):
    model.eval()
    val_loss = []
    preds = []
    for batch in val_iter:
        pred, loss, acc = evaluate_batch_single(model, batch)
        val_loss.append(loss.item())
        preds.extend(pred.data.cpu().numpy())
    val_loss = np.mean(val_loss)
    return val_loss, preds