import torch
import numpy as np
from sklearn.metrics import f1_score


def evaluate_batch_single(model, batch, allow_grad=False, preds_only=False):
    loss_func = torch.nn.CrossEntropyLoss()
    label = None
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
            if label:
                loss = loss_func(preds, label)
                acc = torch.sum(preds_prob.argmax(dim=-1) == label).item()
    return preds, loss, acc

def get_preds(model, val_iter):
    model.eval()
    preds = []
    for batch in val_iter:
        pred, loss, acc = evaluate_batch_single(model, batch)
        preds.extend(pred.data.cpu().numpy())
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