import torch

def evaluate_batch_single(model, batch, allow_grad=False, preds_only=False, device='cpu'):
    loss_func = torch.nn.CrossEntropyLoss()
    with torch.set_grad_enabled(allow_grad):
        preds_prob = []
        seq = batch['input_ids'].to(device)
        attn_masks = batch['attention_mask'].to(device)
        label = batch['labels'].to(device)
        preds = model(seq, attn_masks)
        if len(preds_prob) == 0:
            preds_prob = torch.nn.functional.softmax(preds, dim=-1)
        loss = None
        acc = None
        if not preds_only:
            loss = loss_func(preds, label)
            acc = torch.sum(preds_prob.argmax(dim=-1) == label).item()
    return preds, loss, acc

def evaluate_batch(model, batch, allow_grad=False, ensemble_mean=True, train=False, preds_only=False, device='cpu'):
    if not train:
        model.eval()
    return evaluate_batch_single(model, batch, allow_grad, preds_only=preds_only, device=device)
