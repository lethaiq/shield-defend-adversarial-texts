from model import *
from dataset import *
from utils import *

from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

device = 'cuda:0'
epochs = 5
grad_clip = 3
save_path = './model.pt'
patience = 3
batch_size=128
max_len=64
dataset_name = 'spam'

train_iter, val_iter, test_iter, tokenizer = prepare_dataset_bert('bert-base-uncased', 
                                                                dataset_name, 
                                                                batch_size=batch_size,
                                                                max_len=max_len,
                                                                device=device)
print("Train:", len(train_iter.dataset))
print("Val:", len(val_iter.dataset))
print("Test:", len(test_iter.dataset))

model = BertClassifierDARTS(model_type='bert-base-uncased', 
                            freeze_bert=False, 
                            output_dim=2, 
                            ensemble=0, 
                            device=device)
model = model.to(device)

parameters = model.named_parameters()
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
opt = optim.AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=len(train_iter)*epochs)

model.train()
loss_func = nn.CrossEntropyLoss()

best_val_loss = 9999
cur_patience = 0

for epoch in range(0, epochs):
    running_loss = []
    total_train = 0
    model.train() 

    for batch in tqdm(train_iter):
        preds, loss, acc = evaluate_batch_single(model, batch)
        print(loss)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        scheduler.step()

        running_loss.append(loss.item())

    epoch_loss = np.mean(running_loss)

    model.inference = True
    val_loss, _ = evaluate_without_attack(model, val_iter)
    model.inference = False

    if best_val_loss > val_loss:
        cur_patience = 0
        best_val_loss = val_loss
        if save_path != "":
            print("Best val_loss changed. Saving...")
            torch.save(model.state_dict(), save_path)
    else:
        cur_patience += 1
        if cur_patience >= patience:
            break
