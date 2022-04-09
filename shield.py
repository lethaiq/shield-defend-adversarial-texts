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
base_save_path = './model.pt'
save_path = './shield.pt'
patience = 2
batch_size=128
max_len=64
dataset_name = 'clickbait'
model_type = 'bert-base-uncased'
alpha_darts = 0.5

train_iter, val_iter, test_iter, tokenizer = prepare_dataset_bert(model_type, 
                                                                dataset_name, 
                                                                batch_size=batch_size,
                                                                max_len=max_len,
                                                                device=device)
print("Train:", len(train_iter.dataset))
print("Val:", len(val_iter.dataset))
print("Test:", len(test_iter.dataset))

model = BertClassifierDARTS(model_type=model_type, 
                                    freeze_bert=True,
                                    output_dim=2, 
                                    ensemble=1, 
                                    N=5, 
                                    temperature=1.0,
                                    gumbel=1,
                                    scaler=1,
                                    darts=True,
                                    device=device)

model.load_state_dict(torch.load(base_save_path))
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

decision_parameters = filter(lambda p: 'darts_decision' in p[0], model.named_parameters())
opt_decision = optim.Adam([p[1] for p in decision_parameters], lr=0.25)


model.train()
loss_func = nn.CrossEntropyLoss()

best_val_loss = 9999
cur_patience = 0

for epoch in range(0, epochs):
    total_train = 0
    model.train() 

    for batch in tqdm(train_iter):
        _, loss, acc = evaluate_batch_single(model, batch, allow_grad=True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        scheduler.step()

        idx = np.random.choice(len(val_iter))
        batch_val = val_iter[idx]
        opt_decision.zero_grad()
        _, val_loss, acc = evaluate_batch_single(model, batch_val, allow_grad=True)
        reg_diversity_training, reg_diff = get_diversity_training_term(model, batch_val, logsumexp=False)
        reg_term = torch.tensor(alpha_darts).to(device)*(reg_diversity_training) - torch.tensor(alpha_darts).to(device)*reg_diff
        val_loss = val_loss + reg_term
        val_loss.backward()
        opt_decision.step()

    model.inference = True
    val_loss, preds = evaluate_without_attack(model, val_iter)
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

