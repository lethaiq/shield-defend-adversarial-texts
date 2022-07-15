from model import *
from dataset import *
from utils import *

from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import random
import numpy
import torch
import tensorflow as tf

random.seed(12)
torch.manual_seed(12)
tf.random.set_seed(12)
np.random.seed(12)
torch.use_deterministic_algorithms(True)

device = 'cuda:0'
epochs = 5
grad_clip = 3
base_save_path = './model.pt'
save_path = './shield.pt'
patience = 2
batch_size=32
max_len=128
dataset_name = 'clickbait'
model_type = 'bert-base-uncased'
training_temp = 1.0
alpha_darts = 0.5

train_iter, val_iter, test_iter, tokenizer = prepare_dataset_bert(model_type, 
                                                                dataset_name, 
                                                                batch_size=batch_size,
                                                                max_len=max_len,
                                                                device=device)
print("Train:", len(train_iter.dataset))
print("Val:", len(val_iter.dataset))
print("Test:", len(test_iter.dataset))

def load_model():
	base = BertClassifierDARTS(model_type=model_type, 
	                            freeze_bert=False, 
	                            output_dim=2, 
	                            ensemble=0, 
	                            device=device)
	base.load_state_dict(torch.load(base_save_path))

	model = BertClassifierDARTS(model_type=model_type, 
	                                    freeze_bert=True,
	                                    output_dim=2, 
	                                    ensemble=1, 
	                                    N=5, 
	                                    temperature=training_temp,
	                                    gumbel=1,
	                                    scaler=1,
	                                    darts=True,
	                                    device=device)
	model_dict = model.state_dict()
	pretrained_dict = base.state_dict()
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	model_dict.update(pretrained_dict) 
	model.load_state_dict(model_dict)
	model.inference = False
	model = model.to(device)
	return model

model = load_model()

parameters = filter(lambda p: 'heads' in p[0], model.named_parameters())
opt = optim.Adam([p[1] for p in parameters], lr=3e-5)

decision_parameters = filter(lambda p: 'darts_decision' in p[0], model.named_parameters())
opt_decision = optim.Adam([p[1] for p in decision_parameters], lr=0.1)


model.train()
loss_func = nn.CrossEntropyLoss()

best_val_loss = 9999
cur_patience = 0

val_iter_batches = []
for batch in val_iter:
    val_iter_batches.append(batch)

for epoch in range(0, epochs):
    total_train = 0
    model.train() 

    for batch in tqdm(train_iter):
        _, loss, acc = evaluate_batch_single(model, batch, allow_grad=True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        idx = np.random.choice(len(val_iter_batches))
        batch_val = val_iter_batches[idx]
        opt_decision.zero_grad()
        _, val_loss, acc = evaluate_batch_single(model, batch_val, allow_grad=True)
        reg_diversity_training, reg_diff = get_diversity_training_term(model, batch_val, logsumexp=False)
        reg_term = torch.tensor(alpha_darts).to(device)*(reg_diversity_training) - torch.tensor(alpha_darts).to(device)*reg_diff
        val_loss = val_loss + reg_term
        val_loss.backward()
        opt_decision.step()

    print("SHIELD NAS Decision:", 
    	[F.softmax(model.darts_decision[i], dim=-1).detach().cpu().numpy() for i in range(model.N)])

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

