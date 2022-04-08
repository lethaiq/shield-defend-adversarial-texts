from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch

def load_nlp_dataset(dataset=None):
    data_files={'train': './dataset/{}_train.csv'.format(dataset),
                'validation': './dataset/{}_val.csv'.format(dataset),
                'test': './dataset/{}_test.csv'.format(dataset)}
    train_dataset = load_dataset('csv', data_files=data_files, split='train', ignore_verifications=False, name=dataset)
    eval_dataset = load_dataset('csv', data_files=data_files, split='validation', ignore_verifications=False, name=dataset)
    test_dataset = load_dataset('csv', data_files=data_files, split='test', ignore_verifications=False, name=dataset)
    return train_dataset, eval_dataset, test_dataset

class BertVocab:
    def __init__(self, vocab, vectors, model_type="BERT"):
        if model_type == "BERT":
            self.stoi = vocab.vocab
            self.itos = vocab.ids_to_tokens
        else:
            self.stoi = vocab.get_vocab()
            self.itos = {v: k for k, v in self.stoi.items()}
        self.vectors = vectors
        self.special_tokens = [vocab.pad_token, vocab.sep_token, vocab.unk_token, vocab.cls_token, vocab.mask_token]
        self.stopwords_idx = [self.stoi[a] for a in stopwords.words('english') if a in self.stoi]
        self.padding_idx = vocab.pad_token_id
        self.tokenizer = vocab
        self.name = model_type
        self.unk_token = vocab.unk_token
        self.pad_token = vocab.pad_token
        self.special_tokens_ids = {}
        self.all_special_ids = []
        for token in self.stoi:
            if '[' in token or '##' in token:
                _id = self.stoi[token]
                self.all_special_ids.append(_id)
        for token in self.special_tokens:
            self.all_special_ids.append(self.stoi[token])
            self.special_tokens_ids[str(self.stoi[token])] = 1


def prepare_dataset_bert(model, dataset, batch_size=32, max_len=64):
    train_dataset, eval_dataset, test_dataset = load_nlp_dataset(dataset)
    tokenizer = AutoTokenizer.from_pretrained(
            model,
            cache_dir="~/Downloads",
        )
    
    def encode(examples):
        return tokenizer(examples['text'], truncation=True, max_length=max_len)

    train_dataset = train_dataset.map(encode, batched=True)
    train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

    eval_dataset = eval_dataset.map(encode, batched=True)
    eval_dataset = eval_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

    test_dataset = test_dataset.map(encode, batched=True)
    test_dataset = test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

    def pad_seq(seq, max_batch_len: int, pad_value: int):
        return seq + (max_batch_len - len(seq)) * [pad_value]

    def collate_batch(batch) :
        batch_inputs = list()
        batch_attention_masks = list()
        labels = list()

        max_size = max([len(ex['input_ids']) for ex in batch])
        for item in batch:

            batch_inputs += [pad_seq(item['input_ids'], max_size, tokenizer.pad_token_id)]
            batch_attention_masks += [pad_seq(item['attention_mask'], max_size, 0)]
            labels.append(item['label'])

        return {"input_ids": torch.tensor(batch_inputs, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}

    train_iter = DataLoader(
                    train_dataset,
                    shuffle=True,
                    batch_size=batch_size,
                    collate_fn=collate_batch,
                    drop_last=True,
                )

    val_iter = DataLoader(
                eval_dataset,
                shuffle=True,
                batch_size=batch_size,
                collate_fn=collate_batch,
                drop_last=True,
            )

    test_iter = DataLoader(
                test_dataset,
                shuffle=True,
                batch_size=batch_size,
                collate_fn=collate_batch,
                drop_last=True,
            )

    return train_iter, val_iter, test_iter, tokenizer

