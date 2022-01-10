import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForTokenClassification, BertTokenizerFast, EvalPrediction
from torch.utils.data import Dataset
import os
import pandas as pd
import requests
from tqdm.auto import tqdm
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
import re

model_name = 'Rostlab/prot_bert_bfd'

def load_dataset(path, max_length):
        df = pd.read_csv(path,names=['sequence','structure'],skiprows=1)
        
        df['input_fixed'] = ["".join(seq.split()) for seq in df['sequence']]
        #df['input_fixed'] = [re.sub(r"[UZOB]", "X", seq) for seq in df['input_fixed']]
        seqs = [ list(seq)[:max_length-2] for seq in df['input_fixed']]

        df['label_fixed'] = ["".join(label.split()) for label in df['structure']]
        labels = [ list(label)[:max_length-2] for label in df['label_fixed']]
       
        assert len(seqs) == len(labels)  #样本数
        return seqs, labels

max_length = 512
data_path = './seq_struc_data'  #csv文件：序列码,结构码
train_seqs, train_labels = load_dataset(os.path.join(data_path,'train.csv'), max_length)
val_seqs, val_labels = load_dataset(os.path.join(data_path,'val.csv'), max_length)
test_seqs, test_labels = load_dataset(os.path.join(data_path,'test.csv'), max_length)

seq_tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=False)

train_seqs_encodings = seq_tokenizer(train_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
val_seqs_encodings = seq_tokenizer(val_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
test_seqs_encodings = seq_tokenizer(test_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)


# Consider each label as a tag for each token
unique_tags = set(tag for doc in train_labels for tag in doc)
unique_tags  = sorted(list(unique_tags))  # make the order of the labels unchanged
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

train_labels_encodings = encode_tags(train_labels, train_seqs_encodings)
val_labels_encodings = encode_tags(val_labels, val_seqs_encodings)
test_labels_encodings = encode_tags(test_labels, test_seqs_encodings)


class SS3Dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# we don't want to pass this to the model
_ = train_seqs_encodings.pop("offset_mapping")
_ = val_seqs_encodings.pop("offset_mapping")
_ = test_seqs_encodings.pop("offset_mapping")

train_dataset = SS3Dataset(train_seqs_encodings, train_labels_encodings)
val_dataset = SS3Dataset(val_seqs_encodings, val_labels_encodings)
test_dataset = SS3Dataset(test_seqs_encodings, test_labels_encodings)

def align_predictions(predictions: np.ndarray, label_ids: np.ndarray):
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(id2tag[label_ids[i][j]])
                    preds_list[i].append(id2tag[preds[i][j]])

        return preds_list, out_label_list

def compute_metrics(p: EvalPrediction):
    preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
    return {
        "accuracy": accuracy_score(out_label_list, preds_list),
        #"precision": precision_score(out_label_list, preds_list),
        #"recall": recall_score(out_label_list, preds_list),
        #"f1": f1_score(out_label_list, preds_list),
    }


def model_init():
    #config = AutoConfig.from_pretrained("./config.json")
    return AutoModelForTokenClassification.from_pretrained(model_name,
                                                        num_labels=len(unique_tags),
                                                         id2label=id2tag,
                                                         label2id=tag2id,
                                                         gradient_checkpointing=False)

training_args = TrainingArguments(
    output_dir='./test/results',          # output directory
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=1,   # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=200,                # number of warmup steps for learning rate scheduler
    learning_rate=3e-05,             # learning rate
    weight_decay=0.0,                # strength of weight decay
    logging_dir='./test/logs',            # directory for storing logs
    logging_steps=200,               # How often to print logs
    do_train=True,                   # Perform training
    do_eval=True,                    # Perform evaluation
    evaluation_strategy="epoch",     # evalute after each epoch
    gradient_accumulation_steps=32,  # total number of steps before back propagation
    fp16=True,                       # Use mixed precision
    fp16_opt_level="02",             # mixed precision mode
    run_name="ProBert-BFD-SS3",      # experiment name
    seed=3,                         # Seed for experiment reproducibility
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,

)

trainer = Trainer(
    model_init=model_init,                # the instantiated 🤗 Transformers model to be trained
    args=training_args,                   # training arguments, defined above
    train_dataset=train_dataset,          # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics = compute_metrics,    # evaluation metrics

)

trainer.train()

predictions, label_ids, metrics = trainer.predict(test_dataset)
print(metrics)


idx = 0
sample_ground_truth = " ".join([id2tag[int(tag)] for tag in test_dataset[idx]['labels'][test_dataset[idx]['labels'] != torch.nn.CrossEntropyLoss().ignore_index]])
sample_predictions =  " ".join([id2tag[int(tag)] for tag in np.argmax(predictions[idx], axis=1)[np.argmax(predictions[idx], axis=1) != torch.nn.CrossEntropyLoss().ignore_index]])

sample_sequence = seq_tokenizer.decode(list(test_dataset[idx]['input_ids']), skip_special_tokens=True)

print("Sequence       : {} \nGround Truth is: {}\nprediction is  : {}".format(sample_sequence,
                                                                      sample_ground_truth,
                                                                    # Remove the first token on prediction becuase its CLS token
                                                                      # and only show up to the input length
                                                                      sample_predictions[2:len(sample_sequence)+2]))

trainer.save_model('./prot_bert_bfd_seqstruc/')
seq_tokenizer.save_pretrained('./prot_bert_bfd_seqstruc/')                                                    

