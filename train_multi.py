import argparse
import json
import math
import os
import random
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from dataset import (
    collate_fn,
    TURLColTypeColwiseDataset,
    TURLColTypeTablewiseDataset,
    TURLRelExtColwiseDataset,
    TURLRelExtTablewiseDataset,
    SatoCVColwiseDataset,
    SatoCVTablewiseDataset,
    SportsTablesColwiseDataset,
    SportsTablesTablewiseDataset,
    GittablesCVColwiseDataset,
    GittablesCVTablewiseDataset
)

from model import BertForMultiOutputClassification, BertMultiPairPooler
from util import f1_score_multilabel
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
import logging


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shortcut_name",
        default="bert-base-uncased",
        type=str,
        help="Huggingface model shortcut name ",
    )
    parser.add_argument(
        "--max_length",
        default=128,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--epoch",
        default=30,
        type=int,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--random_seed",
        default=1,
        type=int,
        help="Random seed",
    )

    parser.add_argument(
        "--num_classes",
        default=78,
        type=int,
        help="Number of classes",
    )
    #TOCHECK
    parser.add_argument("--multi_gpu",
                        action="store_true",
                        default=False,
                        help="Use multiple GPU")
    #TODO
    parser.add_argument("--fp16",
                        action="store_true",
                        default=False,
                        help="Use FP16")
    parser.add_argument("--warmup",
                        type=float,
                        default=0.,
                        help="Warmup ratio")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--tasks",
                        type=str,
                        nargs="+",
                        default=["sato0"],
                        choices=[
                            "sato0", "sato1", "sato2", "sato3", "sato4",
                            "msato0", "msato1", "msato2", "msato3", "msato4",
                            "turl", "turl-re", "SportsTables", 'gittables_0', 'gittables_1', 'gittables_2',
                            'gittables_10', 'gittables_11', 'gittables_12', 'gittables_20',
                            'gittables_21', 'gittables_22', 'gittables_30'
                            # Jos Hinzufügen des Datensatz
                        ],
                        help="Task names}")
    parser.add_argument("--colpair",
                        #default=False,
                        action="store_true",
                        help="Use column pair embedding")
    parser.add_argument("--train_ratios",
                        type=str,
                        nargs="+",
                        default=[],
                        help="e.g., --train_ratios turl=0.8 turl-re=0.1")
    parser.add_argument("--from_scratch",
                        #default=False,
                        action="store_true",
                        help="Training from scratch")
    parser.add_argument("--single_col",
                        default=False,
                        action="store_true",
                        help="Training with single column model")
    parser.add_argument("--shuffle_cols",
                        default=False,
                        action="store_true",
                        help="Shuffle the column order in the table before serialization"
                        )
    parser.add_argument(
        "--sport_domains",
        type=str,
        nargs="+",
        default = ["baseball"],
        choices=["baseball", "basketball", "football", "hockey", "soccer"]
    )

    args = parser.parse_args()
    args.tasks = sorted(args.tasks)

    ## TODO Change of number of classes; added the 3 options for 80%, 85% and 90 %
    #Jos Anzahl der Klassen für den neuen Datensatz
    task_num_class_dict = {
        "gittables_0" : 13035,
        "gittables_1" : 4828,
        "gittables_2" : 7833,
        "gittables_10" : 6746,
        "gittables_11" : 4427,
        "gittables_12" : 2236,
        "gittables_20" : 7559,
        "gittables_21" : 4748,
        'gittables_22' : 2566,
        "gittables_30" : 1214,
        "sato0": 78,
        "sato1": 78,
        "sato2": 78,
        "sato3": 78,
        "sato4": 78,
        "msato0": 78,
        "msato1": 78,
        "msato2": 78,
        "msato3": 78,
        "msato4": 78,
        "turl": 255,
        "turl-re": 121,
        "SportsTables":462
    }

    train_ratio_dict = {}
    num_classes_list = []
    for task in args.tasks:
        num_classes_list.append(task_num_class_dict[task])
        # Default training ratio is ALL
        train_ratio_dict[task] = 1.0

    # Training ratio
    for train_ratio in args.train_ratios:
        task, ratio_str = train_ratio.split("=")
        ratio = float(ratio_str)
        assert task in train_ratio_dict, "Invalid task name: {}".format(task)
        assert 0 < ratio <= 1
        train_ratio_dict[task] = ratio

    # For tagname
    train_ratio_str_list = []
    for task in sorted(train_ratio_dict.keys()):
        ratio = train_ratio_dict[task]
        train_ratio_str_list.append("{}-{:.2f}".format(task, ratio))

    if args.colpair:
        assert "turl-re" in args.tasks, "colpair can be only used for Relation Extraction"

    print("args={}".format(json.dumps(vars(args))))

    max_length = args.max_length
    batch_size = args.batch_size
    num_train_epochs = args.epoch

    shortcut_name = args.shortcut_name

    if args.single_col:
        # Single column
        tag_name_col = "single"
    else:
        tag_name_col = "mosato"

    if args.colpair:
        taskname = "{}-colpair".format("".join(args.tasks))
    else:
        taskname = "".join(args.tasks)

    if args.from_scratch:
        tag_name = "model/{}_{}_bert_{}-bs{}-ml-{}".format(
            taskname, tag_name_col, "{}-fromscratch".format(shortcut_name),
            batch_size, max_length)
    else:
        tag_name = "model/{}_{}_bert_{}-bs{}-ml-{}".format(
            taskname, tag_name_col, shortcut_name, batch_size, max_length)
    # TODO: Check
    tag_name += "__{}".format("_".join(train_ratio_str_list))
    tag_name += f"_sc{args.shuffle_cols}"
    tag_name += f"_rs{args.random_seed}"
    if "SportsTables" in tag_name:
        tag_name += f"_{args.sport_domains}"
    print(tag_name)

    dirpath = os.path.dirname(tag_name)
    if not os.path.exists(dirpath):
        print("{} not exists. Created".format(dirpath))
        os.makedirs(dirpath)
        
    ## logging set-up
    logging.basicConfig(filename=f"{tag_name}.log", filemode="w", format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.debug("args={}".format(json.dumps(vars(args))))
    logging.debug(f"Device: {device}")
    logging.debug(f"Model tag name: {tag_name}")

    tokenizer = BertTokenizer.from_pretrained(shortcut_name)
    # model = BertForSequenceClassification.from_pretrained(

    print(num_classes_list)

    models = []
    for i, num_classes in enumerate(num_classes_list):
        if args.single_col:
            model_config = BertConfig.from_pretrained(shortcut_name,
                                                      num_labels=num_classes)
            model = BertForSequenceClassification(model_config)
        else:
            if args.from_scratch:
                # No pre-trained checkpoint
                model_config = BertConfig.from_pretrained(
                    shortcut_name, num_labels=num_classes)
                model = BertForMultiOutputClassification(model_config)
            else:
                # Pre-trained checkpoint
                model = BertForMultiOutputClassification.from_pretrained(
                    shortcut_name,
                    num_labels=num_classes,
                    output_attentions=False,
                    output_hidden_states=False,
                )
            if args.tasks[i] == "turl-re" and args.colpair:
                print("Use column-pair pooling")
                # Use column pair embeddings
                config = BertConfig.from_pretrained(shortcut_name)
                model.bert.pooler = BertMultiPairPooler(config).to(device)

        # For multi-task learning
        if i > 0:
            assert not args.single_col, "TODO: Single-column model for multi-task learning"
            # The multi-task model shares embeddings & encoder part, not sharing the pooling layer
            model.bert.embeddings = models[0].bert.embeddings
            model.bert.encoder = models[0].bert.encoder
            # [Option] The following also shares the pooling layer
            # model.bert = models[0].bert

        models.append(model.to(device))

    # Check if the parameters are shared
    assert 1 == len(
        set([
            model.bert.embeddings.word_embeddings.weight.data_ptr()
            for model in models
        ]))
    assert 1 == len(
        set([
            model.bert.encoder.layer[0].attention.output.dense.weight.data_ptr(
            ) for model in models
        ]))
    assert len(models) == len(
        set([model.bert.pooler.dense.weight.data_ptr() for model in models]))

    train_datasets = []
    train_dataloaders = []
    valid_datasets = []
    valid_dataloaders = []

    print(args.tasks)

    for task in args.tasks:
        train_ratio = train_ratio_dict[task]
        if task in [
                "sato0", "sato1", "sato2", "sato3", "sato4", "msato0",
                "msato1", "msato2", "msato3", "msato4"
        ]:
            cv = int(task[-1])

            if task[0] == "m":
                multicol_only = True
            else:
                multicol_only = False

            if args.single_col:
                dataset_cls = SatoCVColwiseDataset
            else:
                dataset_cls = SatoCVTablewiseDataset

            train_dataset = dataset_cls(cv=cv,
                                        split="train",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        multicol_only=multicol_only,
                                        train_ratio=train_ratio,
                                        device=device)
            valid_dataset = dataset_cls(cv=cv,
                                        split="train",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        multicol_only=multicol_only,
                                        train_ratio=train_ratio,
                                        device=device)

            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                          sampler=train_sampler,
                                          batch_size=batch_size,
                                          collate_fn=collate_fn)
            valid_dataloader = DataLoader(valid_dataset,
                                          batch_size=batch_size,
                                          collate_fn=collate_fn)
        elif "turl" in task:
            if task in ["turl"]:
                # TODO: Double-check if it is compatible with single/multi-column data
                filepath = "data/table_col_type_serialized.pkl"
                if args.single_col:
                    assert task == "turl"  # Single-column model cannot be used for turl-sch
                    # ColumnWise
                    dataset_cls = TURLColTypeColwiseDataset
                else:
                    # Tablewise
                    dataset_cls = TURLColTypeTablewiseDataset
            elif task in ["turl-re"]:
                # TODO: Double-check if it is compatible with single/multi-column data
                filepath = "data/table_rel_extraction_serialized.pkl"
                if args.single_col:
                    assert task == "turl-re"  # Single-column model cannot be used for turl-sch
                    dataset_cls = TURLRelExtColwiseDataset
                else:
                    dataset_cls = TURLRelExtTablewiseDataset
            else:
                raise ValueError("turl tasks must be turl or turl-re.")

            train_dataset = dataset_cls(filepath=filepath,
                                        split="train",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        multicol_only=False,
                                        train_ratio=train_ratio,
                                        device=device)
            valid_dataset = dataset_cls(filepath=filepath,
                                        split="dev",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        multicol_only=False,
                                        device=device)

            # Can be the same
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                          sampler=train_sampler,
                                          batch_size=batch_size,
                                          collate_fn=collate_fn)
            valid_dataloader = DataLoader(valid_dataset,
                                          batch_size=batch_size,
                                          collate_fn=collate_fn)
        elif "SportsTables" in task:
            if args.single_col:
                dataset_cls = SportsTablesColwiseDataset
            else:
                dataset_cls = SportsTablesTablewiseDataset
            
            train_dataset = dataset_cls(tokenizer=tokenizer,
                                        sport_domains=args.sport_domains,
                                        random_state=args.random_seed,
                                        split="train",
                                        max_length=max_length,
                                        train_ratio=train_ratio,
                                        shuffle_cols=args.shuffle_cols,
                                        device=device,
                                        base_dirpath="/ext/daten-wi/slangenecker/SportsTables")
            valid_dataset = dataset_cls(tokenizer=tokenizer,
                                        sport_domains=args.sport_domains,
                                        random_state=args.random_seed,
                                        split="valid",
                                        max_length=max_length,
                                        train_ratio=train_ratio,
                                        shuffle_cols=args.shuffle_cols,
                                        device=device,
                                        base_dirpath="/ext/daten-wi/slangenecker/SportsTables")
            
            
            train_dataloader = DataLoader(train_dataset,
                                          batch_size=batch_size, 
                                          collate_fn=collate_fn)
            valid_dataloader = DataLoader(valid_dataset, 
                                          batch_size=batch_size, 
                                          collate_fn=collate_fn)
            
        # TODO add the datapreparation for the gittables Data; done by example of sato dataset
        elif "gittables" in task:
            if args.single_col:
                dataset_cls = GittablesCVColwiseDataset
            else:
                dataset_cls = GittablesCVTablewiseDataset

            train_dataset = dataset_cls(cv=task.split('_')[-1],
                                        split="train",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        train_ratio=train_ratio,
                                        device=device)
            valid_dataset = dataset_cls(cv=task.split('_')[-1],
                                        split="valid",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        train_ratio=train_ratio,
                                        device=device)

            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                          sampler=train_sampler,
                                          batch_size=batch_size,
                                          collate_fn=collate_fn)
            valid_dataloader = DataLoader(valid_dataset,
                                          batch_size=batch_size,
                                          collate_fn=collate_fn)
        
        else:
            raise ValueError("task name must be either sato or turl.")

        # Store dataloaders
        train_datasets.append(train_dataset)
        train_dataloaders.append(train_dataloader)
        valid_datasets.append(valid_dataset)
        valid_dataloaders.append(valid_dataloader)

    print('Data loading completed')
    print(len(train_dataset))

    optimizers = []
    schedulers = []
    loss_fns = []
    for i, train_dataloader in enumerate(train_dataloaders):
        t_total = len(train_dataloader) * num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in models[i].named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0
            },
            {
                "params": [
                    p for n, p in models[i].named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0
            },
        ]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
        optimizer = AdamW(models[i].parameters(), lr=args.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=t_total)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

        if "sato" in args.tasks[i]:
            loss_fns.append(CrossEntropyLoss())
        elif "turl" in args.tasks[i]:
            loss_fns.append(BCEWithLogitsLoss())
        elif "SportsTables" in args.tasks[i]:
            loss_fns.append(CrossEntropyLoss())
        # TODO add loss function; done like in sato example
        elif "gittables" in args.tasks[i]:
            loss_fns.append(CrossEntropyLoss())
        else:
            raise ValueError("task name must be either sato or turl.")

    set_seed(args.random_seed)

    # Best validation score could be zero
    best_vl_micro_f1s = [-1 for _ in range(len(args.tasks))]
    best_vl_macro_f1s = [-1 for _ in range(len(args.tasks))]
    loss_info_lists = [[] for _ in range(len(args.tasks))]
    best_vl_loss = 1000
    
    # early stopping criteria
    early_stopp_counter = 0
    early_stopp_patience = 10
    

    print('start of training')
    ## Training
    try:
        for epoch in range(num_train_epochs):
            for k, (task, model, train_dataset, valid_dataset, train_dataloader,
                    valid_dataloader, optimizer, scheduler, loss_fn,
                    loss_info_list) in enumerate(
                        zip(args.tasks, models, train_datasets, valid_datasets,
                            train_dataloaders, valid_dataloaders, optimizers,
                            schedulers, loss_fns, loss_info_lists)):
                t1 = time()
                print(epoch, " Epoch ", k, " K")
                model.train()
                tr_loss = 0.
                tr_pred_list = []
                tr_true_list = []

                vl_loss = 0.
                vl_pred_list = []
                vl_true_list = []

                print(len(train_dataloader))

                for batch_idx, batch in enumerate(train_dataloader):
                    if args.single_col:
                        logits = model(batch["data"].T).logits
                        if "sato" in task:
                            tr_pred_list += logits.argmax(
                                1).cpu().detach().numpy().tolist()
                            tr_true_list += batch["label"].cpu().detach().numpy(
                            ).tolist()
                            loss = loss_fn(logits, batch["label"])
                        elif "turl" in task:
                            # TURL & TURL-REL for the single-col case
                            tr_pred_list += (logits >= math.log(0.5)
                                            ).int().detach().cpu().tolist()
                            tr_true_list += batch["label"].cpu().detach().numpy(
                            ).tolist()
                            loss = loss_fn(logits, batch["label"].float())
                        elif "SportsTables" in task:
                            tr_pred_list += logits.argmax(
                                1).cpu().detach().numpy().tolist()
                            tr_true_list += batch["label"].cpu().detach().numpy(
                            ).tolist()
                            loss = loss_fn(logits, batch["label"])
                        # TODO add the loss calculation single_col; done like in sato example
                        elif "gittables" in task:
                            tr_pred_list += logits.argmax(
                                1).cpu().detach().numpy().tolist()
                            tr_true_list += batch["label"].cpu().detach().numpy(
                            ).tolist()
                            loss = loss_fn(logits, batch["label"])
                        
                        else:
                            raise ValueError(
                                "Invalid task for single-col: {}".format(task))
                    else:
                        #print(batch_idx)
                        #print(task)

                        #print
                        # Multi-column model
                        logits, = model(batch["data"])  # (row, col) is opposite?

                        #print('Logits from model')
                        #print(logits)

                        # Align the tensor shape when the size is 1
                        if len(logits.shape) == 2:
                            logits = logits.unsqueeze(0)
                        # DEBUG===
                        # print("batch['data'].shape={} data['label'].shape={} batch['idx'].shape={}".format(
                        #    batch["data"].shape, batch["label"].shape, batch["idx"].shape))
                        # ===
                        cls_indexes = torch.nonzero(
                            batch["data"] == tokenizer.cls_token_id)
                        #print('cls_indexes')
                        filtered_logits = torch.zeros(cls_indexes.shape[0],
                                                    logits.shape[2]).to(device)
                        #print('filtered logits')
                        for n in range(cls_indexes.shape[0]):
                            i, j = cls_indexes[n]
                            logit_n = logits[i, j, :]
                            filtered_logits[n] = logit_n
                        #('for logits')
                        if "sato" in task:
                            tr_pred_list += filtered_logits.argmax(
                                1).cpu().detach().numpy().tolist()
                            tr_true_list += batch["label"].cpu().detach().numpy(
                            ).tolist()
                        elif "turl" in task:
                            if task == "turl-re":
                                all_preds = (filtered_logits >= math.log(0.5)
                                            ).int().detach().cpu().numpy()
                                all_labels = batch["label"].cpu().detach().numpy()
                                # Ignore the very first CLS token
                                idxes = np.where(all_labels > 0)[0]
                                tr_pred_list += all_preds[idxes, :].tolist()
                                tr_true_list += all_labels[idxes, :].tolist()
                            elif task == "turl":
                                # Threshold value = 0.5
                                tr_pred_list += (filtered_logits >= math.log(0.5)
                                                ).int().detach().cpu().tolist()
                                tr_true_list += batch["label"].cpu().detach(
                                ).numpy().tolist()
                        elif "SportsTables" in task:
                            tr_pred_list += filtered_logits.argmax(
                                1).cpu().detach().numpy().tolist()
                            tr_true_list += batch["label"].cpu().detach().numpy(
                            ).tolist()
                        # TODO add the loss function not single_col; done like in sato example
                        elif "gittables" in task:
                            tr_pred_list += filtered_logits.argmax(
                                1).cpu().detach().numpy().tolist()
                            tr_true_list += batch["label"].cpu().detach().numpy(
                            ).tolist()

                        if "sato" in task:
                            loss = loss_fn(filtered_logits, batch["label"])
                        elif "turl" in task:
                            loss = loss_fn(filtered_logits, batch["label"].float())
                        elif "SportsTables" in task:
                            loss = loss_fn(filtered_logits, batch["label"])
                        # TODO calculate loss for not single_col; done like in sato example
                        elif "gittables" in task:
                            loss = loss_fn(filtered_logits, batch["label"])

                    #print('After ifs')

                    loss.backward()
                    tr_loss += loss.item()
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                print('After Batches')
                tr_loss /= (len(train_dataset) / batch_size)

                if "sato" in task:
                    tr_micro_f1 = f1_score(tr_true_list,
                                        tr_pred_list,
                                        average="micro")
                    tr_macro_f1 = f1_score(tr_true_list,
                                        tr_pred_list,
                                        average="macro")
                    tr_class_f1 = f1_score(tr_true_list,
                                        tr_pred_list,
                                        average=None,
                                        labels=np.arange(args.num_classes))
                elif "turl" in task:
                    tr_micro_f1, tr_macro_f1, tr_class_f1, _ = f1_score_multilabel(
                        tr_true_list, tr_pred_list)
                elif "SportsTables" in task:
                    tr_micro_f1 = f1_score(tr_true_list,
                                        tr_pred_list,
                                        average="micro")
                    tr_macro_f1 = f1_score(tr_true_list,
                                        tr_pred_list,
                                        average="macro")
                    tr_class_f1 = f1_score(tr_true_list,
                                        tr_pred_list,
                                        average=None,
                                        labels=np.arange(args.num_classes))
                # TODO add f1 score calculation; done like in sato example
                elif "gittables" in task:
                    tr_micro_f1 = f1_score(tr_true_list,
                                        tr_pred_list,
                                        average="micro", 
                                        zero_division=1)
                    tr_macro_f1 = f1_score(tr_true_list,
                                        tr_pred_list,
                                        average="macro",
                                        zero_division=1)
                    tr_class_f1 = f1_score(tr_true_list,
                                        tr_pred_list,
                                        average=None,
                                        labels=np.arange(args.num_classes),
                                        zero_division=1)

                print('Before Validation')
                # Validation
                model.eval()
                for batch_idx, batch in enumerate(valid_dataloader):
                    if args.single_col:
                        # Single-column
                        logits = model(batch["data"].T).logits
                        if "sato" in task:
                            vl_pred_list += logits.argmax(
                                1).cpu().detach().numpy().tolist()
                            vl_true_list += batch["label"].cpu().detach().numpy(
                            ).tolist()
                            loss = loss_fn(logits, batch["label"])
                        elif "turl" in task:
                            tr_pred_list += (logits >= math.log(0.5)
                                            ).int().detach().cpu().tolist()
                            tr_true_list += batch["label"].cpu().detach().numpy(
                            ).tolist()
                            loss = loss_fn(logits, batch["label"].float())
                        elif "SportsTables" in task:
                            vl_pred_list += logits.argmax(
                                1).cpu().detach().numpy().tolist()
                            vl_true_list += batch["label"].cpu().detach().numpy(
                            ).tolist()
                            loss = loss_fn(logits, batch["label"])
                        # TODO predict and calculate loss for validation single_col; done like in sato example
                        elif "gittables" in task:
                            vl_pred_list += logits.argmax(
                                1).cpu().detach().numpy().tolist()
                            vl_true_list += batch["label"].cpu().detach().numpy(
                            ).tolist()
                            loss = loss_fn(logits, batch["label"])
                        else:
                            raise ValueError(
                                "Invalid task for single-col: {}".format(task))
                    else:
                        # Multi-column
                        logits, = model(batch["data"])
                        if len(logits.shape) == 2:
                            logits = logits.unsqueeze(0)
                        cls_indexes = torch.nonzero(
                            batch["data"] == tokenizer.cls_token_id)
                        filtered_logits = torch.zeros(cls_indexes.shape[0],
                                                    logits.shape[2]).to(device)
                        for n in range(cls_indexes.shape[0]):
                            i, j = cls_indexes[n]
                            logit_n = logits[i, j, :]
                            filtered_logits[n] = logit_n
                        if "sato" in task:
                            vl_pred_list += filtered_logits.argmax(
                                1).cpu().detach().numpy().tolist()
                            vl_true_list += batch["label"].cpu().detach().numpy(
                            ).tolist()
                        elif "turl" in task:
                            if task == "turl-re":
                                all_preds = (filtered_logits >= math.log(0.5)
                                            ).int().detach().cpu().numpy()
                                all_labels = batch["label"].cpu().detach().numpy()
                                idxes = np.where(all_labels > 0)[0]
                                vl_pred_list += all_preds[idxes, :].tolist()
                                vl_true_list += all_labels[idxes, :].tolist()
                            elif task == "turl":
                                # Threshold value = 0.5
                                vl_pred_list += (filtered_logits >= math.log(0.5)
                                                ).int().detach().cpu().tolist()
                                vl_true_list += batch["label"].cpu().detach(
                                ).numpy().tolist()
                        elif "SportsTables" in task:
                            vl_pred_list += filtered_logits.argmax(
                                1).cpu().detach().numpy().tolist()
                            vl_true_list += batch["label"].cpu().detach().numpy(
                            ).tolist()
                        
                        # TODO predict the values for validation not single_col; done like in sato example
                        elif "gittables" in task:
                            vl_pred_list += filtered_logits.argmax(
                                1).cpu().detach().numpy().tolist()
                            vl_true_list += batch["label"].cpu().detach().numpy(
                            ).tolist()
                        

                        if "sato" in task:
                            loss = loss_fn(filtered_logits, batch["label"])
                        elif "turl" in task:
                            loss = loss_fn(filtered_logits, batch["label"].float())
                        elif "SportsTables" in task:
                            loss = loss_fn(filtered_logits, batch["label"])
                        # TODO calculate loss for validation not single_col; done like in sato example
                        elif "gittables" in task:
                            loss = loss_fn(filtered_logits, batch["label"])

                    vl_loss += loss.item()

                vl_loss /= (len(valid_dataset) / batch_size)
                if "sato" in task:
                    vl_micro_f1 = f1_score(vl_true_list,
                                        vl_pred_list,
                                        average="micro")
                    vl_macro_f1 = f1_score(vl_true_list,
                                        vl_pred_list,
                                        average="macro")
                    vl_class_f1 = f1_score(vl_true_list,
                                        vl_pred_list,
                                        average=None,
                                        labels=np.arange(args.num_classes))
                elif "turl" in task:
                    vl_micro_f1, vl_macro_f1, vl_class_f1, _ = f1_score_multilabel(
                        vl_true_list, vl_pred_list)
                elif "SportsTables" in task:
                    vl_micro_f1 = f1_score(vl_true_list,
                                        vl_pred_list,
                                        average="micro")
                    vl_macro_f1 = f1_score(vl_true_list,
                                        vl_pred_list,
                                        average="macro")
                    vl_class_f1 = f1_score(vl_true_list,
                                        vl_pred_list,
                                        average=None,
                                        labels=np.arange(args.num_classes))
                # TODO calculate f1 score for validation data; done like in sato example
                elif "gittables" in task:
                    print('calculate F1 score')
                    vl_micro_f1 = f1_score(vl_true_list,
                                        vl_pred_list,
                                        average="micro",
                                        zero_division=1)
                    vl_macro_f1 = f1_score(vl_true_list,
                                        vl_pred_list,
                                        average="macro",
                                        zero_division=1)
                    vl_class_f1 = f1_score(vl_true_list,
                                        vl_pred_list,
                                        average=None,
                                        labels=np.arange(args.num_classes),
                                        zero_division=1)

                if vl_micro_f1 > best_vl_micro_f1s[k]:
                    best_vl_micro_f1s[k] = vl_micro_f1
                    if len(args.tasks) >= 2:
                        model_savepath = "{}={}_best_micro_f1.pt".format(
                            tag_name, task)
                    else:
                        model_savepath = "{}_best_micro_f1.pt".format(tag_name)
                    torch.save(model.state_dict(), model_savepath)

                if vl_macro_f1 > best_vl_macro_f1s[k]:
                    best_vl_macro_f1s[k] = vl_macro_f1
                    if len(args.tasks) >= 2:
                        model_savepath = "{}={}_best_macro_f1.pt".format(
                            tag_name, task)
                    else:
                        model_savepath = "{}_best_macro_f1.pt".format(tag_name)
                    torch.save(model.state_dict(), model_savepath)

                loss_info_list.append([
                    tr_loss, tr_macro_f1, tr_micro_f1, vl_loss, vl_macro_f1,
                    vl_micro_f1
                ])
                t2 = time()
                logging.debug(
                    "Epoch {} ({}): tr_loss={:.7f} tr_macro_f1={:.4f} tr_micro_f1={:.4f} "
                    .format(epoch, task, tr_loss, tr_macro_f1, tr_micro_f1)+
                    "vl_loss={:.7f} vl_macro_f1={:.4f} vl_micro_f1={:.4f} ({:.2f} sec.)"
                    .format(vl_loss, vl_macro_f1, vl_micro_f1, (t2 - t1)))
                
                # early stopp
                if best_vl_loss > vl_loss:
                    early_stopp_counter = 0
                    best_vl_loss = vl_loss
                else:
                    logging.debug("No improvement of loss on validation dataset during the epoch")
                    early_stopp_counter +=1
            if early_stopp_counter > early_stopp_patience:
                logging.debug(f"Early stopp at epoch: {epoch} / best_vl_loss: {best_vl_loss}")
                break
    except Exception as e:
        print('Exception')
        logging.debug(epoch)
        logging.debug(batch_idx)
        logging.debug(batch)
        logging.error(e, exc_info=True)
        
    for task, loss_info_list in zip(args.tasks, loss_info_lists):
        loss_info_df = pd.DataFrame(loss_info_list,
                                    columns=[
                                        "tr_loss", "tr_f1_macro_f1",
                                        "tr_f1_micro_f1", "vl_loss",
                                        "vl_f1_macro_f1", "vl_f1_micro_f1"
                                    ])
        if len(args.tasks) >= 2:
            loss_info_df.to_csv("{}={}_loss_info.csv".format(tag_name, task))
        else:
            loss_info_df.to_csv("{}_loss_info.csv".format(tag_name))
