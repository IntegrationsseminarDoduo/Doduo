from collections import defaultdict
import argparse
import json
import math
import os
import sys

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

from dataset import (collate_fn, TURLColTypeColwiseDataset,
                     TURLColTypeTablewiseDataset, TURLRelExtColwiseDataset,
                     TURLRelExtTablewiseDataset, SatoCVColwiseDataset,
                     SatoCVTablewiseDataset,SportsTablesTablewiseDataset, SportsTablesColwiseDataset,
                     GittablesCVColwiseDataset, GittablesCVTablewiseDataset)

from model import BertForMultiOutputClassification, BertMultiPairPooler
from util import parse_tagname, f1_score_multilabel
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from ast import literal_eval

import logging

if __name__ == "__main__":
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="",
        type=str,
        help="model name to validate against test data",
    )
    # parser.add_argument(
    #     "--sport_domains",
    #     type=str,
    #     nargs="+",
    #     default = ["baseball"],
    #     choices=["baseball", "basketball", "football", "hockey", "soccer"]
    # )
    
    args = parser.parse_args()

    # ============
    #tag_name = "model/sato_bert_bert-base-uncased-bs32-ml-8"
    tag_name = args.model_name
    
    shuffle_cols = False
    
    if "SportsTables" in tag_name:
        shuffle_cols_tag = tag_name.split("_")[-3]
        if shuffle_cols_tag == "scTrue":
            shuffle_cols = True        
        random_state = int(tag_name.split("_")[-2].split("rs")[1])
        sport_domains = literal_eval(tag_name.split("_")[-1])

    if "gittables" in tag_name:

        cv = tag_name.split('_')[1]
        
    #print(f"Shuffle cols: {shuffle_cols}")
    #print(f"Random state: {random_state}")
    #print(f"Sport domains: {sport_domains}")
    
    multicol_only = False
    shortcut_name, _, max_length = parse_tagname(tag_name)

    print("Max_length of Tokens: ", max_length)

    colpair = False

    # Single-column or multi-column
    if os.path.basename(tag_name).split("_")[1] == "single":
        single_col = True
    else:
        single_col = False
        
    ## logging set-up
    #logging.basicConfig(filename=f"{tag_name.replace('model/', 'eval/')}.log", filemode="w", format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    #logging.debug("args={}".format(json.dumps(vars(args))))
    #logging.debug(f"Model tag name: {tag_name}")

    # Task names
    task = os.path.basename(tag_name).split("_")[0]
    num_classes_list = []
    if task == "turl":
        tasks = ["turl"]
        num_classes_list.append(255)
    elif task == "turl-re" or task == "turl-re-colpair":  # turl-re or turl-re-colpair
        tasks = ["turl-re"]
        num_classes_list.append(121)
        if task == "turl-re-colpair":
            colpair = True
    elif task in [
            "sato0", "sato1", "sato2", "sato3", "sato4", "msato0", "msato1",
            "msato2", "msato3", "msato4"
    ]:
        if task[0] == "m":
            multicol_only = True
        tasks = [task]  # sato, sato0 , ...
        num_classes_list.append(78)
    elif task == "turlturl-re":
        tasks = ["turl", "turl-re"]
        num_classes_list.append(255)
        num_classes_list.append(121)
    elif task == "turlturl-re-colpair":
        tasks = ["turl", "turl-re"]
        num_classes_list.append(255)
        num_classes_list.append(121)
        colpair = True
    elif task == "satoturl":
        tasks = ["sato", "turl"]
        num_classes_list.append(78)
        num_classes_list.append(121)
    elif task == "satoturlturl-re":
        tasks = ["sato", "turl", "turl-re"]
        num_classes_list.append(78)
        num_classes_list.append(121)
        num_classes_list.append(255)
    elif task == "SportsTables":
        tasks = ["SportsTables"]
        num_classes_list.append(462)
    elif task in "gittables":
        tasks = [task]
        num_class_dict = {
        "0" : 13035,
        "1" : 4828,
        "2" : 7833,
        "10" : 6746,
        "11" : 4427,
        "12" : 2236,
        "20" : 7559
        }
        num_classes_list.append(num_class_dict[cv])
    else:
        raise ValueError("Invalid task name(s): {}".format(tag_name))

    for task, num_classes in zip(tasks, num_classes_list):
        #output_filepath = "{}.json".format(tag_name.replace("model/", "eval/"))
        output_filepath = "{}={}.json".format(
            tag_name.replace("model/", "eval/"), task)
        output_dirpath = os.path.dirname(output_filepath)
        if not os.path.exists(output_dirpath):
            print("{} not exist. Created.".format(output_dirpath))
            os.makedirs(output_dirpath)

        #max_length = int(tag_name.split("-")[-1])
        #batch_size = 32
        batch_size = 1
        if len(tasks) == 1:
            f1_macro_model_path = "{}_best_macro_f1.pt".format(tag_name)
            f1_micro_model_path = "{}_best_micro_f1.pt".format(tag_name)
        else:
            f1_macro_model_path = "{}={}_best_macro_f1.pt".format(
                tag_name, task)
            f1_micro_model_path = "{}={}_best_micro_f1.pt".format(
                tag_name, task)
        # ============
        #logging.debug(f"f1 macro model: {f1_macro_model_path}")
        #logging.debug(f"f1 micro model: {f1_micro_model_path}")

        tokenizer = BertTokenizer.from_pretrained(shortcut_name)

        print('num_labels = ', num_classes)

        # WIP
        if single_col:
            model_config = BertConfig.from_pretrained(shortcut_name,
                                                      num_labels=num_classes)
            model = BertForSequenceClassification(model_config).to(device)
        else:
            model = BertForMultiOutputClassification.from_pretrained(
                shortcut_name,
                num_labels=num_classes,
                output_attentions=False,
                output_hidden_states=False,
            ).to(device)

        if task == "turl-re" and colpair:
            print("Use column-pair pooling")
            #logging.debug("Use column-pair pooling")
            # Use column pair embeddings
            config = BertConfig.from_pretrained(shortcut_name)
            model.bert.pooler = BertMultiPairPooler(config).to(device)

        if task in [
                "sato0", "sato1", "sato2", "sato3", "sato4", "msato0",
                "msato1", "msato2", "msato3", "msato4"
        ]:
            if task[0] == "m":
                multicol_only = True
            else:
                multicol_only = False

            cv = int(task[-1])
            if single_col:
                dataset_cls = SatoCVColwiseDataset
            else:
                dataset_cls = SatoCVTablewiseDataset

            test_dataset = dataset_cls(cv=cv,
                                       split="test",
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       multicol_only=multicol_only,
                                       device=device)
            test_dataloader = DataLoader(test_dataset,
                                         batch_size=batch_size,
                                         collate_fn=collate_fn)
        elif "turl" in task:
            if task in ["turl"]:
                filepath = "data/table_col_type_serialized.pkl"
            elif "turl-re" in task:  # turl-re-colpair
                filepath = "data/table_rel_extraction_serialized.pkl"
            else:
                raise ValueError("turl tasks must be turl or turl-re.")

            if single_col:
                if task == "turl":
                    dataset_cls = TURLColTypeColwiseDataset
                elif task == "turl-re":
                    dataset_cls = TURLRelExtColwiseDataset
                else:
                    raise ValueError()
            else:
                if task == "turl":
                    dataset_cls = TURLColTypeTablewiseDataset
                elif task == "turl-re":
                    dataset_cls = TURLRelExtTablewiseDataset
                else:
                    raise ValueError()

            test_dataset = dataset_cls(filepath=filepath,
                                       split="test",
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       multicol_only=False,
                                       device=device)
            test_dataloader = DataLoader(test_dataset,
                                         batch_size=batch_size,
                                         collate_fn=collate_fn)
        elif task == "SportsTables":
            if single_col:
                dataset_cls = SportsTablesColwiseDataset
            else:
                dataset_cls = SportsTablesTablewiseDataset
            
            test_dataset = dataset_cls(tokenizer=tokenizer,
                                        sport_domains=sport_domains,
                                        random_state = random_state,
                                        split = "test",
                                        max_length=max_length,
                                        shuffle_cols=shuffle_cols,
                                        device=device,
                                        base_dirpath="/ext/daten-wi/slangenecker/SportsTables")
            
            test_dataloader = DataLoader(test_dataset,  
                                          batch_size=batch_size, 
                                          collate_fn=collate_fn)
            # valid_dataloader = DataLoader(dataset, 
            #                               sampler=valid_sampler, 
            #                               batch_size=batch_size, 
            #                               collate_fn=collate_fn)
        elif "gittables" in task:
            if single_col:
                dataset_cls = GittablesCVColwiseDataset
            else:
                dataset_cls = GittablesCVTablewiseDataset
            
            test_dataset = dataset_cls(cv=cv,
                                        split="test",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        train_ratio=1.0,
                                        device=device)
            
            test_dataloader = DataLoader(test_dataset,  
                                          batch_size=batch_size, 
                                          collate_fn=collate_fn)
        
        else:
            raise ValueError()

        eval_dict = defaultdict(dict)
        for f1_name, model_path in [("f1_macro", f1_macro_model_path),
                                    ("f1_micro", f1_micro_model_path)]:
            model.load_state_dict(torch.load(model_path, map_location=device))
            ts_pred_list = []
            ts_true_list = []
            # Test
            for batch_idx, batch in enumerate(test_dataloader):
                if single_col:
                    # Single-column
                    logits = model(batch["data"].T).logits
                    if "sato" in task:
                        ts_pred_list += logits.argmax(
                            1).cpu().detach().numpy().tolist()
                        ts_true_list += batch["label"].cpu().detach().numpy(
                        ).tolist()
                    elif "turl" in task:
                        ts_pred_list += (logits >= math.log(0.5)
                                         ).int().detach().cpu().tolist()
                        ts_true_list += batch["label"].cpu().detach().numpy(
                        ).tolist()
                    elif "SportsTables" in task:
                        ts_pred_list += logits.argmax(
                            1).cpu().detach().numpy().tolist()
                        ts_true_list += batch["label"].cpu().detach().numpy(
                        ).tolist()
                    elif "SportsTables" in task: # neu 
                        ts_pred_list += logits.argmax(
                            1).cpu().detach().numpy().tolist()
                        ts_true_list += batch["label"].cpu().detach().numpy(
                        ).tolist()
                    else:
                        raise ValueError(
                            "Invalid task for single-col: {}".format(task))
                else:
                    # Multi-column
                    input_data = batch["data"]
                    logits, = model(input_data)
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
                        ts_pred_list += filtered_logits.argmax(
                            1).cpu().detach().numpy().tolist()
                        ts_true_list += batch["label"].cpu().detach().numpy(
                        ).tolist()
                    elif "turl" in task:
                        if "turl-re" in task:  # turl-re-colpair
                            all_preds = (filtered_logits >= math.log(0.5)
                                         ).int().detach().cpu().numpy()
                            all_labels = batch["label"].cpu().detach().numpy()
                            idxes = np.where(all_labels > 0)[0]
                            ts_pred_list += all_preds[idxes, :].tolist()
                            ts_true_list += all_labels[idxes, :].tolist()
                        elif task == "turl":
                            ts_pred_list += (filtered_logits >= math.log(0.5)
                                             ).int().detach().cpu().tolist()
                            ts_true_list += batch["label"].cpu().detach(
                            ).numpy().tolist()
                    elif "SportsTables" in task:
                        ts_pred_list += filtered_logits.argmax(
                            1).cpu().detach().numpy().tolist()
                        ts_true_list += batch["label"].cpu().detach().numpy(
                        ).tolist()
                    elif "gittables" in task: # Jos neu dazugekommen. 13.06.23 12.10 KR
                        #print(logits.shape)
                        ts_pred_list += filtered_logits.argmax(
                            1).cpu().detach().numpy().tolist()
                        ts_true_list += batch["label"].cpu().detach().numpy(
                        ).tolist()

            if "sato" in task:
                ts_micro_f1 = f1_score(ts_true_list,
                                       ts_pred_list,
                                       average="micro")
                ts_macro_f1 = f1_score(ts_true_list,
                                       ts_pred_list,
                                       average="macro")
                ts_class_f1 = f1_score(ts_true_list,
                                       ts_pred_list,
                                       average=None,
                                       labels=np.arange(78))
                ts_conf_mat = confusion_matrix(ts_true_list,
                                               ts_pred_list,
                                               labels=np.arange(78))
            elif "turl" in task:
                ts_micro_f1, ts_macro_f1, ts_class_f1, ts_conf_mat = f1_score_multilabel(
                    ts_true_list, ts_pred_list)
            elif "SportsTables" in task:
                ts_micro_f1 = f1_score(ts_true_list,
                                       ts_pred_list,
                                       average="micro")
                ts_macro_f1 = f1_score(ts_true_list,
                                       ts_pred_list,
                                       average="macro")
                ts_class_f1 = f1_score(ts_true_list,
                                       ts_pred_list,
                                       average=None,
                                       labels=np.arange(462))
                ts_conf_mat = confusion_matrix(ts_true_list,
                                               ts_pred_list,
                                               labels=np.arange(462))
                class_report = classification_report(test_dataset.get_LabelEncoder().inverse_transform(ts_true_list), test_dataset.get_LabelEncoder().inverse_transform(ts_pred_list), output_dict=True)
                with open(output_filepath+f"_{f1_name}_class_report.json", "w") as fout:
                    json.dump(class_report, fout)
            elif "gittables" in task:
                print('True shape: ', np.array(ts_true_list).shape)
                print('Predicted shape: ', np.array(ts_pred_list).shape)

                ts_micro_f1 = f1_score(ts_true_list,
                                       ts_pred_list,
                                       average="micro", zero_division=1)
                print(ts_micro_f1)
                ts_macro_f1 = f1_score(ts_true_list,
                                       ts_pred_list,
                                       average="macro", zero_division=1)
                ts_class_f1 = f1_score(ts_true_list,
                                       ts_pred_list,
                                       average=None,
                                       labels=np.arange(7833), zero_division=1)
                ts_conf_mat = confusion_matrix(ts_true_list,
                                               ts_pred_list,
                                               labels=np.arange(7833))


            eval_dict[f1_name]["ts_micro_f1"] = ts_micro_f1
            eval_dict[f1_name]["ts_macro_f1"] = ts_macro_f1
            if type(ts_class_f1) != list:
                ts_class_f1 = ts_class_f1.tolist()
            eval_dict[f1_name]["ts_class_f1"] = ts_class_f1
            if type(ts_conf_mat) != list:
                ts_conf_mat = ts_conf_mat.tolist()
            eval_dict[f1_name]["confusion_matrix"] = ts_conf_mat

        with open(output_filepath, "w") as fout:
            json.dump(eval_dict, fout)
