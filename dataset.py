from functools import reduce
import operator
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import transformers
from glob import glob
import json
from sklearn.preprocessing import LabelEncoder
from os.path import join
import random


def collate_fn(samples):
    data = torch.nn.utils.rnn.pad_sequence(
        [sample["data"] for sample in samples])
    label = torch.cat([sample["label"] for sample in samples])
    batch = {"data": data, "label": label}
    if "idx" in samples[0]:
        # For debug purpose
        batch["idx"] = torch.cat([sample["idx"] for sample in samples])
    return batch


class SatoCVColwiseDataset(Dataset):

    def __init__(
            self,
            cv: int,
            split: str,  # train or test
            tokenizer: transformers.PreTrainedTokenizer,
            max_length: int = 128,
            multicol_only: bool = False,
            train_ratio: float = 1.0,
            device: torch.device = None,
            base_dirpath: str = "./data"):
        if device is None:
            device = torch.device('cpu')

        if multicol_only:
            basename = "msato_cv_{}.csv"
        else:
            basename = "sato_cv_{}.csv"

        assert split in ["train", "valid",
                         "test"], "split must be train or test"

        if split in ["train", "valid"]:
            df_list = []
            for i in range(5):
                if i == cv:
                    continue
                filepath = os.path.join(base_dirpath, basename.format(i))
                df_list.append(pd.read_csv(filepath))
            df = pd.concat(df_list, axis=0)
        else:
            # test
            filepath = os.path.join(base_dirpath, basename.format(cv))
            df = pd.read_csv(filepath)

        # [CLS] [SEP] will be automatically added, so max_length should be +2
        # TODO: This will be different, depending on how many columns to have

        # For learning curve
        num_tables = len(df.groupby("table_id"))
        valid_index = int(num_tables * 0.8)
        num_train = int(train_ratio * num_tables * 0.8)

        row_list = []
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            if (split == "train") and ((i >= num_train) or (i >= valid_index)):
                break
            if split == "valid" and i < valid_index:
                continue

            for _, row in group_df.iterrows():
                row_list.append(row)

        self.df = pd.DataFrame(row_list)

        # Convert into torch.Tensor
        self.df["data_tensor"] = self.df["data"].apply(
            lambda x: torch.LongTensor(
                tokenizer.encode(x,
                                 add_special_tokens=True,
                                 max_length=max_length + 2)).to(device))
        self.df["label_tensor"] = self.df["class_id"].apply(
            lambda x: torch.LongTensor([x]).to(device)
        )  # Can we reduce the size?

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"]
        }


class SatoCVTablewiseDataset(Dataset):

    def __init__(
            self,
            cv: int,
            split: str,  # train or test
            tokenizer: transformers.PreTrainedTokenizer,
            max_length: int = 128,
            multicol_only: bool = False,
            train_ratio: float = 1.0,
            device: torch.device = None,
            base_dirpath: str = "./data"):
        if device is None:
            device = torch.device('cpu')

        assert split in ["train", "valid",
                         "test"], "split must be train or test"

        if multicol_only:
            basename = "msato_cv_{}.csv"
        else:
            basename = "sato_cv_{}.csv"
            

        if split in ["train", "valid"]:
            df_list = []
            for i in range(5):
                if i == cv:
                    continue
                filepath = os.path.join(base_dirpath, basename.format(i))
                df_list.append(pd.read_csv(filepath))
            df = pd.concat(df_list, axis=0)
        else:
            # test
            filepath = os.path.join(base_dirpath, basename.format(cv))
            df = pd.read_csv(filepath)

        # [CLS] [SEP] will be automatically added, so max_length should be +2
        # TODO: This will be different, depending on how many columns to have

        # For learning curve
        num_tables = len(df.groupby("table_id"))
        valid_index = int(num_tables * 0.8)
        num_train = int(train_ratio * num_tables * 0.8)

        data_list = []
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            if (split == "train") and ((i >= num_train) or (i >= valid_index)):
                break
            if split == "valid" and i < valid_index:
                continue

            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                x, add_special_tokens=True, max_length=max_length + 2)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(
                group_df["class_id"].values).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])
        """
        # NOTE: msato contains a small portion of single-col tables. keep it to be consistent.  
        if multicol_only:
            # Check
            num_all_tables = len(self.table_df)
            self.table_df = self.table_df[self.table_df["num_col"] > 1]
            assert len(self.table_df) == num_all_tables
        """

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}


class TURLColTypeColwiseDataset(Dataset):
    """TURL column type prediction column-wise (single-column)"""

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 10,
                 multicol_only: bool = False,
                 train_ratio: float = 1.0,
                 device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        with open(filepath, "rb") as fin:
            df_dict = pickle.load(fin)

        assert split in df_dict
        self.df = df_dict[split]
        self.mlb = df_dict["mlb"]  # MultilabelBinarizer model

        # For learning curve
        num_tables = len(self.df.groupby("table_id"))
        num_train = int(train_ratio * num_tables)

        row_list = []
        for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
            if i >= num_train:
                break

            # [WARNING] This potentially affects the evaluation results as well
            if split == "train" and len(group_df) > max_colnum:
                continue

            for _, row in group_df.iterrows():
                row_list.append(row)

        self.df = pd.DataFrame(row_list)
        self.df["data_tensor"] = self.df["data"].apply(
            lambda x: torch.LongTensor(
                tokenizer.encode(
                    x, add_special_tokens=True, max_length=max_length + 2)).to(
                        device)).tolist()

        self.df["label_tensor"] = self.df["label_ids"].apply(
            lambda x: torch.LongTensor([x]).to(device))

        if multicol_only:
            # Do nothing
            pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"]
        }


class TURLColTypeTablewiseDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 10,
                 multicol_only: bool = False,
                 train_ratio: float = 1.0,
                 device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        with open(filepath, "rb") as fin:
            df_dict = pickle.load(fin)

        assert split in df_dict
        self.df = df_dict[split]
        self.mlb = df_dict["mlb"]  # MultilabelBinarizer model

        # For learning curve
        num_tables = len(self.df.groupby("table_id"))
        num_train = int(train_ratio * num_tables)

        data_list = []
        for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
            if i >= num_train:
                break

            # [WARNING] This potentially affects the evaluation results as well
            if split == "train" and len(group_df) > max_colnum:
                continue

            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                x, add_special_tokens=True, max_length=max_length + 2)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(
                group_df["label_ids"].tolist()).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])

        if multicol_only:
            self.table_df = self.table_df[self.table_df["num_col"] > 1]

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }


class TURLRelExtColwiseDataset(Dataset):
    """TURL column relation prediction column-wise (single-column)"""

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 10,
                 multicol_only: bool = False,
                 train_ratio: float = 1.0,
                 device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        with open(filepath, "rb") as fin:
            df_dict = pickle.load(fin)

        assert split in df_dict
        self.df = df_dict[split]
        self.mlb = df_dict["mlb"]  # MultilabelBinarizer model

        # For learning curve
        num_tables = len(self.df.groupby("table_id"))
        num_train = int(train_ratio * num_tables)

        row_list = []
        for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
            if i >= num_train:
                break

            # [WARNING] This potentially affects the evaluation results as well
            if split == "train" and len(group_df) > max_colnum:
                continue

            group_df = group_df.sort_values("column_id")

            for j, (_, row) in enumerate(group_df.iterrows()):
                if j == 0:
                    continue

                row["data_tensor"] = torch.LongTensor(
                    tokenizer.encode(group_df.iloc[0]["data"],
                                     add_special_tokens=True,
                                     max_length=max_length + 2) +
                    tokenizer.encode(row["data"],
                                     add_special_tokens=True,
                                     max_length=max_length + 2)).to(device)

                row_list.append(row)

        self.df = pd.DataFrame(row_list)
        self.df["label_tensor"] = self.df["label_ids"].apply(
            lambda x: torch.LongTensor([x]).to(device))

        if multicol_only:
            # Do nothing
            pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"]
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}


class TURLRelExtTablewiseDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 10,
                 multicol_only: bool = False,
                 train_ratio: float = 1.0,
                 device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        with open(filepath, "rb") as fin:
            df_dict = pickle.load(fin)

        assert split in df_dict
        self.df = df_dict[split]
        self.mlb = df_dict["mlb"]  # MultilabelBinarizer model

        # For learning curve
        num_tables = len(self.df.groupby("table_id"))
        num_train = int(train_ratio * num_tables)

        data_list = []
        for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
            if i >= num_train:
                break

            # It's probably already sorted but just in case.
            group_df = group_df.sort_values("column_id")

            # [WARNING] This potentially affects the evaluation results as well
            if split == "train" and len(group_df) > max_colnum:
                continue

            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                x, add_special_tokens=True, max_length=max_length + 2)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(
                group_df["label_ids"].tolist()).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])

        if multicol_only:
            self.table_df = self.table_df[self.table_df["num_col"] > 1]

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }
        
class SportsTablesColwiseDataset(Dataset):

    def __init__(
            self,
            tokenizer: transformers.PreTrainedTokenizer,
            sport_domains: list = ["baseball"],
            random_state:int = 1,
            split:str = "train",
            max_length: int = 128,
            train_ratio: float = 1.0,
            shuffle_cols: bool = False,
            device: torch.device = None,
            base_dirpath: str = "./data"):

        self.base_dirpath = base_dirpath
        self.sport_domains = sport_domains
        self.random_state = random_state
        self.split = split
        self.shuffle_cols = shuffle_cols
        if device is None:
            device = torch.device('cpu')

        ### 
        label_enc = self.get_LabelEncoder()
        data_list = []
        for idx_sport_domain, sport_domain in enumerate(sport_domains):
            with open(join(self.base_dirpath, sport_domain, "metadata.json")) as f:
                metadata = json.load(f)
                
            with open(join(self.base_dirpath, sport_domain, f"train_valid_test_split_{self.random_state}.json")) as f:
                train_valid_test_split = json.load(f)

            for idx_table_path, table_name_full in enumerate(train_valid_test_split[self.split]):
                # if idx_table_path != 1:
                #     continue
                table_name = table_name_full.split("/")[-1].split(".csv")[0]
                ## search for correct in key in metadata
                table_metadata_key = None
                for key in metadata.keys():
                    if key in table_name:
                        table_metadata_key = key
                if table_metadata_key == None:
                    print(f"CSV {table_name_full} not in metadata.json defined!")
                    continue

                current_df = pd.read_csv(join(self.base_dirpath, sport_domain, table_name_full))
                current_data_list = []
                if self.shuffle_cols:
                    column_list = list(range(len(current_df.columns)))
                    #random.seed(random_state)
                    random.shuffle(column_list)
                else:
                    column_list = list(range(len(current_df.columns)))
                    
                # for i in range(len(current_df.columns))   # no shuffling of the column order
                for i in column_list: # with a shuffling of the column order
                    column_name = current_df.columns[i]
                    # search for defined columns data type and semantic label in metadata
                    if column_name in metadata[table_metadata_key]["textual_cols"].keys():
                        column_data_type = "textual"
                        column_label = metadata[table_metadata_key]["textual_cols"][column_name]
                    elif column_name in metadata[table_metadata_key]["numerical_cols"].keys():
                        column_data_type = "numerical"
                        column_label = metadata[table_metadata_key]["numerical_cols"][column_name]
                    else:
                        print(f"Column {current_df.columns[i]} in {table_name} not labeled in metadata.json!")
                        continue
                    
                    current_data_list.append([
                        table_name,  # table name
                        i,  # column number
                        column_label,
                        " ".join([str(x)
                                    for x in current_df.iloc[:, i].dropna().tolist()]),
                    ])
                df = pd.DataFrame(current_data_list, columns=[
                                        "table_id", "col_idx", "class", "data"])
                if len(df) == 0:
                    print(f"Table {table_name} has no columns with assigned semantic types!")
                    continue
                
                df = df.dropna().reset_index(drop=True)
                df["class_id"] = label_enc.transform(df["class"])
                df = df.reindex(columns=["table_id", "col_idx", "class", "class_id", "data"])
                
                data_list.append(df)

        df = pd.concat(data_list, axis=0)

        # For learning curve
        num_tables = len(df.groupby("table_id"))
        valid_index = int(num_tables * 0.8)
        num_train = int(train_ratio * num_tables * 0.8)

        row_list = []
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            # if (split == "train") and ((i >= num_train) or (i >= valid_index)):
            #     break
            # if split == "valid" and i < valid_index:
            #     continue

            for _, row in group_df.iterrows():
                row_list.append(row)

        self.df = pd.DataFrame(row_list)

        # Convert into torch.Tensor
        self.df["data_tensor"] = self.df["data"].apply(
            lambda x: torch.LongTensor(
                tokenizer.encode(x,
                                 add_special_tokens=True,
                                 max_length=max_length + 2)).to(device))
        self.df["label_tensor"] = self.df["class_id"].apply(
            lambda x: torch.LongTensor([x]).to(device)
        )  # Can we reduce the size?

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"]
        }
        
    def get_all_textual_semantic_types(self,sport_domains: list = ["baseball", "basketball", "football", "hockey", "soccer"]):
        textual_semantic_types = []
        for sports_domain in sport_domains:
            with open(join(self.base_dirpath, sports_domain, "metadata.json")) as f:
                metadata = json.load(f)
                for table_key in metadata.keys():
                    textual_semantic_types.extend(
                        list(metadata[table_key]["textual_cols"].values()))

        return list(set([x for x in textual_semantic_types if x is not None]))


    def get_all_numerical_semantic_types(self,sport_domains: list = ["baseball", "basketball", "football", "hockey", "soccer"]):
        numerical_semantic_types = []
        for sports_domain in sport_domains:
            with open(join(self.base_dirpath, sports_domain, "metadata.json")) as f:
                metadata = json.load(f)
                for table_key in metadata.keys():
                    numerical_semantic_types.extend(
                        list(metadata[table_key]["numerical_cols"].values()))

        return list(set(x for x in numerical_semantic_types if x is not None))


    def get_LabelEncoder(self,sport_domains: list = ["baseball", "basketball", "football", "hockey", "soccer"]):
        all_semantic_types = self.get_all_textual_semantic_types() + \
            self.get_all_numerical_semantic_types()
        label_enc = LabelEncoder()
        label_enc.fit(all_semantic_types)
        return label_enc
        
        
class SportsTablesTablewiseDataset(Dataset):

    def __init__(
            self,
            tokenizer: transformers.PreTrainedTokenizer,
            sport_domains: list = ["baseball"],
            random_state:int = 1,
            split:str = "train",
            max_length: int = 128,
            train_ratio: float = 1.0,
            shuffle_cols: bool = False,
            device: torch.device = None,
            base_dirpath: str = "./data"):
        
        self.base_dirpath = base_dirpath
        self.sport_domains = sport_domains
        self.random_state = random_state
        self.split = split
        self.shuffle_cols = shuffle_cols
        if device is None:
            device = torch.device('cpu')
            
        ### 
        label_enc = self.get_LabelEncoder()
        data_list = []
        for idx_sport_domain, sport_domain in enumerate(sport_domains):
            with open(join(self.base_dirpath, sport_domain, "metadata.json")) as f:
                metadata = json.load(f)
            
            with open(join(self.base_dirpath, sport_domain, f"train_valid_test_split_{self.random_state}.json")) as f:
                train_valid_test_split = json.load(f)

            for idx_table_path, table_name_full in enumerate(train_valid_test_split[self.split]):
                # if idx_table_path != 1:
                #     continue
                table_name = table_name_full.split("/")[-1].split(".csv")[0]
                ## search for correct in key in metadata
                table_metadata_key = None
                for key in metadata.keys():
                    if key in table_name:
                        table_metadata_key = key
                if table_metadata_key == None:
                    print(f"CSV {table_name_full} not in metadata.json defined!")
                    continue

                current_df = pd.read_csv(join(self.base_dirpath, sport_domain, table_name_full))
                current_data_list = []
                if self.shuffle_cols:
                    column_list = list(range(len(current_df.columns)))
                    #random.seed(random_state)
                    random.shuffle(column_list)
                else:
                    column_list = list(range(len(current_df.columns)))
                    
                # for i in range(len(current_df.columns))   # no shuffling of the column order
                for i in column_list: # with a shuffling of the column order
                    column_name = current_df.columns[i]
                    # search for defined columns data type and semantic label in metadata
                    if column_name in metadata[table_metadata_key]["textual_cols"].keys():
                        column_data_type = "textual"
                        column_label = metadata[table_metadata_key]["textual_cols"][column_name]
                    elif column_name in metadata[table_metadata_key]["numerical_cols"].keys():
                        column_data_type = "numerical"
                        column_label = metadata[table_metadata_key]["numerical_cols"][column_name]
                    else:
                        print(f"Column {current_df.columns[i]} in {table_name} not labeled in metadata.json!")
                        continue
                    
                    current_data_list.append([
                        table_name,  # table name
                        i,  # column number
                        column_label,
                        " ".join([str(x)
                                    for x in current_df.iloc[:, i].dropna().tolist()]),
                    ])
                df = pd.DataFrame(current_data_list, columns=[
                                        "table_id", "col_idx", "class", "data"])
                if len(df) == 0:
                    print(f"Table {table_name} has no columns with assigned semantic types!")
                    continue
                
                df = df.dropna().reset_index(drop=True)
                df["class_id"] = label_enc.transform(df["class"])
                df = df.reindex(columns=["table_id", "col_idx", "class", "class_id", "data"])
                
                data_list.append(df)

        self.df = pd.concat(data_list, axis=0)
        

        # For learning curve
        num_tables = len(df.groupby("table_id"))
        valid_index = int(num_tables * 0.8)
        num_train = int(train_ratio * num_tables * 0.8)

        data_list = []
        for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
            # if (split == "train") and ((i >= num_train) or (i >= valid_index)):
            #     break
            # if split == "valid" and i < valid_index:
            #     continue

            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                x, add_special_tokens=True, max_length=max_length + 2)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(
                group_df["class_id"].values).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])
        """
        # NOTE: msato contains a small portion of single-col tables. keep it to be consistent.  
        if multicol_only:
            # Check
            num_all_tables = len(self.table_df)
            self.table_df = self.table_df[self.table_df["num_col"] > 1]
            assert len(self.table_df) == num_all_tables
        """

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}
        
    def get_all_textual_semantic_types(self,sport_domains: list = ["baseball", "basketball", "football", "hockey", "soccer"]):
        textual_semantic_types = []
        for sports_domain in sport_domains:
            with open(join(self.base_dirpath, sports_domain, "metadata.json")) as f:
                metadata = json.load(f)
                for table_key in metadata.keys():
                    textual_semantic_types.extend(
                        list(metadata[table_key]["textual_cols"].values()))

        return list(set([x for x in textual_semantic_types if x is not None]))


    def get_all_numerical_semantic_types(self,sport_domains: list = ["baseball", "basketball", "football", "hockey", "soccer"]):
        numerical_semantic_types = []
        for sports_domain in sport_domains:
            with open(join(self.base_dirpath, sports_domain, "metadata.json")) as f:
                metadata = json.load(f)
                for table_key in metadata.keys():
                    numerical_semantic_types.extend(
                        list(metadata[table_key]["numerical_cols"].values()))

        return list(set(x for x in numerical_semantic_types if x is not None))


    def get_LabelEncoder(self,sport_domains: list = ["baseball", "basketball", "football", "hockey", "soccer"]):
        all_semantic_types = self.get_all_textual_semantic_types() + \
            self.get_all_numerical_semantic_types()
        label_enc = LabelEncoder()
        label_enc.fit(all_semantic_types)
        return label_enc

# TODO add datapreperation for columnwise dataset; done like in sato example
# Changes same as in Tablewise preparation
class GittablesCVColwiseDataset(Dataset):

    def __init__(
            self,
            cv: int,
            split: str,  # train or test
            tokenizer: transformers.PreTrainedTokenizer,
            max_length: int = 128,
            train_ratio: float = 1.0,
            device: torch.device = None,
            base_dirpath: str = "./data"):
        if device is None:
            device = torch.device('cpu')

        
        basename = "gittables_{}.csv"

        assert split in ["train", "valid",
                         "test"], "split must be train or test"

        
        filepath = os.path.join(base_dirpath, basename.format(cv))
        df = pd.read_csv(filepath)

        # [CLS] [SEP] will be automatically added, so max_length should be +2
        # TODO: This will be different, depending on how many columns to have

        # For learning curve
        num_tables = len(df.groupby("table_id"))
        valid_index = int(num_tables * 0.7)
        test_index = int(num_tables * 0.9)
        num_train = int(train_ratio * num_tables * 0.8)

        row_list = []
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            if (split == "train") and ((i >= num_train) or (i >= valid_index)):
                break
            if split == "valid" and i > test_index:
                break
            if split == 'valid' and i < num_train:
                continue
            if split =="test" and i < test_index:
                continue

            for _, row in group_df.iterrows():
                row_list.append(row)

        self.df = pd.DataFrame(row_list)

        # Convert into torch.Tensor
        self.df["data_tensor"] = self.df["data"].apply(
            lambda x: torch.LongTensor(
                tokenizer.encode(x,
                                 add_special_tokens=True,
                                 max_length=max_length + 2, truncation=True)).to(device))
        self.df["label_tensor"] = self.df["class_id"].apply(
            lambda x: torch.LongTensor([x]).to(device)
        )  # Can we reduce the size?

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"]
        }


# TODO add Tablewise dataset class; done like in sato example
class GittablesCVTablewiseDataset(Dataset):

    def __init__(
            self,
            # TODO cv stands now for the used dataset, e.g. gittables_0 => cv = 0
            cv: int,
            split: str,  # train or test
            tokenizer: transformers.PreTrainedTokenizer,
            max_length: int = 128,
            train_ratio: float = 1.0,
            device: torch.device = None,
            base_dirpath: str = "./data"):
        if device is None:
            device = torch.device('cpu')

        assert split in ["train", "valid",
                         "test"], "split must be train or test"

        print(cv)

        # TODO change the basename to gittatbles
        basename = "gittables_{}.csv".format(cv)

        
        filepath = os.path.join(base_dirpath, basename)
        df = pd.read_csv(filepath)

        # [CLS] [SEP] will be automatically added, so max_length should be +2
        # TODO: This will be different, depending on how many columns to have

        # For learning curve
        num_tables = len(df.groupby("table_id"))
        valid_index = int(num_tables * 0.7)
        test_index = int(num_tables * 0.9)
        num_train = int(train_ratio * num_tables * 0.8)

        data_list = []
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            # TODO add new split of 70% train, 20 % validation and 10 % test data
            if (split == "train") and (i >= num_train) :
                break
            if split == "valid" and (i > test_index or i <= valid_index):
                continue
            if split == "test" and i < test_index:
                continue

            #group_df = group_df.fillna()

            # Tokenize the data and insert the CLS and SEP Tokens
            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                x, add_special_tokens=True, max_length=max_length + 2, truncation=True)).tolist(
                )
            # Turn into a tensor
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            # List the positions of the CLS tokens
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            # Turn position list into tensor
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            # Turn table header as class into tensor
            class_ids = torch.LongTensor(
                group_df["class_id"].values).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}
