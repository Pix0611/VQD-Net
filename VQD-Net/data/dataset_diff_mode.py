import os
import pickle
import random
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F



RANDOM_SEED = 42

TRAIN_SIZE_RATIO = 0.8

IMPLEMENTED_DATASET = ["finefs", "rg", "fis-v", "mtl-aqa"]
SAMPLE_NUM = {
    "finefs":{
        "short program": 729,
        "free skating": 428
    },
    "rg":{
        "ribbon": 250,
        "clubs": 250,
        "hoop": 250, 
        "ball": 250
    },
    "fis-v":{
        "all": 500
    },
    "mtl-aqa":{
        "all": 1412
    }
}
FEATURE_SEQ_LENGTH_FILL_ZERO_TO = {
    "finefs":{
        "short program": 192, # 16x
        "free skating": 256 # 16x
    },
    "rg":{
        "ribbon": 80,
        "clubs": 80,
        "hoop": 80, 
        "ball": 80
    },
    "fis-v":{
        "all": 304
    },
    "mtl-aqa":{
        "all": 8 
    }
}
MAX_SCORE = {
    "finefs":{
        "short program": 65.98,
        "free skating": 129.14
    },
    "rg":{
        "ribbon": 21.7,
        "clubs": 24.0,
        "hoop": 23.6, 
        "ball": 23.7
    },
    "fis-v":{
        "all": 80.75
    },
    "mtl-aqa":{
        "all": 104.5
    }
}
MIN_SCORE = {
    "finefs":{
        "short program": 16.68,
        "free skating": 30.37
    },
    "rg":{
        "ribbon": 5.05,
        "clubs": 6.5,
        "hoop": 3.25, 
        "ball": 3.85
    },
    "fis-v":{
        "all": 16.81
    },
    "mtl-aqa":{
        "all": 0.0
    }
}

FINFFS_FS_START_INDEX = 729 


def generate_finefs_labels(label_path):
    label_json_list = os.listdir(r"/data/data-home/ylf/dataset/FineFS/labels")
    with open(label_path, "w") as f:
        for label_json_file in label_json_list:
            if label_json_file[-5:] == ".json":
                index = label_json_file[:-5]
                performance_dict = json.load(open(os.path.join(r"/data/data-home/ylf/dataset/FineFS/labels", label_json_file), "r"))
                score_value = performance_dict["total_element_score"]
                f.write("{} {}\n".format(index, score_value))
    f.close()
    
def generate_mtlaqa_labels(label_path):
    info_dict = pickle.load(open(r"/data/data-home/ylf/dataset/MTL-AQA/info/augmented_final_annotations_dict.pkl", "rb"))
    data_dir = "/data/data-home/ylf/dataset/MTL-AQA/datas"
    not_clear_key = list(info_dict.keys())
    with open(label_path, "w") as f:
        for index, (x,y) in enumerate(not_clear_key):
            score_value = info_dict[(x,y)]["final_score"]
            f.write("{} {}\n".format(index, score_value))
            
            old_filename = os.path.join(data_dir, f"{x:02d}_{y:02d}.pkl")  
            new_filename = os.path.join(data_dir, f"{index}.pkl")          
            if os.path.exists(old_filename):
                os.rename(old_filename, new_filename)
                print(f"rename: {old_filename} -> {new_filename}")
            else:
                print(f"Not exiss: {old_filename}")
            
    f.close()

class AqaDataset(Dataset):
    def __init__(self, dataset_used="finefs", subset=None):
        print("Initializing dataset...")
        self.init_dataset(dataset_used)    
        self.init_subset(subset)
        
        print("Dataset Setting Complete.")
        print("Using dataset: {}".format(self.dataset_used))
        print("Using Subset: {}".format(self.subset))
        
    def init_dataset(self, dataset_used):
        dataset_used = dataset_used.lower()
        self.dataset_used = dataset_used
        if dataset_used == "finefs":
            self.label_path = r"/data/data-home/ylf/dataset/FineFS/labels.txt"
            if not os.path.exists(self.label_path):
                generate_finefs_labels(self.label_path)
            self.label_dict = {}
            with open(self.label_path, "r") as f:
                for line in f:
                    index, score = line.strip().split()
                    self.label_dict[int(index)] = float(score)
            f.close()
            self.features_dir = r"/data/data-home/ylf/dataset/FineFS/datas"
            
        elif dataset_used == 'gdlt' or dataset_used == "rg":
            self.dataset_used = "rg"
            self.label_path = r"/data/data-home/ylf/dataset/RG/labels.txt"
            self.label_dict = {}
            with open(self.label_path, "r") as f:
                next(f)
                for line in f:
                    line_content = line.strip().split()
                    index = line_content[0]
                    score = float(line_content[3]) - float(line_content[4])
                    self.label_dict[index] = score
            self.features_dir = r"/data/data-home/ylf/dataset/RG/swintx_avg_fps25_clip32"
        
        elif dataset_used == r"fis-v":
            self.label_path = r"/data/data-home/ylf/dataset/fis-v/labels.txt"
            self.label_dict = {}
            with open(self.label_path, "r") as f:
                for line in f:
                    index, score1, score2, score3 = line.strip().split()
                    self.label_dict[int(index)] = float(score1)+float(score2)-float(score3)
            f.close()
            self.features_dir = r"/data/data-home/ylf/dataset/fis-v/swintx_avg_fps25_clip32"

        elif dataset_used == r"mtl-aqa":
            self.label_path = r"/data/data-home/ylf/dataset/MTL-AQA/labels.txt"
            if not os.path.exists(self.label_path):
                generate_mtlaqa_labels(self.label_path)
                
            self.label_dict = {}
            with open(self.label_path, "r") as f:
                for line in f:
                    index, score = line.strip().split()
                    self.label_dict[int(index)] = float(score)
            f.close()
            self.features_dir = r"/data/data-home/ylf/dataset/MTL-AQA/datas"

        else:
            print("Default use dataset: {}".format("FineFS"))
            self.label_path = r"/data/data-home/ylf/dataset/FineFS"

    def init_subset(self, subset=None):
        if subset is not None:
            subset = subset.lower() 
            
        if self.dataset_used == "finefs":
            if subset not in ["short program", "sp", "free skating", "fs"]:
                print("Subset {} of FineFS not found. Default set as {}.".format(subset, "short program"))
                self.subset = "short program"
            else:
                self.subset = "short program" if subset in ["short program", "sp"] else "free skating"
        elif self.dataset_used == "mtl-aqa":
            self.subset = "all"
        elif self.dataset_used == "fis-v":
            self.subset = "all"
        elif self.dataset_used == "rg":
            if subset not in ["ribbon", "hoop", "clubs", "ball"]:
                print("Subset {} of RG not found. Default set as {}.".format(subset, "ribbon"))
                self.subset = "ribbon"
            else:
                self.subset = subset
        
    
    def __len__(self):
        return SAMPLE_NUM[self.dataset_used][self.subset]
            
    def __getitem__(self, idx):
        feature = None
        label = 0.0
        
        if self.dataset_used == "finefs":
            if self.subset == "short program":
                feature = torch.load(
                    os.path.join(self.features_dir, "{}.pkl".format(idx))
                )
            elif self.subset == "free skating":
                idx += FINFFS_FS_START_INDEX
                feature = torch.load(
                    os.path.join(self.features_dir, "{}.pkl".format(idx))
                )
            label = self.label_dict[idx]
            
        elif self.dataset_used == "rg":
            idx += 1
            if self.subset == "ball" and idx >=134:
                idx += 1
            feature_path = os.path.join(self.features_dir, "{}_{:03}.npy".format(self.subset.title(), idx))
            feature = np.load(feature_path)
            feature = torch.from_numpy(feature)
            label = self.label_dict["{}_{:03}".format(self.subset.title(), idx)]
        
        elif self.dataset_used == "fis-v":
            idx += 1
            feature_path = os.path.join(self.features_dir, "{}.npy".format(idx))
            feature = np.load(feature_path)
            feature = torch.from_numpy(feature)
            label = self.label_dict[idx]
        
        elif self.dataset_used == "mtl-aqa":
            feature_path = os.path.join(self.features_dir, "{}.pkl".format(idx))
            feature = pickle.load(open(feature_path, "rb"))
            # mtl-aqa的特征长度是768，需要补齐长度
            L, N = feature.shape
            zero_to_append = torch.zeros(L, 1024-N)
            feature = torch.cat((feature, zero_to_append), dim=1)
            # feature = torch.load(feature_path)
            label = self.label_dict[idx]
        
        # 特征的seq——length补齐至指定长度
        feature =  F.pad(feature, (0, 0, 0, FEATURE_SEQ_LENGTH_FILL_ZERO_TO[self.dataset_used][self.subset]-feature.shape[0]), mode="constant", value=0.0)
        
        # 分数进行min-max归一化
        min_score = MIN_SCORE[self.dataset_used][self.subset]
        max_score = MAX_SCORE[self.dataset_used][self.subset]
        label = (label-min_score) / (max_score-min_score)
        
        return feature, label

# 使用示例
if __name__ == "__main__":
    # pass

    dataset_example = AqaDataset(
        dataset_used="MTL-AQA",
        # subset="ball"
    )
    print("Total nums of {}:{} is {}".format(dataset_example.dataset_used, dataset_example.subset, len(dataset_example)))
    max_seq_length = 0
    max_score = 0
    min_score = 100000
    for i in range(len(dataset_example)):
        feature, score = dataset_example[i]
        max_seq_length = max(max_seq_length, feature.shape[0])
        max_score = max(max_score, score)
        min_score = min(min_score, score)
        
    print("Max length is: {}, Max score is: {}, Min score is: {}".format(max_seq_length, max_score, min_score))