 ### checked!
from model import *
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='./Dataset/')  # Dataset/pdb/
parser.add_argument("--feature_path", type=str, default='./Feature/')  # Feature/Protrans/, Feature/DSSP/ 
parser.add_argument("--output_path", type=str, default='./output/')
parser.add_argument("--task", type=str, default='BS')
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--train", action='store_true', default=False)
parser.add_argument("--test", action='store_true', default=False)
parser.add_argument("--seed", type=int, default=2023)
parser.add_argument("--run_id", type=str, default=None)
args = parser.parse_args()

Seed_everything(seed=args.seed)

model_class = GPSite


train_size = {"PRO":335, "PEP":1251, "DNA":661, "RNA":689, "ZN":1646, "CA":1554, "MG":1729, "MN":547, "ATP":347, "HEM":176, "BS":8441}
num_samples = train_size[args.task] * 3 # 1个epoch等于3个


nn_config = {
    'node_input_dim': 1024 + 9 + 184,
    'edge_input_dim': 450,
    'hidden_dim': 128,
    'layer': 4,
    'augment_eps': 0.1,
    'dropout': 0.2,
    'lr': 1e-3,
    'obj_max': 1,   # optimization object: max is better
    'epochs': 25,
    'patience': 6,
    'batch_size': 16,
    'num_samples': num_samples,
    'folds': 5,
    'seed': args.seed
}

if __name__ == '__main__':
    train_and_predict(model_class, nn_config, args)

