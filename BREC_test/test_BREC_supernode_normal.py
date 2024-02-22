# This program is the pipeline for testing expressiveness.
# It includes 4 stages:
#   1. pre-calculation;
#   2. dataset construction;
#   3. model construction;
#   4. evaluation


import numpy as np
import torch
import torch_geometric
import torch_geometric.loader
from loguru import logger
import time
from BRECDataset_v3 import BRECDataset
from tqdm import tqdm
import os
from torch.nn import CosineEmbeddingLoss
import argparse

from torch import nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from torch_geometric.utils import degree
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GIN, MLP, global_add_pool
import torch_geometric.transforms as T

import hashlib
import os.path as osp

from concepts.concepts import *
from concepts.transformations import AddSupernodes
from models.gnn_supernode_normal import *

torch_geometric.seed_everything(1234)

NUM_RELABEL = 32
P_NORM = 2
OUTPUT_DIM = 16
EPSILON_MATRIX = 1e-7
EPSILON_CMP = 1e-6
SAMPLE_NUM = 400
EPOCH = 100
MARGIN = 0.0
LEARNING_RATE = 1e-4
THRESHOLD = 72.34
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-4
LOSS_THRESHOLD = 0.2
SEED = 2023

global_var = globals().copy()
HYPERPARAM_DICT = dict()
for k, v in global_var.items():
    if isinstance(v, int) or isinstance(v, float):
        HYPERPARAM_DICT[k] = v

# part_dict: {graph generation type, range}
part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 360),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
}
part_dict_reduced = {
    "4-vertex_condition": (360, 380),
    "distance_regular": (380, 400),
}

parser = argparse.ArgumentParser(description="BREC Test")

parser.add_argument("--P_NORM", type=str, default="2")
parser.add_argument("--EPOCH", type=int, default=EPOCH)
parser.add_argument("--LEARNING_RATE", type=float, default=LEARNING_RATE)
parser.add_argument("--BATCH_SIZE", type=int, default=BATCH_SIZE)
parser.add_argument("--WEIGHT_DECAY", type=float, default=WEIGHT_DECAY)
parser.add_argument("--OUTPUT_DIM", type=int, default=OUTPUT_DIM)
parser.add_argument("--SEED", type=int, default=SEED)
parser.add_argument("--THRESHOLD", type=float, default=THRESHOLD)
parser.add_argument("--MARGIN", type=float, default=MARGIN)
parser.add_argument("--LOSS_THRESHOLD", type=float, default=LOSS_THRESHOLD)
parser.add_argument("--device", type=int, default=0)

parser.add_argument("--hidden_units", type=int, default=64)
parser.add_argument("--num_layers", type=int, default=6)

# General settings.
args = parser.parse_args()

P_NORM = 2 if args.P_NORM == "2" else torch.inf
EPOCH = args.EPOCH
LEARNING_RATE = args.LEARNING_RATE
BATCH_SIZE = args.BATCH_SIZE
WEIGHT_DECAY = args.WEIGHT_DECAY
OUTPUT_DIM = args.OUTPUT_DIM
SEED = args.SEED
THRESHOLD = args.THRESHOLD
MARGIN = args.MARGIN
LOSS_THRESHOLD = args.LOSS_THRESHOLD
torch_geometric.seed_everything(SEED)
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)


# Stage 1: pre calculation
# Here is for some calculation without data. e.g. generating all the k-substructures
def pre_calculation(*args, **kwargs):
    time_start = time.process_time()

    # Do something

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"pre-calculation time cost: {time_cost}")


# Stage 2: dataset construction
# Here is for dataset construction, including data processing
def get_dataset(device):
    time_start = time.process_time()

    def makefeatures(data):
        data.x = torch.ones((data.num_nodes, 1))
        return data

    def makefeatures_type(data):
        data.x = torch.tensor([1.0, 0.0, 0.0]).repeat(data.num_nodes, 1)
        return data

    concepts_list_ex = [
           {"name": "GCB", "fun": cycle_basis, "args": [200], "features" : torch.tensor([0.0,1.0,0.0])}, # max_num
           {"name": "GMC", "fun": max_cliques, "args": [],    "features" : torch.tensor([0.0,0.0,1.0])}
        ]

    path_name = ''.join(map(lambda x: x['name'] + str(x['args']), concepts_list_ex))
    hash_name = hashlib.sha256(path_name.encode('utf-8')).hexdigest()
    name_vanilla = f"BREC_{hash_name}"
    name_transf = f"BREC_supernode_normal_precalc{hash_name}"
#    name_vanilla = f"BREC_typef_{hash_name}"
#    name_transf = f"BREC_typef_supernode_normal_precalc{hash_name}"

    CHUNK_SIZE = 5000
    DATASET_LEN = 51200

#    if not osp.exists(f'./Data/{name_transf}'):
#        print("Constructing dataset")
#        dataset = BRECDataset(
#                dataset_path="/home/sam/Documents/network/supernode/dataset/BREC_raw",
#                name=name_vanilla,
#                pre_transform=makefeatures
##                pre_transform=makefeatures_type
#                )
#
#        transformed_dataset = [AddSupernodes(concepts_list_ex)(data) for data in dataset]
#        os.makedirs(f'./Data/{name_transf}')
#        for i in range(len(dataset) // CHUNK_SIZE + 1):
#            start_idx = i * CHUNK_SIZE
#            end_idx = min((i + 1) * CHUNK_SIZE, DATASET_LEN)
#            torch.save(
#                transformed_dataset[start_idx : end_idx],
#                f'./Data/{name_transf}/transformed_dataset_chunk_{i}.pth',
#            )

    loaded_dataset = []
    num_chunks = DATASET_LEN // CHUNK_SIZE + 1
    print("loading data")
    for i in tqdm(range(num_chunks)):
#        chunk = torch.load(f'./Data/{name_transf}/transformed_dataset_chunk_{i}.pth')
        chunk = torch.load(f'./Data/TBREC_supernode_normal_precalc_cycb_maxcliq_star2_minl_maxl/transformed_dataset_chunk_{i}.pth')
        loaded_dataset.extend(chunk)

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"dataset construction time cost: {time_cost}")

    return loaded_dataset


# Stage 3: model construction
# Here is for model construction.
def get_model(args, device):
    time_start = time.process_time()

#    model = get_GIN_Sadd(args, device)
#    model = get_GAT_Sadd(args, device)
#    model = get_GIN_SGIN(args, device)
    model = get_GIN_SGIN_noSINIT(args, device)
    model.to(device)

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"model construction time cost: {time_cost}")
    return model


# Stage 4: evaluation
# Here is for evaluation.
def evaluation(dataset, model, path, device, args):
    '''
        When testing on BREC, even on the same graph, the output embedding may be different,
        because numerical precision problem occur on large graphs, and even the same graph is permuted.
        However, if you want to test on some simple graphs without permutation outputting the exact same embedding,
        some modification is needed to avoid computing the inverse matrix of a zero matrix.
    '''
    # If you want to test on some simple graphs without permutation outputting the exact same embedding, please use S_epsilon.
    # S_epsilon = torch.diag(
    #     torch.full(size=(OUTPUT_DIM, 1), fill_value=EPSILON_MATRIX).reshape(-1)
    # ).to(device)
    def T2_calculation(dataset, log_flag=False):
        with torch.no_grad():
            loader = torch_geometric.loader.DataLoader(dataset, batch_size=BATCH_SIZE)
            pred_0_list = []
            pred_1_list = []
            for data in loader:
                pred = model(data.to(device)).detach()
                pred_0_list.extend(pred[0::2])
                pred_1_list.extend(pred[1::2])
            X = torch.cat([x.reshape(1, -1) for x in pred_0_list], dim=0).T
            Y = torch.cat([x.reshape(1, -1) for x in pred_1_list], dim=0).T
            if log_flag:
                logger.info(f"X_mean = {torch.mean(X, dim=1)}")
                logger.info(f"Y_mean = {torch.mean(Y, dim=1)}")
            D = X - Y
            D_mean = torch.mean(D, dim=1).reshape(-1, 1)
            S = torch.cov(D)
            inv_S = torch.linalg.pinv(S)
            # If you want to test on some simple graphs without permutation outputting the exact same embedding, please use inv_S with S_epsilon.
            # inv_S = torch.linalg.pinv(S + S_epsilon)
            return torch.mm(torch.mm(D_mean.T, inv_S), D_mean)

    time_start = time.process_time()

    # Do something
    cnt = 0
    correct_list = []
    fail_in_reliability = 0
    loss_func = CosineEmbeddingLoss(margin=MARGIN)

    for part_name, part_range in part_dict.items():
#    for part_name, part_range in part_dict_reduced.items():
        logger.info(f"{part_name} part starting ---")

        cnt_part = 0
        fail_in_reliability_part = 0
        start = time.process_time()

        for id in tqdm(range(part_range[0], part_range[1])):
            logger.info(f"ID: {id}")
            model = get_model(args, device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            dataset_traintest = dataset[
                id * NUM_RELABEL * 2 : (id + 1) * NUM_RELABEL * 2
            ]
            dataset_reliability = dataset[
                (id + SAMPLE_NUM)
                * NUM_RELABEL
                * 2 : (id + SAMPLE_NUM + 1)
                * NUM_RELABEL
                * 2
            ]
            model.train()
            e = 0
            for _ in range(EPOCH):
                traintest_loader = torch_geometric.loader.DataLoader(
                    dataset_traintest, batch_size=BATCH_SIZE
                )
                loss_all = 0
                for data in traintest_loader:
                    optimizer.zero_grad()
                    pred = model(data.to(device))
                    loss = loss_func(
                        pred[0::2],
                        pred[1::2],
                        torch.tensor([-1] * (len(pred) // 2)).to(device),
                    )
                    loss.backward()
                    optimizer.step()
                    loss_all += len(pred) / 2 * loss.item()
                loss_all /= NUM_RELABEL
                e += 1
                logger.info(f"Loss: {loss_all}")
#                if loss_all < LOSS_THRESHOLD:
#                    logger.info("Early Stop Here")
#                    break
                scheduler.step(loss_all)

            model.eval()
            T_square_traintest = T2_calculation(dataset_traintest, True)
            T_square_reliability = T2_calculation(dataset_reliability, True)

            isomorphic_flag = False
            reliability_flag = False
            if T_square_traintest > THRESHOLD and not torch.isclose(
                T_square_traintest, T_square_reliability, atol=EPSILON_CMP
            ):
                isomorphic_flag = True
            if T_square_reliability < THRESHOLD:
                reliability_flag = True

            if isomorphic_flag:
                cnt += 1
                cnt_part += 1
                correct_list.append(id)
                logger.info(f"Correct num in current part: {cnt_part}")
            if not reliability_flag:
                fail_in_reliability += 1
                fail_in_reliability_part += 1
            logger.info(f"isomorphic: {isomorphic_flag} {T_square_traintest}")
            logger.info(f"reliability: {reliability_flag} {T_square_reliability}")

        end = time.process_time()
        time_cost_part = round(end - start, 2)

        logger.info(
            f"{part_name} part costs time {time_cost_part}; Correct in {cnt_part} / {part_range[1] - part_range[0]}"
        )
        logger.info(
            f"Fail in reliability: {fail_in_reliability_part} / {part_range[1] - part_range[0]}"
        )

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"evaluation time cost: {time_cost}")

    Acc = round(cnt / SAMPLE_NUM, 2)
    logger.info(f"Correct in {cnt} / {SAMPLE_NUM}, Acc = {Acc}")

    logger.info(f"Fail in reliability: {fail_in_reliability} / {SAMPLE_NUM}")
    logger.info(correct_list)

    logger.add(f"{path}/result_show.txt", format="{message}", encoding="utf-8")
    logger.info(
        "Real_correct\tCorrect\tFail\tOUTPUT_DIM\tBATCH_SIZE\tLEARNING_RATE\tWEIGHT_DECAY\tSEED"
    )
    logger.info(
        f"{cnt-fail_in_reliability}\t{cnt}\t{fail_in_reliability}\t{OUTPUT_DIM}\t{BATCH_SIZE}\t{LEARNING_RATE}\t{WEIGHT_DECAY}\t{SEED}"
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    OUT_PATH = "result_BREC"
    NAME = "XGIN_SGIN_NoSINIT_cycb_maxcliq_star2_minl_max"
    path = os.path.join(OUT_PATH, NAME)
    os.makedirs(path, exist_ok=True)

    logger.remove(handler_id=None)
    LOG_NAME = os.path.join(path, "log.txt")
    logger.add(LOG_NAME, rotation="5MB")

    logger.info(args)

    pre_calculation()
    dataset = get_dataset(device=device)
    model = get_model(args, device)
    evaluation(dataset, model, OUT_PATH, device, args)


if __name__ == "__main__":
    main()
