from argparse import ArgumentParser, Namespace
from copy import deepcopy
import os
from pathlib import Path
import pickle
import random
import sys

import torch

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())
from algorithm.server.fedavg import FedAvgServer, get_fedavg_argparser
from algorithm.client.fedccrl import FedCCRLClient
from data.dataset import FLDataset
from utils.tools import local_time


def get_fedccrl_argparser():
    parser = get_fedavg_argparser()
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--r", type=float, default=0.1, help="upload ratio")
    parser.add_argument("--p", type=float, default=1, help="probability to process MixStyle")
    parser.add_argument(
        "--lambda1",
        type=float,
        default=0.1,
        help="the hyper-parameter for representation alignment",
    )
    parser.add_argument(
        "--lambda2",
        type=float,
        default=0.1,
        help="the hyper-parameter for prediction alignment",
    )
    parser.add_argument("-t", type=float, default=0.1, help="temperature")

    return parser


class FedCCRLServer(FedAvgServer):
    def __init__(self, algo="FedCCRL", args: Namespace = None):
        if args is None:
            args = get_fedccrl_argparser().parse_args()
        super().__init__(algo, args)

    def initialize_clients(self):
        self.client_list = [
            FedCCRLClient(self.args, FLDataset(self.args, client_id), client_id, self.logger)
            for client_id in range(self.num_client)
        ]

    def generate_statistic_pool(self):
        # generate style pool
        statistic_pool = {"mean": [], "std": []}
        for client_id in range(self.num_client):
            local_statistic_pool = self.client_list[client_id].compute_statistic()
            statistic_pool["mean"].append(local_statistic_pool["mean"])
            statistic_pool["std"].append(local_statistic_pool["std"])
            torch.cuda.empty_cache()

        return statistic_pool

    def process_classification(self):
        # distribute original model weights to clients
        for client_id in range(self.num_client):
            self.client_list[client_id].load_model_weights(
                deepcopy(self.classification_model.state_dict())
            )

        self.best_accuracy = 0
        for round_id in range(self.args.round):
            self.round_id = round_id
            self.logger.log("=" * 20, f"Round {round_id}", "=" * 20)
            # generate style pool
            statistic_pool = self.generate_statistic_pool()
            for client_id in range(self.num_client):
                self.client_list[client_id].download_statistic_pool(deepcopy(statistic_pool))
                self.client_list[client_id].train()

            aggregated_weights = self.aggregate_model()
            self.classification_model.load_state_dict(aggregated_weights)
            for client_id in range(self.num_client):
                self.client_list[client_id].load_model_weights(aggregated_weights)
            if (round_id + 1) % self.args.test_gap == 0:
                self.validate_and_test()

    def validate_and_test(self):
        self.classification_model.eval()
        self.classification_model.to(self.device)
        valid_acc = self.evaluate(self.validation_set)
        self.logger.log(f"{local_time()}, Validation, Accuracy: {valid_acc:.2f}%")
        test_acc = self.evaluate(self.test_set)
        self.logger.log(f"{local_time()}, Test, Accuracy: {test_acc:.2f}%")
        self.classification_model.to(torch.device("cpu"))

        # if test_acc > self.best_accuracy:
        self.save_checkpoint(self.round_id)
        self.best_accuracy = test_acc
        test_accuracy_file = os.path.join(self.path2output_dir, "test_accuracy.pkl")
        with open(test_accuracy_file, "wb") as f:
            pickle.dump(test_acc, f)

    def visualize_augmentation_effect(
        self,
    ):
        statistic_pool = self.generate_statistic_pool()
        for client_id in range(self.num_client):
            self.client_list[client_id].download_statistic_pool(deepcopy(statistic_pool))
        path2save = os.path.join(PROJECT_DIR, "image", "augmentation", local_time())
        if not os.path.exists(path2save):
            os.makedirs(path2save)
        random.seed(None)
        client_id = random.randint(0, self.num_client - 1)
        self.client_list[client_id].visualize_augmentation_effect(path2save)


if __name__ == "__main__":
    server = FedCCRLServer()
    server.process_classification()
