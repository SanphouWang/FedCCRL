from copy import deepcopy
import os
from pathlib import Path
import sys
from argparse import Namespace, ArgumentParser
import pickle
import time
import pandas as pd
from typing import Dict, List
import numpy as np
from rich.console import Console
import torch
from multiprocessing import cpu_count, Pool

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())

from data.partition_data import partition_and_statistic, get_partition_arguments, ALL_DOMAINS
from algorithm.server.fedavg import FedAvgServer, get_fedavg_argparser
from algorithm.server.fedprox import FedProxServer, get_fedprox_argparser
from algorithm.server.fedsr import FedSRServer, get_fedsr_argparser
from algorithm.server.GA import GAServer, get_GA_argparser
from algorithm.server.fediir import FedIIRServer, get_fediir_argparser
from algorithm.server.fedadg import FedADGServer, get_fedadg_argparser
from algorithm.server.fedms import FedMSServer, get_fedms_argparser
from utils.tools import local_time

algo2server = {
    "FedAvg": FedAvgServer,
    "FedProx": FedProxServer,
    "FedSR": FedSRServer,
    "GA": GAServer,
    "FedIIR": FedIIRServer,
    "FedADG": FedADGServer,
    "FedMS": FedMSServer,
}
algo2argparser = {
    "FedAvg": get_fedavg_argparser(),
    "FedProx": get_fedprox_argparser(),
    "FedSR": get_fedsr_argparser(),
    "GA": get_GA_argparser(),
    "FedIIR": get_fediir_argparser(),
    "FedADG": get_fedadg_argparser(),
    "FedMS": get_fedms_argparser(),
}


def get_main_argparser():
    parser = ArgumentParser(description="Main arguments.")
    parser.add_argument("--algo", type=str, default="FedAvg", choices=list(algo2server.keys()))
    parser.add_argument(
        "-d", "--dataset", type=str, default="pacs", choices=["pacs", "vlcs", "office_home"]
    )
    return parser


def process(test_domain):
    time.sleep(np.random.randint(0, 5))
    # 1. partition data
    data_args = get_partition_arguments()
    data_args.test_domain = test_domain
    data_args.dataset = main_args.dataset
    dir_name = os.path.join(begin_time, test_domain)
    data_args.directory_name = dir_name
    partition_and_statistic(deepcopy(data_args))
    # 2. train
    fl_args = algo2argparser[algo].parse_args()
    fl_args.partition_info_dir = dir_name
    fl_args.output_dir = begin_time
    fl_args.dataset = main_args.dataset
    if algo == "FedADG":
        fl_args.optimizer = "sgd"
    server = algo2server[algo](args=deepcopy(fl_args))
    server.process_classification()


def get_table():
    test_accuracy = {}
    path2dir = os.path.join("out", algo, main_args.dataset, begin_time)
    for domain in domains:
        with open(os.path.join(path2dir, domain, "test_accuracy.pkl"), "rb") as f:
            test_accuracy[domain] = round(pickle.load(f), 2)
    average_accuracy = round(np.mean(list(test_accuracy.values())), 2)
    test_accuracy["average"] = average_accuracy
    test_accuracy_df = pd.DataFrame(test_accuracy, index=[algo])
    test_accuracy_df.to_csv(os.path.join(path2dir, "test_accuracy.csv"))
    return test_accuracy


if __name__ == "__main__":
    main_args = get_main_argparser().parse_args()
    begin_time = local_time()
    # begin_time = "2024-09-05-11:40:05"
    algo = main_args.algo
    domains = ALL_DOMAINS[main_args.dataset]
    sys.argv = [sys.argv[0]]
    multiprocess = True
    if multiprocess:
        num_processes = min(len(domains), cpu_count())
        pool = Pool(processes=num_processes)
        try:
            pool.map(process, domains)
            pool.close()
            pool.join()
        except Exception as e:
            pool.terminate()
            pool.join()
            raise RuntimeError("An error occurred in one of the worker processes.") from e
    else:
        for domain in domains:
            process(domain)
    get_table()
