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

# PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
# sys.path.append(PROJECT_DIR.as_posix())

from data.partition_data import (
    partition_and_statistic,
    get_partition_arguments,
    ALL_DOMAINS,
)
from algorithm.server.fedavg import FedAvgServer, get_fedavg_argparser
from algorithm.server.fedprox import FedProxServer, get_fedprox_argparser
from algorithm.server.fedsr import FedSRServer, get_fedsr_argparser
from algorithm.server.GA import GAServer, get_GA_argparser
from algorithm.server.fediir import FedIIRServer, get_fediir_argparser
from algorithm.server.fedadg import FedADGServer, get_fedadg_argparser
from algorithm.server.fedms import FedMSServer, get_fedms_argparser
from algorithm.server.fedmsfa import FedMSFAServer, get_fedmsfa_argparser
from algorithm.server.ccst import CCSTServer, get_ccst_argparser
from algorithm.server.fedccrl import FedCCRLServer, get_fedccrl_argparser
from utils.tools import local_time

algo2server = {
    "FedAvg": FedAvgServer,
    "FedProx": FedProxServer,
    "FedSR": FedSRServer,
    "GA": GAServer,
    "FedIIR": FedIIRServer,
    "FedADG": FedADGServer,
    "FedMS": FedMSServer,
    "FedMSFA": FedMSFAServer,
    "CCST": CCSTServer,
    "FedCCRL": FedCCRLServer,
}
algo2argparser = {
    "FedAvg": get_fedavg_argparser(),
    "FedProx": get_fedprox_argparser(),
    "FedSR": get_fedsr_argparser(),
    "GA": get_GA_argparser(),
    "FedIIR": get_fediir_argparser(),
    "FedADG": get_fedadg_argparser(),
    "FedMS": get_fedms_argparser(),
    "FedMSFA": get_fedmsfa_argparser(),
    "CCST": get_ccst_argparser(),
    "FedCCRL": get_fedccrl_argparser(),
}


def get_output_dir(args):
    if algo in ["FedAvg", "GA", "FedADG"]:
        return begin_time
    elif algo == "CCST":
        output_dir = f"k_{args.k}_upload_ratio_{args.upload_ratio}"
    elif algo == "FedSR":
        output_dir = f"L2R_{args.L2R_coeff}_CMI_{args.CMI_coeff}"
    elif algo == "FedIIR":
        output_dir = f"gamma_{args.gamma}_ema_{args.ema}"
    elif algo == "FedProx":
        output_dir = f"mu_{args.mu}"
    elif algo == "FedCCRL":
        return begin_time

    output_dir = output_dir + "_" + begin_time
    return output_dir


def get_main_argparser():
    parser = ArgumentParser(description="Main arguments.")
    parser.add_argument("-a", "--algo", type=str, default="FedMS", choices=list(algo2server.keys()))
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="pacs",
        choices=["pacs", "vlcs", "office_home"],
    )
    return parser


def process(test_domain):
    time.sleep(np.random.randint(0, 5))
    # 1. partition data
    if resume_dataset_dir is None:
        data_args = get_partition_arguments()
        data_args.test_domain = test_domain
        dir_name = os.path.join(begin_time, test_domain)
        data_args.directory_name = dir_name
        partition_and_statistic(deepcopy(data_args))
    else:
        dir_name = os.path.join(resume_dataset_dir, test_domain)
    # 2. train
    fl_args, _ = algo2argparser[algo].parse_known_args()
    fl_args.partition_info_dir = dir_name
    fl_args.output_dir = (
        get_output_dir(fl_args) if resume_run_log_dir is None else resume_run_log_dir
    )
    if "domainnet" in fl_args.dataset:
        fl_args.batch_size = 128
    if algo == "FedADG":
        fl_args.optimizer = "sgd"
    server = algo2server[algo](args=deepcopy(fl_args))
    server.process_classification()


def get_table():
    test_accuracy = {}
    args, _ = algo2argparser[algo].parse_known_args()
    path2dir = os.path.join(
        "out",
        algo,
        dataset,
        get_output_dir(args) if resume_run_log_dir is None else resume_run_log_dir,
    )
    for domain in domains:
        with open(os.path.join(path2dir, domain, "test_accuracy.pkl"), "rb") as f:
            test_accuracy[domain] = round(pickle.load(f), 2)
    average_accuracy = round(np.mean(list(test_accuracy.values())), 2)
    test_accuracy["average"] = average_accuracy
    test_accuracy_df = pd.DataFrame(test_accuracy, index=[algo])
    test_accuracy_df.to_csv(os.path.join(path2dir, "test_accuracy.csv"))
    return test_accuracy


if __name__ == "__main__":
    begin_time = local_time()
    algo = sys.argv[1]
    assert algo in algo2server.keys()
    del sys.argv[1]
    if "-d" or "--dataset" in sys.argv:
        try:
            index = sys.argv.index("-d")
        except:
            index = sys.argv.index("--dataset")
        dataset = sys.argv[index + 1]
        assert dataset in ALL_DOMAINS.keys()
    else:
        raise ValueError("Please specify the dataset.")
    # resume_run_log_dir = "eta_1.0_delta_0.1_2024-09-29-16:24:59"
    # resume_dataset_dir = "2024-09-29-16:24:59"
    resume_run_log_dir = None
    resume_dataset_dir = None

    domains = ALL_DOMAINS[dataset]
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
