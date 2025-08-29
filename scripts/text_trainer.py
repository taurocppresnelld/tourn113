#!/usr/bin/env python3
"""
Standalone script for text model training (InstructText, DPO, and GRPO)
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import uuid
import pathlib
import torch
import pandas as pd
import time

import yaml
from transformers import AutoTokenizer
from transformers import AutoConfig
from huggingface_hub import HfApi

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import create_dataset_entry
from core.config.config_handler import save_config
from core.config.config_handler import update_flash_attention
from core.dataset_utils import adapt_columns_for_dpo_dataset
from core.dataset_utils import adapt_columns_for_grpo_dataset
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import TaskType
from miner.logic.job_handler import create_reward_funcs_file


def remove_duplicates(data, field_one="instruction", field_two="output"):
    seen = set()
    unique = []
    for entry in data:
        key = (entry.get(field_one, "").strip(), entry.get(field_two, "").strip())
        if key not in seen:
            seen.add(key)
            unique.append(entry)
        # else:
        #     print(entry.get(field_one, "").strip())
    
    len_data = len(data)
    len_unique = len(unique)
    if 0.5 > len_unique/len_data:
        return data

    return unique


def normalize_text(entry, field_one="instruction", field_two="output"):
    import re
    entry[field_one] = re.sub(r"\s+", " ", entry.get(field_one, "").strip())
    entry[field_two] = re.sub(r"\s+", " ", entry.get(field_two, "").strip())
    return entry


def expand_dataframe(df: pd.DataFrame, x: int, y: int) -> pd.DataFrame:
    """
    Expand the DataFrame by repeating the first `x` rows until it has at least `y` rows.

    Args:
        df (pd.DataFrame): Original DataFrame.
        x (int): Number of initial rows to duplicate.
        y (int): Target number of rows.

    Returns:
        pd.DataFrame: Expanded DataFrame with at least `y` rows.
    """
    if x > len(df):
        raise ValueError("x exceeds the number of rows in the original DataFrame.")

    base = df.head(x)
    n_repeat = math.ceil(y / x)
    expanded = pd.concat([base] * n_repeat, ignore_index=True)

    return expanded.head(y)


def format_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, remaining_seconds)


def clean_dataset(data, len_initial, len_max, field_one="instruction", field_two="output"):
    # # if len_max < len(data):
    data = remove_duplicates(data, field_one, field_two)
    print(f"Remove duplicates data {len(data)} ({len(data) / len_initial:.1%})")

    # if len_max < len(data):
    data = [normalize_text(e, field_one, field_two) for e in data]
    print(f"Normalize data {len(data)} ({len(data) / len_initial:.1%})")

    return data


def clean_dict_list(
    data: list[dict],
    required_fields: list[str] = None,
) -> list[dict]:
    """
    Cleans a list of dictionaries, ensuring all values are strings.
    Drops any items that raise exceptions or have empty required fields.

    Parameters:
        data (list[dict]): List of input examples.
        required_fields (list[str]): Fields that must not be empty after cleaning.

    Returns:
        list[dict]: Cleaned and filtered list of dictionaries.
    """

    def to_clean_string(val):
        if isinstance(val, str):
            return val.strip()
        elif isinstance(val, list):
            return " ".join(str(x).strip() for x in val if x is not None)
        elif isinstance(val, dict):
            return str(val)
        elif val is None:
            return ""
        else:
            return str(val).strip()

    cleaned = []
    failed_items = 0
    empty_required = 0

    for i, item in enumerate(data):
        try:
            cleaned_item = {k: to_clean_string(v) for k, v in item.items()}
            if required_fields:
                if not all(cleaned_item.get(f, "").strip() for f in required_fields):
                    empty_required += 1
                    continue
            cleaned.append(cleaned_item)
        except Exception as e:
            print(f"dropping item {i} due to error: {e}")
            failed_items += 1

    print(f"Dropped {failed_items} items due to exceptions.")
    if required_fields and empty_required > 0:
        print(f"Dropped {empty_required} items with empty required fields: {required_fields}")

    return cleaned


def is_richer(chosen: str, rejected: str) -> bool:
    """Returns True if chosen is richer than rejected."""
    # Measure richness (you can customize this)
    chosen_words = len(chosen.split())
    rejected_words = int(len(rejected.split())*0.4)

    chosen_chars = len(chosen)
    rejected_chars = int(len(rejected)*0.4)

    # Require both word count and char count to be higher
    return (chosen_words > rejected_words) and (chosen_chars > rejected_chars)


def _adapt_columns_for_text_dataset(dataset_path: str, dataset_type: InstructTextDatasetType, dataset_hour: int, job_id: str, config: dict, apply_formatting: bool = False):
    data = None
    dummy_set = 0
    gpu_count = 8
    # if torch.cuda.is_available():
    #     gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        gpu_count = 2

    # print(f"micro_batch_size {config['micro_batch_size']}")
    # print(f"gradient_accumulation_steps {config['gradient_accumulation_steps']}")
    # print(f"gpu_count {gpu_count}")

    try:
        print(f"Full ===============================")

        dummy_sec_step = cst.TEXT_STEP_SEC
        dummy_full_step = int(dataset_hour*60*60*cst.TEXT_STEP_WEIGHT/dummy_sec_step)
        if dummy_full_step < 15:
            dummy_full_step = 15
        config["max_steps"] = dummy_full_step
        config['save_steps'] = int(dummy_full_step*0.15)

        print(f"Max steps {config['max_steps']}")
        print(f"Save steps {config['save_steps']}")

        # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*8)
        dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
        print(f"Dataset min {dummy_set}")

        dummy_sec = cst.TEXT_COMP_SEC
        # dummy_step = int(dataset_hour*60*60*cst.TEXT_HOUR*cst.TEXT_COMP_WEIGHT/dummy_sec/(config['max_steps']/cst.TEXT_COMP_STEP_DIV))
        dummy_step = int(dummy_set/cst.TEXT_COMP_DIV)
        if dummy_step < 2*8:
            dummy_step = 2*8
        print(f"Completion steps {dummy_step}")

        dummy_sec_step = cst.TEXT_STEP_SEC
        # print(f"Total text time {format_seconds(config['max_steps']*dummy_sec_step + int(dummy_step*dummy_sec*config['max_steps']/cst.TEXT_COMP_STEP_DIV))}")
        dummy_cycle = dummy_sec_step + dummy_step*dummy_sec
        dummy_total = config['max_steps']*dummy_cycle
        dummy_cycle_set = int(dummy_set/config['max_steps'])
        print(f"Cycle dataset {dummy_cycle_set}")
        print(f"Cycle text time {format_seconds(dummy_cycle)}")
        print(f"Full text time {format_seconds(dummy_total)}")
        # print(f"Full config {config}")
        dummy_full = int(dummy_total/3600)


        # print(f"Target ===============================")

        # dummy_target_sec = dataset_hour*60*60*cst.TEXT_HOUR
        # if dummy_target_sec < dummy_total:
        #     config["max_steps"] = int(dummy_target_sec/dummy_cycle)
        #     if config["max_steps"] < 1:
        #         config["max_steps"] = 1
                
        #     # dummy_save = int(config["max_steps"]/(dataset_hour*2))
        #     dummy_save = int(config["max_steps"]*0.15)
        #     if dummy_save > 100:
        #         dummy_save = 100
        #     config["save_steps"] = dummy_save

        #     print(f"Max steps {config['max_steps']}")
        #     print(f"Save steps {config['save_steps']}")

        #     # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*8)
        #     dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
        #     print(f"Dataset min {dummy_set}")

        #     dummy_sec = cst.TEXT_COMP_SEC
        #     # dummy_step = int(dataset_hour*60*60*cst.TEXT_HOUR*cst.TEXT_COMP_WEIGHT/dummy_sec/(config['max_steps']/cst.TEXT_COMP_STEP_DIV))
        #     dummy_step = int(dummy_set/cst.TEXT_COMP_DIV)
        #     if dummy_step < 2*8:
        #         dummy_step = 2*8
        #     print(f"Completion steps {dummy_step}")

        # print(f"Target step {config['max_steps']}")
        # print(f"Target text time {format_seconds(config['max_steps']*dummy_cycle)}")
        # # print(f"Target config {config}")


        # if gpu_count > 1:
        #     print(f"Target multigpu ===============================")

        #     config['max_steps'] = config['max_steps']*(gpu_count+1)

        #     # dummy_save = int(config["max_steps"]/(dataset_hour*2))
        #     dummy_save = int(config["max_steps"]*0.15)
        #     if dummy_save > 100:
        #         dummy_save = 100
        #     config["save_steps"] = dummy_save

        #     print(f"Max steps {config['max_steps']}")
        #     print(f"Save steps {config['save_steps']}")

        #     # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*8)
        #     dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
        #     print(f"Dataset min {dummy_set}")

        #     dummy_sec = cst.TEXT_COMP_SEC
        #     # dummy_step = int(dataset_hour*60*60*cst.TEXT_HOUR*cst.TEXT_COMP_WEIGHT/dummy_sec/(config['max_steps']/cst.TEXT_COMP_STEP_DIV))
        #     dummy_step = int(dummy_set/cst.TEXT_COMP_DIV)
        #     if dummy_step < 2*8:
        #         dummy_step = 2*8
        #     print(f"Completion steps {dummy_step}")

        #     print(f"Target step multigpu {config['max_steps']}")
        #     print(f"Target text multigpu time {format_seconds(config['max_steps']*dummy_cycle)}")
        #     # print(f"Target config {config}")
        #     dummy_multi = int(config['max_steps']*dummy_cycle/3600)


        # print(f"Flash ===============================")

        # # dummy_flash = dataset_hour*cst.TEXT_HOUR
        # dummy_flash = dataset_hour*dummy_full/dummy_multi
        # if dummy_full < dummy_multi:
        #     dummy_flash = dataset_hour*1
        # # if dummy_flash < 24:
        # #     dummy_flash = 24

        # # config["max_steps"] = int(config["max_steps"]/(config['max_steps']*dummy_cycle/3600)*dummy_flash)*3
        # config["max_steps"] = int(dummy_full_step*(dataset_hour/dummy_full))
        # if config["max_steps"] > dummy_full_step:
        #     config["max_steps"] = dummy_full_step

        # # dummy_save = int(config["max_steps"]/(dataset_hour*2))
        # dummy_save = int(config["max_steps"]*0.15)
        # config["save_steps"] = dummy_save
        # # config["max_steps"] = 3
        # if config["save_steps"] < 20:
        #     config["save_steps"] = 20

        # print(f"Flash step {config['max_steps']}")
        # print(f"Save step {config['save_steps']}")
        # # print(f"Flash config {config}")

        # # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*1)
        # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
        # print(f"Dataset min {dummy_set}")


        # print(f"Final ===============================")

        # config["max_steps"] = int(dummy_full_step*cst.TEXT_HOUR)
        # # config["max_steps"] = 3
        # # if config["max_steps"] > dummy_full_step:
        # #     config["max_steps"] = dummy_full_step

        # # dummy_save = int(config["max_steps"]/(dataset_hour*2))
        # dummy_save = int(config["max_steps"]*0.15)
        # config["save_steps"] = dummy_save
        # if config["save_steps"] < 20:
        #     config["save_steps"] = 20

        # dummy_warmup = int(config["max_steps"]*0.25)
        # config["warmup_steps"] = dummy_warmup
        # if config["warmup_steps"] < 20:
        #     config["warmup_steps"] = 20

        # print(f"Final step {config['max_steps']}")
        # print(f"Save step {config['save_steps']}")
        # print(f"Warm step {config['warmup_steps']}")
        # # print(f"Flash config {config}")

        # # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*1)
        # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
        # print(f"Dataset min {dummy_set}")


        # # save_config(config, config_path)


    except Exception as e:
        df = pd.DataFrame(data)
        print(f"Failed to count dataset: {str(e)}")


    print(f"Dataset ===============================")

    with open(dataset_path, 'r') as f:
        data = json.load(f)
    file_size = os.path.getsize(dataset_path)
    print(f"File path: {dataset_path}")
    print(f"File size: {file_size} bytes")
    # data = clean_dict_list(data, required_fields=[dataset_type.field_instruction])
    pd.set_option('display.max_columns', None) 
    df = pd.DataFrame(data)
    # df = clean_dataframe_text_fields(df, required_fields=[dataset_type.field_instruction])
    result = df.head(3)
    print(result)

    len_initial = len(data)
    print(f"Init data {len_initial}")


    dummy_max = config["max_steps"]
    if len_initial > dummy_set:
        dummy_max = int(len_initial/dummy_cycle_set)
    config["max_steps"] = int(dummy_max*1.05)

    dummy_save = int(dummy_max*0.10)
    config["save_steps"] = dummy_save
    if config["save_steps"] < 20:
        config["save_steps"] = 20
    elif config["save_steps"] > 200:
        config["save_steps"] = 100
    else:
        config["save_steps"] = 50

    dummy_warmup = int(dummy_max*0.20)
    config["warmup_steps"] = dummy_warmup
    if config["warmup_steps"] < 20:
        config["warmup_steps"] = 20
    if config["warmup_steps"] > 200:
        config["warmup_steps"] = 200

    # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*1)
    dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
    if dummy_set > len_initial:
        dummy_set = len_initial
    if dummy_set > 150000:
        dummy_set = 150000

    # save_config(config, config_path)


    if dummy_set > df.shape[0]:
        print(f"Expanded text from {df.shape[0]} → {dummy_set} samples")
        df_expanded = expand_dataframe(df, x=df.shape[0], y=dummy_set)
        df = df_expanded

    elif dummy_set < df.shape[0]:
        print(f"Resized text from {df.shape[0]} → {dummy_set} samples")
        df = df.sample(n=dummy_set, random_state=42)

    output_data = df.to_dict(orient='records')
    with open(dataset_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    file_sizex = os.path.getsize(dataset_path)
    print(f"File size: {file_sizex} bytes ({file_sizex / file_size:.1%})") 


    print(f"Maximum ===============================")

    print(f"Maximum step {config['max_steps']}")
    print(f"Save step {config['save_steps']}")
    print(f"Warm step {config['warmup_steps']}")
    print(f"Dataset min {dummy_set}")

    return config


def _adapt_columns_for_dpo_dataset(dataset_path: str, dataset_type: DpoDatasetType, dataset_hour: int, job_id: str, config: dict, apply_formatting: bool = False):
    data = None
    dummy_set = 0
    gpu_count = 8
    # if torch.cuda.is_available():
    #     gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        gpu_count = 2

    # print(f"micro_batch_size {config['micro_batch_size']}")
    # print(f"gradient_accumulation_steps {config['gradient_accumulation_steps']}")
    # print(f"gpu_count {gpu_count}")

    try:
        print(f"Full ===============================")

        dummy_sec_step = cst.DPO_STEP_SEC
        dummy_full_step = int(dataset_hour*60*60*cst.DPO_STEP_WEIGHT/dummy_sec_step)
        if dummy_full_step < 15:
            dummy_full_step = 15
        config["max_steps"] = dummy_full_step
        config['save_steps'] = int(dummy_full_step*0.15)

        print(f"Max steps {config['max_steps']}")
        print(f"Save steps {config['save_steps']}")

        # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*2)
        dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
        print(f"Dataset min {dummy_set}")

        dummy_sec = cst.DPO_COMP_SEC
        # dummy_step = int(dataset_hour*60*60*cst.DPO_HOUR*cst.DPO_COMP_WEIGHT/dummy_sec/(config['max_steps']/cst.DPO_COMP_STEP_DIV))
        dummy_step = int(dummy_set/cst.DPO_COMP_DIV)
        if dummy_step < 2*2:
            dummy_step = 2*2
        print(f"Completion steps {dummy_step}")

        dummy_sec_step = cst.DPO_STEP_SEC
        # print(f"Total dpo time {format_seconds(config['max_steps']*dummy_sec_step + int(dummy_step*dummy_sec*config['max_steps']/cst.DPO_COMP_STEP_DIV))}")
        dummy_cycle = dummy_sec_step + dummy_step*dummy_sec
        dummy_total = config['max_steps']*dummy_cycle
        dummy_cycle_set = int(dummy_set/config['max_steps'])
        print(f"Cycle dataset {dummy_cycle_set}")
        print(f"Cycle dpo time {format_seconds(dummy_cycle)}")
        print(f"Full dpo time {format_seconds(dummy_total)}")
        # print(f"Full config {config}")
        dummy_full = int(dummy_total/3600)


        # print(f"Target ===============================")

        # dummy_target_sec = dataset_hour*60*60*cst.DPO_HOUR
        # if dummy_target_sec < dummy_total:
        #     config["max_steps"] = int(dummy_target_sec/dummy_cycle)
        #     if config["max_steps"] < 1:
        #         config["max_steps"] = 1
                
        #     # dummy_save = int(config["max_steps"]/(dataset_hour*2))
        #     dummy_save = int(config["max_steps"]*0.15)
        #     if dummy_save > 100:
        #         dummy_save = 100
        #     config["save_steps"] = dummy_save

        #     print(f"Max steps {config['max_steps']}")
        #     print(f"Save steps {config['save_steps']}")

        #     # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*2)
        #     dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
        #     print(f"Dataset min {dummy_set}")

        #     dummy_sec = cst.DPO_COMP_SEC
        #     # dummy_step = int(dataset_hour*60*60*cst.DPO_HOUR*cst.DPO_COMP_WEIGHT/dummy_sec/(config['max_steps']/cst.DPO_COMP_STEP_DIV))
        #     dummy_step = int(dummy_set/cst.DPO_COMP_DIV)
        #     if dummy_step < 2*2:
        #         dummy_step = 2*2
        #     print(f"Completion steps {dummy_step}")

        # print(f"Target step {config['max_steps']}")
        # print(f"Target dpo time {format_seconds(config['max_steps']*dummy_cycle)}")
        # # print(f"Target config {config}")


        # if gpu_count > 1:
        #     print(f"Target multigpu ===============================")

        #     config['max_steps'] = config['max_steps']*(gpu_count+1)

        #     # dummy_save = int(config["max_steps"]/(dataset_hour*2))
        #     dummy_save = int(config["max_steps"]*0.15)
        #     if dummy_save > 100:
        #         dummy_save = 100
        #     config["save_steps"] = dummy_save

        #     print(f"Max steps {config['max_steps']}")
        #     print(f"Save steps {config['save_steps']}")

        #     # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*2)
        #     dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
        #     print(f"Dataset min {dummy_set}")

        #     dummy_sec = cst.DPO_COMP_SEC
        #     # dummy_step = int(dataset_hour*60*60*cst.DPO_HOUR*cst.DPO_COMP_WEIGHT/dummy_sec/(config['max_steps']/cst.DPO_COMP_STEP_DIV))
        #     dummy_step = int(dummy_set/cst.DPO_COMP_DIV)
        #     if dummy_step < 2*2:
        #         dummy_step = 2*2
        #     print(f"Completion steps {dummy_step}")

        #     print(f"Target step multigpu {config['max_steps']}")
        #     print(f"Target dpo multigpu time {format_seconds(config['max_steps']*dummy_cycle)}")
        #     # print(f"Target config {config}")
        #     dummy_multi = int(config['max_steps']*dummy_cycle/3600)


        # print(f"Flash ===============================")

        # # dummy_flash = dataset_hour*cst.DPO_HOUR
        # dummy_flash = dataset_hour*dummy_full/dummy_multi
        # if dummy_full < dummy_multi:
        #     dummy_flash = dataset_hour*1
        # # if dummy_flash < 16:
        # #     dummy_flash = 16

        # # config["max_steps"] = int(config["max_steps"]/(config['max_steps']*dummy_cycle/3600)*dummy_flash)*3
        # config["max_steps"] = int(dummy_full_step*(dataset_hour/dummy_full))
        # if config["max_steps"] > dummy_full_step:
        #     config["max_steps"] = dummy_full_step

        # # dummy_save = int(config["max_steps"]/(dataset_hour*2))
        # dummy_save = int(config["max_steps"]*0.15)
        # config["save_steps"] = dummy_save
        # # config["max_steps"] = 3
        # if config["save_steps"] < 20:
        #     config["save_steps"] = 20

        # print(f"Flash step {config['max_steps']}")
        # print(f"Save step {config['save_steps']}")
        # # print(f"Flash config {config}")

        # # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*1)
        # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
        # print(f"Dataset min {dummy_set}")


        # print(f"Final ===============================")

        # config["max_steps"] = int(dummy_full_step*cst.DPO_HOUR)
        # # config["max_steps"] = 3
        # # if config["max_steps"] > dummy_full_step:
        # #     config["max_steps"] = dummy_full_step

        # # dummy_save = int(config["max_steps"]/(dataset_hour*2))
        # dummy_save = int(config["max_steps"]*0.15)
        # config["save_steps"] = dummy_save
        # if config["save_steps"] < 20:
        #     config["save_steps"] = 20

        # dummy_warmup = int(config["max_steps"]*0.25)
        # config["warmup_steps"] = dummy_warmup
        # if config["warmup_steps"] < 20:
        #     config["warmup_steps"] = 20

        # print(f"Final step {config['max_steps']}")
        # print(f"Save step {config['save_steps']}")
        # print(f"Warm step {config['warmup_steps']}")
        # # print(f"Flash config {config}")

        # # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*1)
        # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
        # print(f"Dataset min {dummy_set}")


        # # save_config(config, config_path)


    except Exception as e:
        df = pd.DataFrame(data)
        print(f"Failed to count dataset: {str(e)}")


    print(f"Dataset ===============================")

    with open(dataset_path, 'r') as f:
        data = json.load(f)
    file_size = os.path.getsize(dataset_path)
    print(f"File path: {dataset_path}")
    print(f"File size: {file_size} bytes")
    # data = clean_dict_list(data, required_fields=[dataset_type.field_prompt])
    pd.set_option('display.max_columns', None) 
    df = pd.DataFrame(data)
    # df = clean_dataframe_text_fields(df, required_fields=[dataset_type.field_prompt])
    result = df.head(3)
    print(result)

    len_initial = len(data)
    print(f"Init data {len_initial}")


    dummy_max = config["max_steps"]
    if len_initial > dummy_set:
        dummy_max = int(len_initial/dummy_cycle_set)
    config["max_steps"] = int(dummy_max*1.05)

    dummy_save = int(dummy_max*0.10)
    config["save_steps"] = dummy_save
    if config["save_steps"] < 20:
        config["save_steps"] = 20
    elif config["save_steps"] > 200:
        config["save_steps"] = 100
    else:
        config["save_steps"] = 50

    dummy_warmup = int(dummy_max*0.20)
    config["warmup_steps"] = dummy_warmup
    if config["warmup_steps"] < 20:
        config["warmup_steps"] = 20
    if config["warmup_steps"] > 200:
        config["warmup_steps"] = 200

    # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*1)
    dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
    if dummy_set > len_initial:
        dummy_set = len_initial
    if dummy_set > 150000:
        dummy_set = 150000

    # save_config(config, config_path)


    if dummy_set > df.shape[0]:
        print(f"Expanded text from {df.shape[0]} → {dummy_set} samples")
        df_expanded = expand_dataframe(df, x=df.shape[0], y=dummy_set)
        df = df_expanded

    elif dummy_set < df.shape[0]:
        print(f"Resized text from {df.shape[0]} → {dummy_set} samples")
        df = df.sample(n=dummy_set, random_state=42)

    output_data = df.to_dict(orient='records')
    with open(dataset_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    file_sizex = os.path.getsize(dataset_path)
    print(f"File size: {file_sizex} bytes ({file_sizex / file_size:.1%})") 


    print(f"Maximum ===============================")

    print(f"Maximum step {config['max_steps']}")
    print(f"Save step {config['save_steps']}")
    print(f"Warm step {config['warmup_steps']}")
    print(f"Dataset min {dummy_set}")

    return config


def _adapt_columns_for_grpo_dataset(dataset_path: str, dataset_type: GrpoDatasetType, dataset_hour: int, job_id: str, config: dict):
    data = None
    dummy_set = 0
    gpu_count = 8
    # if torch.cuda.is_available():
    #     gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        gpu_count = 2

    # print(f"micro_batch_size {config['micro_batch_size']}")
    # print(f"gradient_accumulation_steps {config['gradient_accumulation_steps']}")
    # print(f"gpu_count {gpu_count}")

    try:
        print(f"Full ===============================")

        dummy_sec_step = cst.GRPO_STEP_SEC
        dummy_full_step = int(dataset_hour*60*60*cst.GRPO_STEP_WEIGHT/dummy_sec_step)
        if dummy_full_step < 15:
            dummy_full_step = 15
        config["max_steps"] = dummy_full_step
        config['save_steps'] = int(dummy_full_step*0.15)

        print(f"Max steps {config['max_steps']}")
        print(f"Save steps {config['save_steps']}")

        # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*1)
        dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
        print(f"Dataset min {dummy_set}")

        dummy_sec = cst.GRPO_COMP_SEC
        # # dummy_step = int(dataset_hour*60*60*cst.GRPO_HOUR*cst.GRPO_COMP_WEIGHT/dummy_sec/(config['max_steps']/cst.GRPO_COMP_STEP_DIV))
        # dummy_step = int(dummy_set/cst.GRPO_COMP_DIV)
        # if dummy_step < 2*1:
        #     dummy_step = 2*1
        dummy_step = 8
        print(f"Completion steps {dummy_step}")

        # dummy_sec_step = cst.GRPO_STEP_SEC
        # print(f"Total grpo time {format_seconds(config['max_steps']*dummy_sec_step + int(dummy_step*dummy_sec*config['max_steps']/cst.GRPO_COMP_STEP_DIV))}")
        dummy_cycle = dummy_sec_step + dummy_step*dummy_sec
        dummy_total = config['max_steps']*dummy_cycle
        dummy_cycle_set = int(dummy_set/config['max_steps'])
        print(f"Cycle dataset {dummy_cycle_set}")
        print(f"Cycle grpo time {format_seconds(dummy_cycle)}")
        print(f"Full grpo time {format_seconds(dummy_total)}")
        # print(f"Full config {config}")
        dummy_full = int(dummy_total/3600)


        # print(f"Target ===============================")

        # dummy_target_sec = dataset_hour*60*60*cst.GRPO_HOUR
        # if dummy_target_sec < dummy_total:
        #     config["max_steps"] = int(dummy_target_sec/dummy_cycle)
        #     if config["max_steps"] < 1:
        #         config["max_steps"] = 1
                
        #     # dummy_save = int(config["max_steps"]/(dataset_hour*2))
        #     dummy_save = int(config["max_steps"]*0.15)
        #     if dummy_save > 100:
        #         dummy_save = 100
        #     config["save_steps"] = dummy_save

        #     print(f"Max steps {config['max_steps']}")
        #     print(f"Save steps {config['save_steps']}")

        #     # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*1)
        #     dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
        #     print(f"Dataset min {dummy_set}")

        #     # dummy_sec = cst.GRPO_COMP_SEC
        #     # # dummy_step = int(dataset_hour*60*60*cst.GRPO_HOUR*cst.GRPO_COMP_WEIGHT/dummy_sec/(config['max_steps']/cst.GRPO_COMP_STEP_DIV))
        #     # dummy_step = int(dummy_set/cst.GRPO_COMP_DIV)
        #     # if dummy_step < 2*1:
        #     #     dummy_step = 2*1
        #     dummy_step = 8
        #     print(f"Completion steps {dummy_step}")

        # print(f"Target step {config['max_steps']}")
        # print(f"Target grpo time {format_seconds(config['max_steps']*dummy_cycle)}")
        # # print(f"Target config {config}")


        # if gpu_count > 1:
        #     print(f"Target multigpu ===============================")

        #     config['max_steps'] = config['max_steps']*(gpu_count+1)

        #     # dummy_save = int(config["max_steps"]/(dataset_hour*2))
        #     dummy_save = int(config["max_steps"]*0.15)
        #     if dummy_save > 100:
        #         dummy_save = 100
        #     config["save_steps"] = dummy_save

        #     print(f"Max steps {config['max_steps']}")
        #     print(f"Save steps {config['save_steps']}")

        #     # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*1)
        #     dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
        #     print(f"Dataset min {dummy_set}")

        #     # dummy_sec = cst.GRPO_COMP_SEC
        #     # # dummy_step = int(dataset_hour*60*60*cst.GRPO_HOUR*cst.GRPO_COMP_WEIGHT/dummy_sec/(config['max_steps']/cst.GRPO_COMP_STEP_DIV))
        #     # dummy_step = int(dummy_set/cst.GRPO_COMP_DIV)
        #     # if dummy_step < 2*1:
        #     #     dummy_step = 2*1
        #     dummy_step = 8
        #     print(f"Completion steps {dummy_step}")

        #     print(f"Target step multigpu {config['max_steps']}")
        #     print(f"Target grpo multigpu time {format_seconds(config['max_steps']*dummy_cycle)}")
        #     # print(f"Target config {config}")
        #     dummy_multi = int(config['max_steps']*dummy_cycle/3600)


        # print(f"Flash ===============================")

        # # dummy_flash = dataset_hour*cst.GRPO_HOUR
        # dummy_flash = dataset_hour*dummy_full/dummy_multi
        # if dummy_full < dummy_multi:
        #     dummy_flash = dataset_hour*1
        # # if dummy_flash < 8:
        # #     dummy_flash = 8

        # # config["max_steps"] = int(config["max_steps"]/(config['max_steps']*dummy_cycle/3600)*dummy_flash)*3
        # config["max_steps"] = int(dummy_full_step*(dataset_hour/dummy_full))
        # if config["max_steps"] > dummy_full_step:
        #     config["max_steps"] = dummy_full_step

        # # dummy_save = int(config["max_steps"]/(dataset_hour*2))
        # dummy_save = int(config["max_steps"]*0.15)
        # config["save_steps"] = dummy_save
        # # config["max_steps"] = 3
        # if config["save_steps"] < 15:
        #     config["save_steps"] = 15

        # print(f"Flash step {config['max_steps']}")
        # print(f"Save step {config['save_steps']}")
        # # print(f"Flash config {config}")

        # # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*1)
        # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
        # print(f"Dataset min {dummy_set}")


        # print(f"Final ===============================")

        # config["max_steps"] = int(dummy_full_step*cst.GRPO_HOUR)
        # # config["max_steps"] = 3
        # # if config["max_steps"] > dummy_full_step:
        # #     config["max_steps"] = dummy_full_step

        # # dummy_save = int(config["max_steps"]/(dataset_hour*2))
        # dummy_save = int(config["max_steps"]*0.15)
        # config["save_steps"] = dummy_save
        # if config["save_steps"] < 20:
        #     config["save_steps"] = 20

        # dummy_warmup = int(config["max_steps"]*0.25)
        # config["warmup_steps"] = dummy_warmup
        # if config["warmup_steps"] < 20:
        #     config["warmup_steps"] = 20

        # print(f"Final step {config['max_steps']}")
        # print(f"Save step {config['save_steps']}")
        # print(f"Warm step {config['warmup_steps']}")
        # # print(f"Flash config {config}")

        # # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*1)
        # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
        # print(f"Dataset min {dummy_set}")


        # # save_config(config, config_path)


    except Exception as e:
        df = pd.DataFrame(data)
        print(f"Failed to count dataset: {str(e)}")


    print(f"Dataset ===============================")

    with open(dataset_path, 'r') as f:
        data = json.load(f)
    file_size = os.path.getsize(dataset_path)
    print(f"File path: {dataset_path}")
    print(f"File size: {file_size} bytes")
    # data = clean_dict_list(data, required_fields=[dataset_type.field_prompt])
    pd.set_option('display.max_columns', None) 
    df = pd.DataFrame(data)
    # df = clean_dataframe_text_fields(df, required_fields=[dataset_type.field_prompt])
    result = df.head(3)
    print(result)

    len_initial = len(data)
    print(f"Init data {len_initial}")


    dummy_max = config["max_steps"]
    if len_initial > dummy_set:
        dummy_max = int(len_initial/dummy_cycle_set)
    config["max_steps"] = int(dummy_max*1.05)

    dummy_save = int(dummy_max*0.10)
    config["save_steps"] = dummy_save
    if config["save_steps"] < 20:
        config["save_steps"] = 20
    elif config["save_steps"] > 200:
        config["save_steps"] = 100
    else:
        config["save_steps"] = 50

    dummy_warmup = int(dummy_max*0.20)
    config["warmup_steps"] = dummy_warmup
    if config["warmup_steps"] < 20:
        config["warmup_steps"] = 20
    if config["warmup_steps"] > 200:
        config["warmup_steps"] = 200

    # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*1)
    dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
    if dummy_set > len_initial:
        dummy_set = len_initial
    if dummy_set > 150000:
        dummy_set = 150000

    # save_config(config, config_path)


    if dummy_set > df.shape[0]:
        print(f"Expanded text from {df.shape[0]} → {dummy_set} samples")
        df_expanded = expand_dataframe(df, x=df.shape[0], y=dummy_set)
        df = df_expanded

    elif dummy_set < df.shape[0]:
        print(f"Resized text from {df.shape[0]} → {dummy_set} samples")
        df = df.sample(n=dummy_set, random_state=42)

    output_data = df.to_dict(orient='records')
    with open(dataset_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    file_sizex = os.path.getsize(dataset_path)
    print(f"File size: {file_sizex} bytes ({file_sizex / file_size:.1%})") 


    print(f"Maximum ===============================")

    print(f"Maximum step {config['max_steps']}")
    print(f"Save step {config['save_steps']}")
    print(f"Warm step {config['warmup_steps']}")
    print(f"Dataset min {dummy_set}")

    return config


def _adapt_columns_for_chat_dataset(dataset_path: str, dataset_type: ChatTemplateDatasetType, dataset_hour: int, job_id: str, config: dict, apply_formatting: bool = False):
    data = None
    dummy_set = 0
    gpu_count = 8
    # if torch.cuda.is_available():
    #     gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        gpu_count = 2

    # print(f"micro_batch_size {config['micro_batch_size']}")
    # print(f"gradient_accumulation_steps {config['gradient_accumulation_steps']}")
    # print(f"gpu_count {gpu_count}")

    try:
        print(f"Full ===============================")

        dummy_sec_step = cst.CHAT__STEP_SEC
        dummy_full_step = int(dataset_hour*60*60*cst.CHAT__STEP_WEIGHT/dummy_sec_step)
        if dummy_full_step < 15:
            dummy_full_step = 15
        config["max_steps"] = dummy_full_step
        config['save_steps'] = int(dummy_full_step*0.15)

        print(f"Max steps {config['max_steps']}")
        print(f"Save steps {config['save_steps']}")

        # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*8)
        dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
        print(f"Dataset min {dummy_set}")

        dummy_sec = cst.CHAT__COMP_SEC
        # dummy_step = int(dataset_hour*60*60*cst.CHAT__HOUR*cst.CHAT__COMP_WEIGHT/dummy_sec/(config['max_steps']/cst.CHAT__COMP_STEP_DIV))
        dummy_step = int(dummy_set/cst.CHAT__COMP_DIV)
        if dummy_step < 2*8:
            dummy_step = 2*8
        print(f"Completion steps {dummy_step}")

        dummy_sec_step = cst.CHAT__STEP_SEC
        # print(f"Total text time {format_seconds(config['max_steps']*dummy_sec_step + int(dummy_step*dummy_sec*config['max_steps']/cst.CHAT__COMP_STEP_DIV))}")
        dummy_cycle = dummy_sec_step + dummy_step*dummy_sec
        dummy_total = config['max_steps']*dummy_cycle
        dummy_cycle_set = int(dummy_set/config['max_steps'])
        print(f"Cycle dataset {dummy_cycle_set}")
        print(f"Cycle text time {format_seconds(dummy_cycle)}")
        print(f"Full text time {format_seconds(dummy_total)}")
        # print(f"Full config {config}")
        dummy_full = int(dummy_total/3600)


        # print(f"Target ===============================")

        # dummy_target_sec = dataset_hour*60*60*cst.CHAT__HOUR
        # if dummy_target_sec < dummy_total:
        #     config["max_steps"] = int(dummy_target_sec/dummy_cycle)
        #     if config["max_steps"] < 1:
        #         config["max_steps"] = 1
                
        #     # dummy_save = int(config["max_steps"]/(dataset_hour*2))
        #     dummy_save = int(config["max_steps"]*0.15)
        #     if dummy_save > 100:
        #         dummy_save = 100
        #     config["save_steps"] = dummy_save

        #     print(f"Max steps {config['max_steps']}")
        #     print(f"Save steps {config['save_steps']}")

        #     # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*8)
        #     dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
        #     print(f"Dataset min {dummy_set}")

        #     dummy_sec = cst.CHAT__COMP_SEC
        #     # dummy_step = int(dataset_hour*60*60*cst.CHAT__HOUR*cst.CHAT__COMP_WEIGHT/dummy_sec/(config['max_steps']/cst.CHAT__COMP_STEP_DIV))
        #     dummy_step = int(dummy_set/cst.CHAT__COMP_DIV)
        #     if dummy_step < 2*8:
        #         dummy_step = 2*8
        #     print(f"Completion steps {dummy_step}")

        # print(f"Target step {config['max_steps']}")
        # print(f"Target text time {format_seconds(config['max_steps']*dummy_cycle)}")
        # # print(f"Target config {config}")


        # if gpu_count > 1:
        #     print(f"Target multigpu ===============================")

        #     config['max_steps'] = config['max_steps']*(gpu_count+1)

        #     # dummy_save = int(config["max_steps"]/(dataset_hour*2))
        #     dummy_save = int(config["max_steps"]*0.15)
        #     if dummy_save > 100:
        #         dummy_save = 100
        #     config["save_steps"] = dummy_save

        #     print(f"Max steps {config['max_steps']}")
        #     print(f"Save steps {config['save_steps']}")

        #     # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*8)
        #     dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
        #     print(f"Dataset min {dummy_set}")

        #     dummy_sec = cst.CHAT__COMP_SEC
        #     # dummy_step = int(dataset_hour*60*60*cst.CHAT__HOUR*cst.CHAT__COMP_WEIGHT/dummy_sec/(config['max_steps']/cst.CHAT__COMP_STEP_DIV))
        #     dummy_step = int(dummy_set/cst.CHAT__COMP_DIV)
        #     if dummy_step < 2*8:
        #         dummy_step = 2*8
        #     print(f"Completion steps {dummy_step}")

        #     print(f"Target step multigpu {config['max_steps']}")
        #     print(f"Target text multigpu time {format_seconds(config['max_steps']*dummy_cycle)}")
        #     # print(f"Target config {config}")
        #     dummy_multi = int(config['max_steps']*dummy_cycle/3600)


        # print(f"Flash ===============================")

        # # dummy_flash = dataset_hour*cst.CHAT__HOUR
        # dummy_flash = dataset_hour*dummy_full/dummy_multi
        # if dummy_full < dummy_multi:
        #     dummy_flash = dataset_hour*1
        # # if dummy_flash < 24:
        # #     dummy_flash = 24

        # # config["max_steps"] = int(config["max_steps"]/(config['max_steps']*dummy_cycle/3600)*dummy_flash)*3
        # config["max_steps"] = int(dummy_full_step*(dataset_hour/dummy_full))
        # if config["max_steps"] > dummy_full_step:
        #     config["max_steps"] = dummy_full_step

        # # dummy_save = int(config["max_steps"]/(dataset_hour*2))
        # dummy_save = int(config["max_steps"]*0.15)
        # config["save_steps"] = dummy_save
        # # config["max_steps"] = 3
        # if config["save_steps"] < 20:
        #     config["save_steps"] = 20

        # print(f"Flash step {config['max_steps']}")
        # print(f"Save step {config['save_steps']}")
        # # print(f"Flash config {config}")

        # # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*1)
        # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
        # print(f"Dataset min {dummy_set}")


        # print(f"Final ===============================")

        # config["max_steps"] = int(dummy_full_step*cst.CHAT__HOUR)
        # # config["max_steps"] = 3
        # # if config["max_steps"] > dummy_full_step:
        # #     config["max_steps"] = dummy_full_step

        # # dummy_save = int(config["max_steps"]/(dataset_hour*2))
        # dummy_save = int(config["max_steps"]*0.15)
        # config["save_steps"] = dummy_save
        # if config["save_steps"] < 20:
        #     config["save_steps"] = 20

        # dummy_warmup = int(config["max_steps"]*0.25)
        # config["warmup_steps"] = dummy_warmup
        # if config["warmup_steps"] < 20:
        #     config["warmup_steps"] = 20

        # print(f"Final step {config['max_steps']}")
        # print(f"Save step {config['save_steps']}")
        # print(f"Warm step {config['warmup_steps']}")
        # # print(f"Flash config {config}")

        # # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*1)
        # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
        # print(f"Dataset min {dummy_set}")


        # # save_config(config, config_path)


    except Exception as e:
        df = pd.DataFrame(data)
        print(f"Failed to count dataset: {str(e)}")


    print(f"Dataset ===============================")

    with open(dataset_path, 'r') as f:
        data = json.load(f)
    file_size = os.path.getsize(dataset_path)
    print(f"File path: {dataset_path}")
    print(f"File size: {file_size} bytes")
    # data = clean_dict_list(data, required_fields=[dataset_type.field_instruction])
    pd.set_option('display.max_columns', None) 
    df = pd.DataFrame(data)
    # df = clean_dataframe_text_fields(df, required_fields=[dataset_type.field_instruction])
    result = df.head(3)
    print(result)

    len_initial = len(data)
    print(f"Init data {len_initial}")


    dummy_max = config["max_steps"]
    if len_initial > dummy_set:
        dummy_max = int(len_initial/dummy_cycle_set)
    config["max_steps"] = int(dummy_max*1.05)

    dummy_save = int(dummy_max*0.10)
    config["save_steps"] = dummy_save
    if config["save_steps"] < 20:
        config["save_steps"] = 20
    elif config["save_steps"] > 200:
        config["save_steps"] = 100
    else:
        config["save_steps"] = 50

    dummy_warmup = int(dummy_max*0.20)
    config["warmup_steps"] = dummy_warmup
    if config["warmup_steps"] < 20:
        config["warmup_steps"] = 20
    if config["warmup_steps"] > 200:
        config["warmup_steps"] = 200

    # dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps']*gpu_count/1/16*1)
    dummy_set = int(config['micro_batch_size']*config['gradient_accumulation_steps']*config['max_steps'])
    if dummy_set > len_initial:
        dummy_set = len_initial
    if dummy_set > 150000:
        dummy_set = 150000

    # save_config(config, config_path)


    if dummy_set > df.shape[0]:
        print(f"Expanded text from {df.shape[0]} → {dummy_set} samples")
        df_expanded = expand_dataframe(df, x=df.shape[0], y=dummy_set)
        df = df_expanded

    elif dummy_set < df.shape[0]:
        print(f"Resized text from {df.shape[0]} → {dummy_set} samples")
        df = df.sample(n=dummy_set, random_state=42)

    output_data = df.to_dict(orient='records')
    with open(dataset_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    file_sizex = os.path.getsize(dataset_path)
    print(f"File size: {file_sizex} bytes ({file_sizex / file_size:.1%})") 


    print(f"Maximum ===============================")

    print(f"Maximum step {config['max_steps']}")
    print(f"Save step {config['save_steps']}")
    print(f"Warm step {config['warmup_steps']}")
    print(f"Dataset min {dummy_set}")

    return config


def get_learning_rate(config: dict, task_type: str, trainable_params: int):
    """
    Auto-calculate learning rate based on trainable parameter count 
    and training type (instruct, dpo, grpo).
    
    Args:
        trainable_params (int): Number of trainable parameters (e.g., LoRA params).
        training_type (str): One of ['instruct', 'dpo', 'grpo'].
    
    Returns:
        float: Suggested learning rate.
    """

    # Base coefficient for scaling (adjust per training type)

    if task_type == TaskType.DPOTASK.value:
        c = 3e-2   # slightly lower than instruct
    elif task_type == TaskType.INSTRUCTTEXTTASK.value:
        c = 5e-2   # more forgiving
    elif task_type == TaskType.GRPOTASK.value:
        c = 1.5e-2 # RL-based, more sensitive
    elif task_type == TaskType.CHATTASK.value:
        c = 5e-2   # more forgiving

    # if training_type.lower() == "instruct":
    #     c = 5e-2   # more forgiving
    # elif training_type.lower() == "dpo":
    #     c = 3e-2   # slightly lower than instruct
    # elif training_type.lower() == "grpo":
    #     c = 1.5e-2 # RL-based, more sensitive
    # else:
    #     raise ValueError("training_type must be 'instruct', 'dpo', or 'grpo'.")

    # Scale LR inversely with sqrt of params
    lr = c / (trainable_params ** 0.5)

    # Clamp to reasonable bounds for stability

    if task_type == TaskType.DPOTASK.value:
        lr = max(min(lr, 5e-4), 3e-5)
    elif task_type == TaskType.INSTRUCTTEXTTASK.value:
        lr = max(min(lr, 1e-3), 5e-5)
    elif task_type == TaskType.GRPOTASK.value:
        lr = max(min(lr, 2e-4), 1e-5)
    elif task_type == TaskType.CHATTASK.value:
        lr = max(min(lr, 1e-3), 5e-5)

    # if training_type.lower() == "instruct":
    #     lr = max(min(lr, 1e-3), 5e-5)
    # elif training_type.lower() == "dpo":
    #     lr = max(min(lr, 5e-4), 3e-5)
    # elif training_type.lower() == "grpo":
    #     lr = max(min(lr, 2e-4), 1e-5)

    config['learning_rate'] = lr

    return config


def parse_runtime_logs(task_id: str):
    import re
    import ast

    """
    Parses a log file and extracts JSON-like loss entries.
    Each entry should look like:
    {'loss': 1.2788, 'grad_norm': 0.22516657412052155, 'learning_rate': 9e-06, 'epoch': 0.01}
    
    Returns:
        List of dicts containing the parsed entries.
    """
    pattern = re.compile(r"\{['\"]train_runtime['\"].*?\}")
    entries = []
    
    filelog = os.path.join("/workspace/axolotl/configs", f"{task_id}.log")
    with open(filelog, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                entry_str = match.group(0)
                try:
                    # Safely evaluate the JSON-like dict string
                    entry = ast.literal_eval(entry_str)
                    # print(f"{entry}")
                    entries.append(entry)
                except (ValueError, SyntaxError):
                    # Skip lines that don't parse correctly
                    continue
    return entries


def parse_loss_logs(task_id: str):
    import re
    import ast

    """
    Parses a log file and extracts JSON-like loss entries.
    Each entry should look like:
    {'loss': 1.2788, 'grad_norm': 0.22516657412052155, 'learning_rate': 9e-06, 'epoch': 0.01}
    
    Returns:
        List of dicts containing the parsed entries.
    """
    pattern = re.compile(r"\{['\"]loss['\"].*?\}")
    entries = []
    
    filelog = os.path.join("/workspace/axolotl/configs", f"{task_id}.log")
    with open(filelog, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                entry_str = match.group(0)
                try:
                    # Safely evaluate the JSON-like dict string
                    entry = ast.literal_eval(entry_str)
                    # print(f"{entry}")
                    if entry['learning_rate'] > 0.0:
                        entries.append(entry)
                except (ValueError, SyntaxError):
                    # Skip lines that don't parse correctly
                    continue
    return entries


def get_custom_text_config(param_nums: int):
    distributed = "ddp"
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            distributed = "ds"

    TEXT_CONFIG = {
        "0_1_b": {
            "learning_rate": 0.0001,
            "distributed": "ddp",
            "gpu_count": 1,
            "batch_size": 140,
            "use_lora": False
        },
        "1_2_b": {
            "learning_rate": 0.0001,
            "distributed": "ddp",
            "gpu_count": 1,
            "use_lora": False,
            "batch_size": 100,
        },
        "2_4_b": {
            "learning_rate": 8e-5,
            "distributed": "ddp",
            "gpu_count": 1,
            "batch_size": 48,
        },
        "4_5_b": {
            "learning_rate": 6e-5,
            "distributed": "ddp",
            "gpu_count": 2,
            "batch_size": 40,
        },
        "5_9_b": {
            "learning_rate": 4e-5,
            "distributed": "ddp",
            "gpu_count": 2,
            "batch_size": 30,
        },
        "9_12_b": {
            "learning_rate": 0.00015,
            "distributed": "ddp",
            "gpu_count": 2,
            "use_lora": True,
            "batch_size": 32,
        },
        "12_15_b": {
            "learning_rate": 0.0001,
            "distributed": "ddp",
            "gpu_count": 4,
            "use_lora": True,
            "batch_size": 20,
        },
        "15_40_b": {
            "learning_rate": 8e-5,
            "distributed": distributed,
            "gpu_count": 4,
            "use_lora": True,
            "batch_size": 10,
        },
        "40_80_b": {
            "learning_rate": 6e-5,
            "distributed": distributed,
            "gpu_count": 8,
            "use_lora": True,
            "batch_size": 6,
        }        
    }

    for key in TEXT_CONFIG:
        TEXT_CONFIG[key]["label"] = key

    if param_nums < 1_000_000_000:
        return TEXT_CONFIG["0_1_b"]
    elif param_nums < 2_000_000_000:
        return TEXT_CONFIG["1_2_b"]
    elif param_nums < 4_000_000_000:
        return TEXT_CONFIG["2_4_b"]
    elif param_nums < 5_000_000_000:
        return TEXT_CONFIG["4_5_b"]
    elif param_nums < 9_000_000_000:
        return TEXT_CONFIG["5_9_b"]
    elif param_nums < 12_000_000_000:
        return TEXT_CONFIG["9_12_b"]
    elif param_nums < 15_000_000_000:  
        return TEXT_CONFIG["12_15_b"]
    elif param_nums < 35_000_000_000:
        return TEXT_CONFIG["15_40_b"]
    elif param_nums < 80_000_000_000:
        return TEXT_CONFIG["40_80_b"]
    else:
        print(f"Model size {param_nums} is not supported")
        return {
            "learning_rate": 4e-5,
            "distributed": distributed,
            "gpu_count": 8,
            "batch_size": 6,
            "use_lora": True
        }


def get_custom_dpo_config(param_nums: int):
    distributed = "ddp"
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            distributed = "ds"

    DPO_CONFIG = {
        "0_1_b": {
            "learning_rate": 1e-5,
            "distributed": "ddp",
            "gpu_count": 1,
            "batch_size": 16,
        },
        "1_2_b": {
            "learning_rate": 1e-5,
            "distributed": "ddp",
            "gpu_count": 1,
            "batch_size": 10,
        },
        "2_4_b": {
            "learning_rate": 1e-5,
            "distributed": "ddp",
            "gpu_count": 1,
            "batch_size": 4,
            "use_lora": True
        },
        "4_5_b": {
            "learning_rate": 1e-5,
            "distributed": "ddp",
            "gpu_count": 2,
            "batch_size": 4,
            "use_lora": True
        },
        "5_9_b": {
            "learning_rate": 1e-5,
            "distributed": "ddp",
            "gpu_count": 2,
            "batch_size": 4,
            "use_lora": True
        },
        "9_12_b": {
            "learning_rate": 8e-6,
            "distributed": distributed,
            "gpu_count": 2,
            "use_lora": True,
            "batch_size": 4,
        },
        "12_15_b": {
            "learning_rate": 8e-6,
            "distributed": distributed,
            "gpu_count": 4,
            "use_lora": True,
            "batch_size": 4,
        },
        "15_40_b": {
            "learning_rate": 8e-6,
            "distributed": distributed,
            "gpu_count": 4,
            "use_lora": True,
            "batch_size": 2,
        },
        "40_80_b": {
            "learning_rate": 8e-6,
            "distributed": distributed,
            "gpu_count": 8,
            "use_lora": True,
            "batch_size": 2,
        }        
    }

    for key in DPO_CONFIG:
        DPO_CONFIG[key]["label"] = key
        
    if param_nums < 1_000_000_000:
        return DPO_CONFIG["0_1_b"]
    elif param_nums < 2_000_000_000:
        return DPO_CONFIG["1_2_b"]
    elif param_nums < 4_000_000_000:
        return DPO_CONFIG["2_4_b"]
    elif param_nums < 5_000_000_000:
        return DPO_CONFIG["4_5_b"]
    elif param_nums < 9_000_000_000:
        return DPO_CONFIG["5_9_b"]
    elif param_nums < 12_000_000_000:
        return DPO_CONFIG["9_12_b"]
    elif param_nums < 15_000_000_000:  
        return DPO_CONFIG["12_15_b"]
    elif param_nums < 35_000_000_000:
        return DPO_CONFIG["15_40_b"]
    elif param_nums < 80_000_000_000:
        return DPO_CONFIG["40_80_b"]
    else:
        print(f"Model size {param_nums} is not supported")
        return {
            "learning_rate": 4e-5,
            "distributed": distributed,
            "gpu_count": 8,
            "batch_size": 6,
            "use_lora": True
        }


def get_custom_grpo_config(param_nums: int):
    distributed = "ddp"
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            distributed = "ds"

    GRPO_CONFIG = {
        "0_1_b": {
            "learning_rate": 0.0002,
            "distributed": "ddp",
            "gpu_count": 1,
            "batch_size": 8,
        },
        "1_2_b": {
            "learning_rate": 0.0002,
            "distributed": "ddp",
            "gpu_count": 1,
            "batch_size": 10,
        },
        "2_4_b": {
            "learning_rate": 0.0002,
            "distributed": "ddp",
            "gpu_count": 1,
            "batch_size": 8,
            "use_lora": True
        },
        "4_5_b": {
            "learning_rate": 0.0002,
            "distributed": "ddp",
            "gpu_count": 2,
            "batch_size": 8,
            "use_lora": True
        },
        "5_9_b": {
            "learning_rate": 0.0002,
            "distributed": "ddp",
            "gpu_count": 2,
            "batch_size": 4,
            "use_lora": True
        },
        "9_12_b": {
            "learning_rate": 0.0002,
            "distributed": "ddp",
            "gpu_count": 2,
            "use_lora": True,
            "batch_size": 4,
        },
        "12_15_b": {
            "learning_rate": 0.0002,
            "distributed": "ddp",
            "gpu_count": 4,
            "use_lora": True,
            "batch_size": 2,
        },
        "15_40_b": {
            "learning_rate": 0.0002,
            "distributed": distributed,
            "gpu_count": 4,
            "use_lora": True,
            "batch_size": 1,
        },
        "40_80_b": {
            "learning_rate": 0.0002,
            "distributed": distributed,
            "gpu_count": 8,
            "use_lora": True,
            "batch_size": 1,
        }        
    }

    for key in GRPO_CONFIG:
        GRPO_CONFIG[key]["label"] = key
        
    if param_nums < 1_000_000_000:
        return GRPO_CONFIG["0_1_b"]
    elif param_nums < 2_000_000_000:
        return GRPO_CONFIG["1_2_b"]
    elif param_nums < 4_000_000_000:
        return GRPO_CONFIG["2_4_b"]
    elif param_nums < 5_000_000_000:
        return GRPO_CONFIG["4_5_b"]
    elif param_nums < 9_000_000_000:
        return GRPO_CONFIG["5_9_b"]
    elif param_nums < 12_000_000_000:
        return GRPO_CONFIG["9_12_b"]
    elif param_nums < 15_000_000_000:  
        return GRPO_CONFIG["12_15_b"]
    elif param_nums < 35_000_000_000:
        return GRPO_CONFIG["15_40_b"]
    elif param_nums < 80_000_000_000:
        return GRPO_CONFIG["40_80_b"]
    else:
        print(f"Model size {param_nums} is not supported")
        return {
            "learning_rate": 4e-5,
            "distributed": distributed,
            "gpu_count": 8,
            "batch_size": 6,
            "use_lora": True
        }


def modify_text_config(config: dict, model: str, model_params: int, model_arch: str):
    base_config = get_custom_text_config(model_params)
    print(f"base_config: {base_config}")

    config["learning_rate"] = base_config["learning_rate"]
    # config["micro_batch_size"] = base_config["batch_size"]

    if base_config.get("use_lora", False):
        config["adapter"] = "lora"
        config["lora_alpha"] = 64
        config["lora_r"] = 32
        config["lora_dropout"] = 0.05
        config["lora_target_linear"] = True
    
    if base_config.get("distributed", "ddp") == "ds":
        config["deepspeed"] = "/workspace/axolotl/zero3.json"

    return config


def modify_dpo_config(config: dict, model: str, model_params: int, model_arch: str):
    base_config = get_custom_dpo_config(model_params)
    print(f"base_config: {base_config}")

    config["learning_rate"] = base_config["learning_rate"]
    config["sample_packing"] = False
    # config["gradient_accumulation_steps"] = 2
    # config["micro_batch_size"] = base_config["batch_size"]

    if base_config.get("use_lora", False):
        config["adapter"] = "lora"
        config["lora_alpha"] = 64
        config["lora_r"] = 32
        config["lora_dropout"] = 0.05
        config["lora_target_linear"] = True
    
    if base_config.get("distributed", "ddp") == "ds":
        config["deepspeed"] = "/workspace/axolotl/zero3.json"

    return config


def modify_grpo_config(config: dict, model: str, model_params: int, model_arch: str):
    base_config = get_custom_grpo_config(model_params)
    print(f"base_config: {base_config}")

    config["learning_rate"] = base_config["learning_rate"]
    config["sample_packing"] = False
    # config["micro_batch_size"] = base_config["batch_size"]

    if base_config.get("use_lora", False):
        config["adapter"] = "lora"
        config["lora_alpha"] = 64
        config["lora_r"] = 32
        config["lora_dropout"] = 0.05
        config["lora_target_linear"] = True
    
    if base_config.get("distributed", "ddp") == "ds":
        config["deepspeed"] = "/workspace/axolotl/zero3.json"

    return config


def modify_model_config(config: dict, model: str, model_params: int, model_arch: str):
    # gradient_checkpointing
    if "falcon-rw" in model.lower():
        config["gradient_checkpointing"] = False


    # use_vllm
    if model in [
        "Eurdem/Defne_llama3_2x8B", 
        "heegyu/WizardVicuna-open-llama-3b-v2", 
        "openlm-research/open_llama_3b", 
        "TitanML/tiny-mixtral", 
        "dunzhang/stella_en_1.5B_v5", 
        "oopsung/llama2-7b-n-ox-test-v1", 
        "microsoft/phi-2", 
        "databricks/dolly-v2-3b"
        ]:
        config["use_vllm"] = False
    if "falcon-rw" in model.lower():
        config["use_vllm"] = False

    if model_arch.strip().lower() in ["gptneoforcausallm", "bloomforcausallm"]:
        config["use_vllm"] = False
    else:
        config["use_vllm"] = True


    # flash_attention
    if model == "microsoft/phi-2":
        config["flash_attention"] = True
    if "falcon-rw" in model.lower():
        config["flash_attention"] = True

    if model_arch.strip().lower() in ["gptneoforcausallm", "bloomforcausallm"]:
        config["flash_attention"] = True
    else:
        config["flash_attention"] = False


    if model_arch.strip().lower() in [
        "qwen2forcausallm",
        "llamaforcausallm",
        "gemma2forcausallm",
        "mixtralforcausallm",
        "mistralforcausallm",
        "qwen3forcausallm",
        "phi3forcausallm",
        "gemmaforcausallm"
        ]:
        config["use_liger"] = True
    else:
        config["use_liger"] = False


    return config


def get_model_architecture(model: str) -> str:
    try:
        config = AutoConfig.from_pretrained(model)
        architectures = config.architectures
        if len(architectures) > 1:
            return "Multiple architectures"
        return architectures[0].strip().lower()
    except:
        return "Unknown"


def get_model_num_params(model: str, model_path: str) -> int:
    MODEL_CONFIG = {
        "facebook/opt-1.3b": {"model_size": 1_300_000_000},
        "facebook/opt-3b": {"model_size": 3_000_000_000},
        "facebook/opt-6.7b": {"model_size": 6_700_000_000},
        "facebook/opt-13b": {"model_size": 13_000_000_000},
        "EleutherAI/gpt-neo-1.3B": {"model_size": 1_300_000_000},
        "EleutherAI/gpt-neo-125m": {"model_size": 125_000_000},
        "bigscience/bloom-560m": {"model_size": 560_000_000},
        "TinyLlama/TinyLlama_v1.1": {"model_size": 1_100_000_000},
    }

    if model in MODEL_CONFIG:
        return MODEL_CONFIG[model_id]["model_size"]
    try:
        hf_api = HfApi()
        model_info = hf_api.model_info(model_path)
        size = model_info.safetensors.total
        return size
    except Exception as e:
        print(f"Error getting model size from safetensors: {e}")
        try:
            import re
            model_size = re.search(r"(\d+)(?=[bB])", model)
            model_size = (
                int(model_size.group(1)) * 1_000_000_000 if model_size else None
            )
            print(f"Model size from regex: {model_size}")
            return model_size
        except Exception as e:
            print(f"Error getting model size from regex: {e}")
            return None


def customize_config(config: dict, task_type: str, model: str, model_path: str, model_params: int):
    # param_nums = get_model_num_params(model, model_path)
    # param_nums_trainable = int(param_nums*0.003)
    # print(f"param_nums: {param_nums}")
    # print(f"param_nums_trainable: {param_nums_trainable}")

    model_arch = get_model_architecture(model)
    print(f"model_arch: {model_arch}") 

    new_config = config
    # print(f"input_config: {new_config}")

    if task_type == TaskType.DPOTASK.value:
        new_config = modify_dpo_config(config, model, model_params, model_arch)
    elif task_type == TaskType.INSTRUCTTEXTTASK.value:
        new_config = modify_text_config(config, model, model_params, model_arch)
    elif task_type == TaskType.GRPOTASK.value:
        new_config = modify_grpo_config(config, model, model_params, model_arch)
    elif task_type == TaskType.CHATTASK.value:
        new_config = modify_text_config(config, model, model_params, model_arch)

    new_config = modify_model_config(config, model, model_params, model_arch)

    # print(f"output_config: {new_config}")

    return new_config


def get_axolotl_base_config_path(dataset_type, level="default") -> str:
    root_dir = pathlib.Path(train_cst.AXOLOTL_DIRECTORIES["root"])
    if isinstance(dataset_type, InstructTextDatasetType):
        return str(root_dir / f"base_text_{level}.yml")
    elif isinstance(dataset_type, DpoDatasetType):
        return str(root_dir / f"base_dpo_{level}.yml")
    elif isinstance(dataset_type, GrpoDatasetType):
        return str(root_dir / f"base_grpo_{level}.yml")
    elif isinstance(dataset_type, ChatTemplateDatasetType):
        return str(root_dir / f"base_chat_{level}.yml")
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset_type)}")


def patch_wandb_symlinks(base_dir:str):
    for root, _, files in os.walk(base_dir):
        for name in files:
            full_path = os.path.join(root, name)

            if os.path.islink(full_path):
                target_path = os.readlink(full_path)

                print(f"Symlink: {full_path} → {target_path}")
                try:
                    os.unlink(full_path)
                except Exception as e:
                    print(f"Failed to unlink {full_path}: {e}")
                    continue

                if os.path.exists(target_path):
                    print("Copying real file")
                    try:
                        shutil.copy(target_path, full_path)
                    except Exception as e:
                        print(f"Failed to copy: {e}")
                else:
                    print("Target not found, creating dummy")
                    pathlib.Path(full_path).touch()


def patch_model_metadata(output_dir: str, base_model_id: str):
    try:
        adapter_config_path = os.path.join(output_dir, "adapter_config.json")

        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                config = json.load(f)

            config["base_model_name_or_path"] = base_model_id

            with open(adapter_config_path, "w") as f:
                json.dump(config, f, indent=2)

            print(f"Updated adapter_config.json with base_model: {base_model_id}", flush=True)
        else:
            print(" adapter_config.json not found", flush=True)

        readme_path = os.path.join(output_dir, "README.md")

        if os.path.exists(readme_path):
            with open(readme_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                if line.strip().startswith("base_model:"):
                    new_lines.append(f"base_model: {base_model_id}\n")
                else:
                    new_lines.append(line)

            with open(readme_path, "w") as f:
                f.writelines(new_lines)

            print(f"Updated README.md with base_model: {base_model_id}", flush=True)
        else:
            print("README.md not found", flush=True)

    except Exception as e:
        print(f"Error updating metadata: {e}", flush=True)
        pass


def copy_dataset_to_axolotl_directories(dataset_path):
    dataset_filename = os.path.basename(dataset_path)
    data_path, root_path = train_paths.get_axolotl_dataset_paths(dataset_filename)
    shutil.copy(dataset_path, data_path)
    shutil.copy(dataset_path, root_path)

    return data_path


def copy_dataset_if_needed(dataset_path, file_format):
    """Copy dataset to Axolotl directories for non-HF datasets."""
    if file_format != FileFormat.HF.value:
        dataset_filename = os.path.basename(dataset_path)

        os.makedirs("/workspace/axolotl/data", exist_ok=True)
        os.makedirs("/workspace/axolotl", exist_ok=True)

        data_path = f"/workspace/axolotl/data/{dataset_filename}"
        root_path = f"/workspace/axolotl/{dataset_filename}"

        shutil.copy(dataset_path, data_path)
        shutil.copy(dataset_path, root_path)

        return data_path
    return dataset_path


def create_config(task_id, model, dataset, dataset_type, file_format, output_dir, addconfig, expected_repo_name=None, log_wandb=True, hours_to_complete=3, is_warmup=True, level="default", batch=32, seq=1024, lrate=0.0002, runtime=10, elaptime=0):
    # time_percent = 0.89
    # time_limit = 15
    time_percent = 0.87
    time_limit = 25

    warmup_percent = 0.10
    warmup_limit = 10
    warmup_step = 5

    """Create the axolotl config file with appropriate settings."""
    config_path = get_axolotl_base_config_path(dataset_type, level)

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config['micro_batch_size'] = int(batch/config['gradient_accumulation_steps'])
    config['sequence_len'] = seq
    config['learning_rate'] = lrate

    config["datasets"] = [create_dataset_entry(dataset, dataset_type, FileFormat(file_format))]
    model_path = str(train_paths.get_text_base_model_path(model))
    config["base_model"] = model_path
    config["mlflow_experiment_name"] = dataset
    os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = str(output_dir)

    if log_wandb:
        config["wandb_runid"] = f"{task_id}_{expected_repo_name}"
        config["wandb_name"] = f"{task_id}_{expected_repo_name}"
        config["wandb_mode"] = "offline"
        os.makedirs(train_cst.WANDB_LOGS_DIR, exist_ok=True)
    else:
        for key in list(config.keys()):
            if key.startswith("wandb"):
                config.pop(key)

    # config = update_flash_attention(config, model)

    if isinstance(dataset_type, InstructTextDatasetType):
        task_type = TaskType.INSTRUCTTEXTTASK.value
    elif isinstance(dataset_type, DpoDatasetType):
        task_type = TaskType.DPOTASK.value
    elif isinstance(dataset_type, GrpoDatasetType):
        task_type = TaskType.GRPOTASK.value
    elif isinstance(dataset_type, ChatTemplateDatasetType):
        task_type = TaskType.CHATTASK.value
    else:
        task_type = TaskType.INSTRUCTTEXTTASK.value

    if file_format == FileFormat.S3.value and task_type == TaskType.DPOTASK.value:
        config = _adapt_columns_for_dpo_dataset(dataset, dataset_type, hours_to_complete, task_id, config, False)
    elif file_format == FileFormat.S3.value and task_type == TaskType.GRPOTASK.value:
        config = _adapt_columns_for_grpo_dataset(dataset, dataset_type, hours_to_complete, task_id, config)
    elif file_format == FileFormat.S3.value and task_type == TaskType.INSTRUCTTEXTTASK.value:
        config = _adapt_columns_for_text_dataset(dataset, dataset_type, hours_to_complete, task_id, config, False)
    elif file_format == FileFormat.S3.value and task_type == TaskType.CHATTASK.value:
        config = _adapt_columns_for_chat_dataset(dataset, dataset_type, hours_to_complete, task_id, config, False)

    if isinstance(dataset_type, DpoDatasetType):
        config["rl"] = "dpo"
    elif isinstance(dataset_type, GrpoDatasetType):
        filename, reward_funcs_names = create_reward_funcs_file(
            [reward_function.reward_func for reward_function in dataset_type.reward_functions],
            task_id,
            destination_dir=train_cst.AXOLOTL_DIRECTORIES["src"],
        )
        config["trl"]["reward_funcs"] = [f"{filename}.{func_name}" for func_name in reward_funcs_names]
        config["trl"]["reward_weights"] = [reward_function.reward_weight for reward_function in dataset_type.reward_functions]

    if file_format != FileFormat.HF.value:
        for ds in config["datasets"]:
            ds["ds_type"] = "json"

            if "path" in ds:
                ds["path"] = train_cst.AXOLOTL_DIRECTORIES["data"]

            ds["data_files"] = [os.path.basename(dataset)]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        config["special_tokens"] = {"pad_token": tokenizer.eos_token}


    print(f"current_config: {config}")


    print(f"Total hours {hours_to_complete}")


    trainable_params = 5000000000
    trainable_params_hour = int(trainable_params/hours_to_complete)

    try:
        model_param_path = os.path.join("/workspace/axolotl", "model_param.json")
        with open(model_param_path, 'r') as f:
            data_models = json.load(f)

            for data in data_models:
                if data['model_name'].lower() == model.lower():
                    print(f"model: {data['model_name']}")

                    trainable_params = int(data['trainable_params'])
                    trainable_params_hour = int(trainable_params/hours_to_complete)
                    all_params = int(data['all_params'])
                    trainable_percent = data['trainable_percent']
                    print(f"trainable_params: {trainable_params}")
                    print(f"trainable_params_hour: {trainable_params_hour}")
                    print(f"all_params: {all_params}")
                    print(f"trainable_percent: {trainable_percent}")

        config = customize_config(config, task_type, model, model_path, all_params)

        # config = get_learning_rate(config, task_type, trainable_params)

    except Exception as e:
        print(f"Error checking and logging base model size: {e}")


    if is_warmup:
        config['max_steps'] = warmup_step
        config['warmup_steps'] = warmup_step

    else:
        max_steps_percent_limit = int((hours_to_complete*60*60*time_percent-(warmup_limit*60))-elaptime)
        max_steps_percent_percent = int((hours_to_complete*60*60*time_percent-(hours_to_complete*60*60*warmup_percent))-elaptime)
        max_steps_limit_limit = int((hours_to_complete*60*60-(time_limit*60)-(warmup_limit*60))-elaptime)
        max_steps_limit_percent = int((hours_to_complete*60*60-(time_limit*60)-(hours_to_complete*60*60*warmup_percent))-elaptime)

        my_warmup = [max_steps_percent_limit, max_steps_percent_percent, max_steps_limit_limit, max_steps_limit_percent]
        my_warmup_min = max(my_warmup)
        config['max_steps'] = int(my_warmup_min/runtime)

        print(f"Final time {format_seconds(my_warmup_min)}")

    print(f"max_steps: {config['max_steps']}")


    # config['max_steps'] = 0
    # config['max_steps'] = 10
    # config['max_steps'] = 20

    print(f"max_steps: {config['max_steps']}")
    

    if config['warmup_steps'] > config['max_steps']:
        config['warmup_steps'] = config['max_steps']


    config.update(addconfig)


    print(f"custom_config: {config}")


    config_path = os.path.join(train_cst.AXOLOTL_DIRECTORIES["configs"], f"{task_id}.yml")
    save_config(config, config_path)
    return config_path


def run_training(task_id, model, dataset, dataset_type, file_format, output_dir, expected_repo_name=None, hours_to_complete=3, log_wandb=True):
    start_time = time.time()

    docker_level = ["default","cache","flash","tuple","float","low"]
    docker_batch = [24,24,20,20,16,16,12,12,8,8,4,4]
    # docker_batch = [8,8,4,4]
    docker_seq = [1024,512,1024,512,1024,512,1024,512,1024,512,1024,512,1024,512,1024,512]
    docker_lrate = 0.0002
    docker_runtime = 10
    docker_config = {}

    if isinstance(dataset_type, GrpoDatasetType):
        docker_level = ["default","cache","flash","tuple","float","low"]
        docker_batch = [16,16,12,12,8,8,4,4]

    docker_failed = True
    idx = 0
    bdx = 0

    # time_percent = 0.89
    # time_limit = 15
    time_percent = 0.87
    time_limit = 25


    try:
        while docker_failed:
            docker_error = ""
            dummy_batch = docker_batch[bdx]
            dummy_batch = dummy_batch - (dummy_batch % 4)

            end_time = time.time()
            elapsed_time = end_time - start_time

            config_path = create_config(
                task_id, 
                model, 
                dataset, 
                dataset_type, 
                file_format, 
                output_dir, 
                docker_config,
                expected_repo_name,
                log_wandb,
                hours_to_complete,
                is_warmup=True,
                level=docker_level[idx],
                batch=dummy_batch,
                seq=docker_seq[bdx],
                lrate=docker_lrate,
                runtime=docker_runtime,
                elaptime=elapsed_time
            )

            try:
                print(f"Docker WARMUP ===============================")


                print(f"Starting training with config: {config_path}", flush=True)
                """Run the training process using the specified config file."""
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)

                print(f"Starting training with level: {docker_level[idx]}", flush=True)
                print(f"Starting training with gradient: {config['gradient_accumulation_steps']}", flush=True)
                print(f"Starting training with batch: {config['micro_batch_size']}", flush=True)
                print(f"Starting training with seq: {config['sequence_len']}", flush=True)
                print(f"Starting training with lrate: {config['learning_rate']}", flush=True)

                training_env = os.environ.copy()
                training_env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
                training_env["HF_HUB_DISABLE_TELEMETRY"] = "1"

                # training_command = [
                # "accelerate", "launch",
                # "-m", "axolotl.cli.train",
                # config_path
                # ]

                # training_command = f"huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential; wandb login $WANDB_TOKEN; accelerate launch -m axolotl.cli.train {config_path}" 

                training_command = f"accelerate launch -m axolotl.cli.train {config_path}" 

                print("Starting training subprocess...\n", flush=True)
                
                process = subprocess.Popen(
                    training_command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )


                filelog = os.path.join("/workspace/axolotl/configs", f"{task_id}.log")
                with open(filelog, "w") as f:
                    for line in process.stdout:
                        f.write(line)
                        f.flush()

                        print(line, end="", flush=True)

                        end_time = time.time()
                        elapsed_time = end_time - start_time

                        if "CUDA out of memory" in line:
                            docker_error = "OutOfMemoryError"
                            sys.exit(docker_error) 
                        elif "Expected to mark a variable ready only once" in line:
                            docker_error = "Variablereadyonlyonce"
                            sys.exit(docker_error) 
                        elif "Caching is incompatible with gradient" in line:
                            docker_error = "Cachingisincompatible"
                            sys.exit(docker_error) 
                        elif "get_max_length" in line:
                            docker_error = "Getmaxlength"
                            sys.exit(docker_error) 
                        elif "mat1 and mat2 must have the same dtype" in line:
                            docker_error = "Musthavethesamedtype"
                            sys.exit(docker_error) 
                        elif "but found Float" in line:
                            docker_error = "ButfoundFloat"
                            sys.exit(docker_error) 
                        elif "tuple index out of range" in line:
                            docker_error = "Tupleindexoutofrange"
                            sys.exit(docker_error) 
                        elif "list index out of range" in line:
                            docker_error = "Listindexoutofrange"
                            sys.exit(docker_error) 
                        elif "DPOTrainer.create_model_card" in line:
                            docker_error = "Dpotrainermodelcard"
                            sys.exit(docker_error) 
                        elif elapsed_time > int(hours_to_complete*60*60*time_percent):
                            docker_error = "Outoftimepercent"
                            sys.exit(docker_error) 
                        elif elapsed_time > int((hours_to_complete*60*60)-(time_limit*60)):
                            docker_error = "Outoftimelimit"
                            sys.exit(docker_error) 


                return_code = process.wait()
                if return_code != 0:
                    if "OutOfMemoryError" in docker_error:
                        raise torch.OutOfMemoryError()
                    else:
                        raise subprocess.CalledProcessError(return_code, training_command)

                print("Training subprocess completed successfully.", flush=True)


                docker_failed = False


            except SystemExit as e:
                if "OutOfMemoryError" in docker_error:
                    print("Training subprocess OutOfMemoryError!", flush=True)
                    if bdx < len(docker_batch):
                        bdx = bdx + 1
                        if bdx >= len(docker_batch):
                            bdx = len(docker_batch)-1
                    if dummy_batch <= 8:
                        idx = idx + 1
                        if idx >= len(docker_level):
                            idx = len(docker_level)-1
                    docker_failed = True
                elif "Variablereadyonlyonce" in docker_error:
                    print("Training subprocess Variablereadyonlyonce!", flush=True)
                    docker_config['gradient_checkpointing']= False
                    docker_failed = True
                elif "Cachingisincompatible" in docker_error:
                    print("Training subprocess Cachingisincompatible!", flush=True)
                    docker_config['gradient_checkpointing']= False
                    docker_failed = True
                elif "Getmaxlength" in docker_error:
                    print("Training subprocess Getmaxlength!", flush=True)
                    idx = idx + 1
                    if idx >= len(docker_level):
                        idx = len(docker_level)-1
                    docker_failed = True
                elif "Musthavethesamedtype" in docker_error:
                    print("Training subprocess Musthavethesamedtype!", flush=True)
                    idx = idx + 1
                    if idx >= len(docker_level):
                        idx = len(docker_level)-1
                    docker_failed = True
                elif "ButfoundFloat" in docker_error:
                    print("Training subprocess ButfoundFloat!", flush=True)
                    idx = idx + 1
                    if idx >= len(docker_level):
                        idx = len(docker_level)-1
                    docker_failed = True
                elif "Tupleindexoutofrange" in docker_error:
                    print("Training subprocess Tupleindexoutofrange!", flush=True)
                    idx = idx + 1
                    if idx >= len(docker_level):
                        idx = len(docker_level)-1
                    docker_failed = True
                elif "Listindexoutofrange" in docker_error:
                    print("Training subprocess Listindexoutofrange!", flush=True)
                    bdx = bdx + 1
                    if bdx >= len(docker_batch):
                        bdx = len(docker_batch)-1
                    idx = idx + 1
                    if idx >= len(docker_level):
                        idx = len(docker_level)-1
                    docker_failed = True
                elif "Dpotrainermodelcard" in docker_error:
                    print("Training subprocess Dpotrainermodelcard!", flush=True)
                    docker_failed = False
                elif "Outoftimepercent" in docker_error:
                    print("Training subprocess Outoftimepercent!", flush=True)
                    docker_failed = False
                elif "Outoftimelimit" in docker_error:
                    print("Training subprocess Outoftimelimit!", flush=True)
                    docker_failed = False


            except subprocess.CalledProcessError as e:
                print("Training subprocess failed!", flush=True)
                print(f"Exit Code: {e.returncode}", flush=True)
                print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)

                print("Training subprocess unknown!", flush=True)
                idx = idx + 1
                if idx >= len(docker_level):
                    idx = len(docker_level)-1
                docker_failed = True

                # raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")


    except Exception as e:
        print(f"Error processing job main: {str(e)}")

    finally:
        print(f"Docker WARMUP finally ===============================")

        try:
            logs = parse_loss_logs(task_id)
            for entry in logs:
                print(f"Entry {entry}")

            best_log = min(logs, key=lambda x: x['loss'])
            # config["learning_rate"] = best_log['learning_rate']
            # save_config(config, config_path)
            # docker_lrate = best_log['learning_rate']

            print(f"Best rate: {best_log['learning_rate']}")

        except Exception as e:
            print(f"Failed to get learning rate: {e}")

        try:
            runtimes = parse_runtime_logs(task_id)
            docker_runtime = runtimes[0]['train_runtime']/config['max_steps']

            print(f"Avg runtime: {docker_runtime}")

        except Exception as e:
            print(f"Failed to get avg runtime: {e}")


        docker_failed = True

        try:
            while docker_failed:
                docker_error = ""
                dummy_batch = docker_batch[bdx]
                dummy_batch = dummy_batch - (dummy_batch % 4)

                end_time = time.time()
                elapsed_time = end_time - start_time

                config_path = create_config(
                    task_id, 
                    model, 
                    dataset, 
                    dataset_type, 
                    file_format, 
                    output_dir, 
                    docker_config,
                    expected_repo_name,
                    log_wandb,
                    hours_to_complete,
                    is_warmup=False,
                    level=docker_level[idx],
                    batch=dummy_batch,
                    seq=docker_seq[bdx],
                    lrate=docker_lrate,
                    runtime=docker_runtime,
                    elaptime=elapsed_time
                )

                try:
                    print(f"Docker TRAINING ===============================")


                    print(f"Starting training with config: {config_path}", flush=True)
                    """Run the training process using the specified config file."""
                    with open(config_path, "r") as f:
                        config = yaml.safe_load(f)

                    print(f"Starting training with level: {docker_level[idx]}", flush=True)
                    print(f"Starting training with gradient: {config['gradient_accumulation_steps']}", flush=True)
                    print(f"Starting training with batch: {config['micro_batch_size']}", flush=True)
                    print(f"Starting training with seq: {config['sequence_len']}", flush=True)
                    print(f"Starting training with lrate: {config['learning_rate']}", flush=True)

                    training_env = os.environ.copy()
                    training_env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
                    training_env["HF_HUB_DISABLE_TELEMETRY"] = "1"

                    # training_command = [
                    # "accelerate", "launch",
                    # "-m", "axolotl.cli.train",
                    # config_path
                    # ]

                    # training_command = f"huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential; wandb login $WANDB_TOKEN; accelerate launch -m axolotl.cli.train {config_path}" 

                    training_command = f"accelerate launch -m axolotl.cli.train {config_path}" 

                    print("Starting training subprocess...\n", flush=True)
                    
                    process = subprocess.Popen(
                        training_command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )

                    # process_log = process.stdout

                    filelog = os.path.join("/workspace/axolotl/configs", f"{task_id}.log")
                    with open(filelog, "w") as f:
                        for line in process.stdout:
                            f.write(line)
                            f.flush()

                            print(line, end="", flush=True)

                            end_time = time.time()
                            elapsed_time = end_time - start_time

                            if "CUDA out of memory" in line:
                                docker_error = "OutOfMemoryError"
                                sys.exit(docker_error) 
                            elif "Expected to mark a variable ready only once" in line:
                                docker_error = "Variablereadyonlyonce"
                                sys.exit(docker_error) 
                            elif "Caching is incompatible with gradient" in line:
                                docker_error = "Cachingisincompatible"
                                sys.exit(docker_error) 
                            elif "get_max_length" in line:
                                docker_error = "Getmaxlength"
                                sys.exit(docker_error) 
                            elif "mat1 and mat2 must have the same dtype" in line:
                                docker_error = "Musthavethesamedtype"
                                sys.exit(docker_error) 
                            elif "but found Float" in line:
                                docker_error = "ButfoundFloat"
                                sys.exit(docker_error) 
                            elif "tuple index out of range" in line:
                                docker_error = "Tupleindexoutofrange"
                                sys.exit(docker_error) 
                            elif "list index out of range" in line:
                                docker_error = "Listindexoutofrange"
                                sys.exit(docker_error) 
                            elif "DPOTrainer.create_model_card" in line:
                                docker_error = "Dpotrainermodelcard"
                                sys.exit(docker_error) 
                            elif elapsed_time > int(hours_to_complete*60*60*time_percent):
                                docker_error = "Outoftimepercent"
                                sys.exit(docker_error) 
                            elif elapsed_time > int((hours_to_complete*60*60)-(time_limit*60)):
                                docker_error = "Outoftimelimit"
                                sys.exit(docker_error) 


                    return_code = process.wait()
                    if return_code != 0:
                        if "OutOfMemoryError" in docker_error:
                            raise torch.OutOfMemoryError()
                        else:
                            raise subprocess.CalledProcessError(return_code, training_command)

                    print("Training subprocess completed successfully.", flush=True)


                    docker_failed = False


                except SystemExit as e:
                    if "OutOfMemoryError" in docker_error:
                        print("Training subprocess OutOfMemoryError!", flush=True)
                        if bdx < len(docker_batch):
                            bdx = bdx + 1
                            if bdx >= len(docker_batch):
                                bdx = len(docker_batch)-1
                        if dummy_batch <= 8:
                            idx = idx + 1
                            if idx >= len(docker_level):
                                idx = len(docker_level)-1
                        docker_failed = True
                    elif "Variablereadyonlyonce" in docker_error:
                        print("Training subprocess Variablereadyonlyonce!", flush=True)
                        docker_config['gradient_checkpointing']= False
                        docker_failed = True
                    elif "Cachingisincompatible" in docker_error:
                        print("Training subprocess Cachingisincompatible!", flush=True)
                        docker_config['gradient_checkpointing']= False
                        docker_failed = True
                    elif "Getmaxlength" in docker_error:
                        print("Training subprocess Getmaxlength!", flush=True)
                        idx = idx + 1
                        if idx >= len(docker_level):
                            idx = len(docker_level)-1
                        docker_failed = True
                    elif "Musthavethesamedtype" in docker_error:
                        print("Training subprocess Musthavethesamedtype!", flush=True)
                        idx = idx + 1
                        if idx >= len(docker_level):
                            idx = len(docker_level)-1
                        docker_failed = True
                    elif "ButfoundFloat" in docker_error:
                        print("Training subprocess ButfoundFloat!", flush=True)
                        idx = idx + 1
                        if idx >= len(docker_level):
                            idx = len(docker_level)-1
                        docker_failed = True
                    elif "Tupleindexoutofrange" in docker_error:
                        print("Training subprocess Tupleindexoutofrange!", flush=True)
                        idx = idx + 1
                        if idx >= len(docker_level):
                            idx = len(docker_level)-1
                        docker_failed = True
                    elif "Listindexoutofrange" in docker_error:
                        print("Training subprocess Listindexoutofrange!", flush=True)
                        bdx = bdx + 1
                        if bdx >= len(docker_batch):
                            bdx = len(docker_batch)-1
                        idx = idx + 1
                        if idx >= len(docker_level):
                            idx = len(docker_level)-1
                        docker_failed = True
                    elif "Dpotrainermodelcard" in docker_error:
                        print("Training subprocess Dpotrainermodelcard!", flush=True)
                        docker_failed = False
                    elif "Outoftimepercent" in docker_error:
                        print("Training subprocess Outoftimepercent!", flush=True)
                        docker_failed = False
                    elif "Outoftimelimit" in docker_error:
                        print("Training subprocess Outoftimelimit!", flush=True)
                        docker_failed = False

                except subprocess.CalledProcessError as e:
                    print("Training subprocess failed!", flush=True)
                    print(f"Exit Code: {e.returncode}", flush=True)
                    print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)

                    print("Training subprocess unknown!", flush=True)
                    idx = idx + 1
                    if idx >= len(docker_level):
                        idx = len(docker_level)-1
                    docker_failed = True

                    # raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")


        except Exception as e:
            print(f"Error processing job main: {str(e)}")

        finally:
            print(f"Docker TRAINING finally ===============================")


async def main():
    print("---STARTING TEXT TRAINING SCRIPT---", flush=True)
    parser = argparse.ArgumentParser(description="Text Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset", required=True, help="Dataset path or HF dataset name")
    parser.add_argument("--dataset-type", required=True, help="JSON string of dataset type config")
    parser.add_argument("--task-type", required=True, choices=["InstructTextTask", "DpoTask", "GrpoTask"], help="Type of task")
    parser.add_argument("--file-format", required=True, choices=["csv", "json", "hf", "s3"], help="File format")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    args = parser.parse_args()

    for directory in train_cst.AXOLOTL_DIRECTORIES.values():
        os.makedirs(directory, exist_ok=True)
    try:
        dataset_type_dict = json.loads(args.dataset_type)

        if args.task_type == TaskType.DPOTASK.value:
            dataset_type = DpoDatasetType(**dataset_type_dict)
        elif args.task_type == TaskType.INSTRUCTTEXTTASK.value:
            dataset_type = InstructTextDatasetType(**dataset_type_dict)
        elif args.task_type == TaskType.GRPOTASK.value:
            dataset_type = GrpoDatasetType(**dataset_type_dict)
        elif args.task_type == TaskType.CHATTASK.value:
            dataset_type = ChatTemplateDatasetType(**dataset_type_dict)
        else:
            sys.exit(f"Unsupported task type: {args.task_type}")
    except Exception as e:
        sys.exit(f"Error creating dataset type object: {e}")

    dataset_path = train_paths.get_text_dataset_path(args.task_id)
    if args.task_type == TaskType.DPOTASK.value:
        adapt_columns_for_dpo_dataset(dataset_path, dataset_type, apply_formatting=True)
    elif args.task_type == TaskType.GRPOTASK.value:
        adapt_columns_for_grpo_dataset(dataset_path, dataset_type)

    dataset_path = copy_dataset_to_axolotl_directories(dataset_path)

    output_dir = train_paths.get_checkpoints_output_path(args.task_id, args.expected_repo_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # config_path = create_config(
    #     args.task_id,
    #     args.model,
    #     dataset_path,
    #     dataset_type,
    #     args.file_format,
    #     output_dir,
    #     args.expected_repo_name,
    #     log_wandb=True
    # )

    # run_training(config_path)

    run_training(
        args.task_id,
        args.model,
        dataset_path,
        dataset_type,
        args.file_format,
        output_dir,
        args.expected_repo_name,
        args.hours_to_complete,
        log_wandb=True
    )    

    patch_model_metadata(output_dir, args.model)

    patch_wandb_symlinks(train_cst.WANDB_LOGS_DIR)


if __name__ == "__main__":
    asyncio.run(main())
