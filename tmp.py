from typing import Any
from debug_code.utils import squeeze
from utils.args import parse_args
from utils.checkpoints import CheckpointSaver
from datasets.base.builder import build_dataset, build_loader
from models.base.builder import build_model
import torch
import numpy as np
import utils.logging as logging
from tqdm import tqdm
import torch.nn.functional as F
from utils.utils import (
    accuracy,
    get_current_time_str,
    load_checkpoint,
    log_test_result,
    save_checkpoint,
)
import wandb
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
from test import inductive_evaluate
from functools import partial
import sys

logger = logging.get_logger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def compute_accuracy(cfg, input, output):
    if cfg.TRAIN.type == 'iter':
        return accuracy(output['logits'], input['labels'])
    elif cfg.TRAIN.type == 'meta':
        acc = accuracy(output["logits"], input["target_labels"])
        return acc

def compute_loss(cfg, input, output):
    if cfg.TRAIN.type == 'iter':
        loss = F.cross_entropy(output['logits'], input['labels'])
        return loss
    
    elif cfg.TRAIN.type == 'meta':
        loss = F.cross_entropy(output["logits"], input["target_labels"].long())
        if hasattr(cfg, "USE_JDOT") and cfg.USE_JDOT:
            loss += cfg.JDOT_ALPHA * output["jdot_loss"]

            classification_labels = torch.concatenate(
                [input["real_support_labels"], input["real_target_labels"]], dim=0
            )
            classificaion_loss = F.cross_entropy(
                output["class_text_logits"], classification_labels.long()
            )
            loss += cfg.CLS_ALPHA * classificaion_loss
        if "loss" in output:
            loss += output["loss"]
        return loss

def remove_file(file_path):
    if file_path != None and os.path.exists(file_path):
        os.remove(file_path)

def train_epoch(
    cfg,
    train_loader,
    val_loader,
    model,
    optimizer,
    scheduler,
    start_idx=0
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        model_ema (model): the ema model to update.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (Config): The global config object.
    """
    # Enable train mode.
    model.to(DEVICE)
    model.train()

    if cfg.TRAIN.type == 'iter':
        num_iter = cfg.TRAIN.NUM_TRAIN_ITER    
    elif cfg.TRAIN.type == 'meta':
        num_iter = cfg.TRAIN.NUM_TRAIN_TASKS
    loader_iter = iter(train_loader)


    running_acc = 0
    running_loss = 0

    log_interval = cfg.STEP_PER_LOG
    val_interval = cfg.TRAIN.VAL_FRE_ITER
    assert val_interval % log_interval == 0
    gradient_accumulation_steps = cfg.GRADIENT_ACCUMULATION_STEPS
    
    last_checkpoint_path = None
    keep_num_best = cfg.num_best_checkpoint if hasattr(cfg, "num_best_checkpoint") else 3
    ckpt_saver = CheckpointSaver(keep_num_best=keep_num_best)
    
    for cur_iter in tqdm(range(start_idx, num_iter)):
        wandb_log = {}
        try:
            input = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            input = next(loader_iter)
        
        input = squeeze(input, DEVICE)
        
        input["split"] = "train"
        
        #with profiler.profile(with_stack=True, profile_memory=True) as prof:
        output = model(input)
            
        loss = compute_loss(cfg, input, output)
        loss.backward()

        acc = compute_accuracy(cfg, input, output)
        
        running_acc += acc.item()
        running_loss += loss.item()
        
        del loss

        if (cur_iter + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        
        # evaluate
        if (cur_iter + 1) % val_interval == 0:
            with torch.no_grad():
                res = inductive_evaluate(
                    cfg, model, val_loader, cfg.TRAIN.NUM_VAL_TASKS, "val"
                )
                wandb_log.update({"val_acc": res["acc"], "val_loss": res["loss"]})
               
                stat = {"iter": cur_iter}
                ckpt_saver(
                    res["acc"],
                    cfg.OUTPUT_DIR + f"/best_{cur_iter+1:05}_acc{res['acc']:.3f}.pt",
                    partial(
                        save_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        stat=stat,
                    ),
                )

                remove_file(last_checkpoint_path)
                last_checkpoint_path = (
                    cfg.OUTPUT_DIR + f"/last_{cur_iter+1:05}_acc{res['acc']:.3f}.pt"
                )
                save_checkpoint(
                    model, optimizer, scheduler, stat=stat, save_path=last_checkpoint_path
                )
            model.train()

        if (cur_iter + 1) % log_interval == 0:
            wandb_log.update(
                {
                    f"train_loss_{log_interval}": running_loss / log_interval,
                    f"train_acc_{log_interval}": running_acc / log_interval,
                }
            )
            running_loss = 0
            running_acc = 0
        
        lr = optimizer.param_groups[0]["lr"]
        wandb_log["lr"] = lr
        wandb.log(wandb_log)

    return {"best_checkpoint": ckpt_saver.get_best_checkpoint_path()}


def init_session(cfg, args):
    seed(cfg.RANDOM_SEED)

    if not args.test_only:
        wandb.login(key="fa9099de08cfa896a63091aff05becd9345b786c")
        run_group = None
        run_name = None
        if hasattr(cfg, "wandb"):
            run_group = cfg.wandb.group
            run_name = cfg.wandb.name
        wandb.init(
            entity="aiotlab", project="few-shot-action-recognition", group=run_group, name=run_name
        )
        # debug mode
        if (
            os.environ.get("WANDB_MODE") is not None
            and os.environ.get("WANDB_MODE") == "disabled"
        ):
            pass

        else:
            output_dir = cfg.OUTPUT_DIR
            output_dir = os.path.join(
                output_dir, f"{get_current_time_str()}_w{wandb.run.id}"
            )
            os.makedirs(output_dir, exist_ok=True)
            cfg.OUTPUT_DIR = output_dir
            
            file_name = "config.py"
            save_path = os.path.join(output_dir, file_name)
            print("Save config into: ", save_path)
            cfg.dump(save_path)

def build_optimizer(model, cfg):
    optimizer = torch.optim.AdamW(model.parameters(), cfg.BASE_LR)
    return optimizer

def build_scheduler(optimizer, optimizer_cfg, scheduler_cfg):
    min_lr = scheduler_cfg.min_lr if hasattr(scheduler_cfg, "min_lr") else optimizer_cfg.BASE_LR/100
    T_0 = scheduler_cfg.T_0 if hasattr(scheduler_cfg, "T_0") else 1000
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0, eta_min=min_lr
    )
    return scheduler

# class Optimizer:
#     def __init__(self, model, general_cfg, optimizer_cfg, scheduler_cfg=None):
#         self.optimizer = build_optimizer(model, optimizer_cfg)
#         self.scheduler = None
#         if scheduler_cfg is not None:
#             self.scheduler = CosineAnnealingWarmRestarts(
#                 self.optimizer, **scheduler_cfg
#             )
    
#     def step(self):
#         self.optimizer.step()
#         self.optimizer.zero_grad()
    
#     def zero_grad(self):
#         pass
    
#     def state_dict(self):
#         pass
        

def train():
    args = parse_args()
    cfg = args.cfg
    init_session(cfg, args)
    
    # dataset_name = cfg.TRAIN.DATASET
    # dataset = build_dataset(dataset_name, cfg, "train")
    
    checkpoint_path = None
    if not args.test_only:
        
        # dataloader
        train_loader = build_loader(cfg, "train")
        val_loader = build_loader(cfg, "val")

        # model
        start_train_iter = 0
        model, model_ema = build_model(cfg)
        
        # optimizer
        optimizer = build_optimizer(model, cfg.OPTIMIZER)
        
        #scheduler
        cfg.SCHEDULER = cfg.SCHEDULER if hasattr(cfg, "SCHEDULER") else {}
        scheduler = build_scheduler(optimizer, cfg.OPTIMIZER, cfg.SCHEDULER)
        
        if args.checkpoint_loadpath is not None:
            stats = load_checkpoint(args.checkpoint_loadpath, model, optimizer, scheduler)
            start_train_iter = stats["iter"] + 1

        res = train_epoch(cfg, train_loader, val_loader, model, optimizer, scheduler, start_train_iter)

        load_checkpoint(res["best_checkpoint"], model)
        checkpoint_path = res['best_checkpoint']
    
    else:
        model, model_ema = build_model(cfg)

        try:
            if args.checkpoint_loadpath is not None:
                load_checkpoint(args.checkpoint_loadpath, model)
                checkpoint_path = args.checkpoint_loadpath
        except:
            print("load checkpoint failed")

    test_loader = build_loader(cfg, "test")

    res = inductive_evaluate(cfg, model, test_loader, cfg.TRAIN.NUM_TEST_TASKS)

    print("\n")
    print("Shell command:", ' '.join(sys.argv))
    print("Checkpoint: ", checkpoint_path)
    print(
        f"Test accuracy: {res['acc']}. Confidence_interval: {res['acc_confidence_interval']}"
    )
    print("\n")
    log_test_result(cfg, checkpoint_path, res['acc'])
    
    wandb.finish()
    

if __name__ == "__main__":
    train()
