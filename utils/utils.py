from datetime import datetime, timedelta
import os
import torch
import numpy as np
def get_current_time_str():
    utc_now = datetime.utcnow()

    # Calculate the Vietnam local time (ICT, UTC+7)
    vietnam_time = utc_now + timedelta(hours=7)

    return vietnam_time.strftime("%d-%m_%H-%M")

def save_checkpoint(save_path, model, optimizer, iter, scheduler=None, metadata=None):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    
    state['metadata'] = {"iter": iter}
    
    if metadata is not None:
        state["metadata"].update(metadata)
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)

def load_checkpoint(load_path, model, optimizer=None, scheduler=None, return_metadata=True):
    ckpt = torch.load(load_path)
    
    model.load_state_dict(ckpt['model'])
    
    if optimizer != None and hasattr(ckpt, "optimizer"):
        optimizer.load_state_dict(ckpt['optimizer'])
    
    if scheduler != None and hasattr(ckpt, "scheduler"):
        scheduler.load_state_dict(ckpt['scheduler'])

    if return_metadata and hasattr(ckpt, 'metadata'):
        return ckpt['metadata']

class CheckpointSaver:
    def __init__(self, keep_num_best=1):
        self.keep_num_best = keep_num_best
        self.best_checkpoints = []

    def __call__(self, acc, save_path):
        if (
            len(self.best_checkpoints) < self.keep_num_best
            or acc > self.best_checkpoints[-1][0]
        ):
            self.best_checkpoints.append((acc, save_path))
            self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)
            if len(self.best_checkpoints) > self.keep_num_best:
                try:
                    os.remove(self.best_checkpoints[-1][1])
                except:
                    pass
                # remove_file(self.best_checkpoints[-1][1])
                self.best_checkpoints.pop()
            return True
        return False

    def get_best_checkpoint_path(self):
        return self.best_checkpoints[0][1]
    
    def get_best_acc(self):
        return self.best_checkpoints[0][0]
    
def remove_file(file_path):
    try:
        os.remove(file_path)
    except:
        pass

def log_test_result(cfg, checkpoint_path, test_acc):
    dataset = cfg.DATA.DATA_ROOT_DIR.split('/')[1]
    
    method = get_method(cfg)
        
    res = [dataset, cfg.TRAIN.SHOT, method, checkpoint_path, test_acc]
    with open("test_result.csv", "a") as f:
        f.write(",".join([str(i) for i in res]) + "\n")

def get_method(cfg):
    if hasattr(cfg, "TEMPO_PRIOR"):
        method = f"tpm{cfg.TEMPO_PRIOR:0.1f}"
    else:
        method = "sdtw" 
    return method
      
def calculate_temporal_prior(num_segments, sigma=1.0):
    T = (
        (torch.arange(num_segments).view(num_segments, 1) - torch.arange(num_segments).view(1, num_segments))
        / float(num_segments)
        / np.sqrt(2 / 64)
    )
    T = -(T**2) / 2.0 / (sigma**2)
    T = 1 / (sigma * np.sqrt(2 * np.pi)) * torch.exp(T)
    T = T / T.sum(1, keepdim=True)
    return T