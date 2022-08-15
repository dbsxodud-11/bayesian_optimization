
import torch
import wandb
from botorch.utils.transforms import unnormalize

def wandb_init(exp_name, func_name, model_name, acqf_name, num_trials, seed):
    wandb.init(project="Bayesian Optimization", name=exp_name,
               config={"func_name": func_name,
                       "model_name": model_name,
                       "acqf_name": acqf_name,
                       "num_trials": num_trials,
                       "seed": seed})

def wandb_log(trial, curr_y, best_y):
    wandb.log({"Trial": trial, "Curr Value": curr_y, "Best Value": best_y})

def wandb_finish():
    wandb.finish()

def eval_objective(func, x):
    """This is a helper function we use to unnormalize and evalaute a point"""
    return func(unnormalize(x, func.bounds)).unsqueeze(-1)

def get_initial_points(func, n_init, dtype, device):
    """Get initial points"""
    X_init = torch.rand(size=(n_init, func.dim)).to(dtype=dtype, device=device)
    Y_init = eval_objective(func, X_init)
    return X_init, Y_init