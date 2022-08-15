import argparse
from datetime import datetime

import torch
from torch.quasirandom import SobolEngine
import wandb
from tqdm import tqdm

from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.generation import MaxPosteriorSampling
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.test_functions import Ackley, Branin, Cosine8, Hartmann, Rosenbrock
from gpytorch.mlls import ExactMarginalLogLikelihood

from utils import wandb_init, wandb_log, wandb_finish, get_initial_points, eval_objective

def set_function(func_name, dtype, device):
    if func_name == "Ackley":
        return Ackley(negate=True).to(dtype=dtype, device=device)
    elif func_name == "Branin":
        return Branin(negate=True).to(dtype=dtype, device=device)
    elif func_name == "Cosine8":
        return Cosine8(negate=True).to(dtype=dtype, device=device)
    elif func_name == "Hartmann":
        return Hartmann(negate=True).to(dtype=dtype, device=device)
    elif func_name == "Rosenbrock":
        return Rosenbrock(negate=True).to(dtype=dtype, device=device)
    else:
        raise NotImplementedError()

def set_model(model_name, X_init, Y_init):
    if model_name == "GP":
        model = SingleTaskGP(X_init, Y_init)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        return model
    else:
        raise NotImplementedError()

def set_acqf(acqf_name, model, best_f):
    if acqf_name == "UCB":
        return UpperConfidenceBound(model, beta=0.2)
    elif acqf_name == "EI":
        return ExpectedImprovement(model, best_f=best_f)
    elif acqf_name == "TS":
        return MaxPosteriorSampling(model, replacement=False)
    else:
        raise NotImplementedError()

def select_x(acqf, bounds, q, num_restarts, raw_samples, dtype, device):
    if isinstance(acqf, (UpperConfidenceBound, ExpectedImprovement)):
        X_next, _ = optimize_acqf(acqf, bounds, q=q, num_restarts=num_restarts, raw_samples=raw_samples)
    if isinstance(acqf, MaxPosteriorSampling):
        n_candidates = 1000
        sobol = SobolEngine(bounds.shape[-1], scramble=True)
        X_cand = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        X_next = acqf(X_cand, num_samples=q)
    return X_next

def evaluate(func_name, model_name, acqf_name, num_trials, seed, device="cpu", dtype="double"):
    exp_name = datetime.now().strftime("%y/%m/%d-%H:%M")
    wandb_init(exp_name, func_name, model_name, acqf_name, num_trials, seed)

    # hyperparameters (subject to change)
    q = 1 # number of candidates
    num_restarts = 10 # search best input repeatedly
    raw_samples = 512 # initial points per optimization

    function = set_function(func_name, dtype=dtype, device=device)
    bounds = torch.stack([torch.zeros(function.dim), torch.ones(function.dim)]).to(dtype=dtype, device=device)
    X_train, Y_train = get_initial_points(function, n_init=10, dtype=dtype, device=device)

    Y_best = Y_train.max().item()
    for trial in tqdm(range(num_trials)):
        model = set_model(model_name, X_train, Y_train)
        acqf = set_acqf(acqf_name, model, Y_best)
        X_new = select_x(acqf, bounds, q, num_restarts, raw_samples, dtype=dtype, device=device)
        Y_new = eval_objective(function, X_new)

        # Add new data points to buffer
        X_train = torch.cat([X_train, X_new], dim=0)
        Y_train = torch.cat([Y_train, Y_new], dim=0)

        # Update best value so far
        Y_new = Y_new.item()
        if Y_new > Y_best:
            Y_best = Y_new
        
        # Logging
        wandb_log(trial, Y_new, Y_best)
    wandb_finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--func_name",
                        choices=["Ackley", "Branin", "Cosine8", "Rosenbrock"], help="function for evaluation")
    parser.add_argument("--model_name",
                        choices=["GP"], help="choose surrogate model")
    parser.add_argument("--acqf_name",
                        choices=["UCB", "EI", "TS"], help="choose acquisition function")
    parser.add_argument("--num_trials",
                        type=int, default=100, help="number of trials for adaptation")
    parser.add_argument("--seed",
                        type=int, default=42, help="for reproduction")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    evaluate(args.func_name, args.model_name, args.acqf_name, args.num_trials, args.seed, device, dtype)