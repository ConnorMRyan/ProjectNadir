# run_benchmark.py
import time
import os
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from benchmark_config import get_benchmark_configs
from reporting import ConsoleLogger
from safe_wrappers import safe_scalar_func


def run_single_trial(config):
    """Runs a single optimization trial, with overflow safety tracking."""
    func_name_dim, dim, opt_name, trial_num = config['func_name_dim'], config['dim'], config['opt_name'], config['trial_num']
    obj_func, grad_func, true_min_val, start_range = config['obj_func'], config['grad_func'], config['true_min_val'], config['start_range']
    opt_class, opt_params, iterations = config['opt_class'], config['opt_params'], config['iterations']

    start_pos = np.random.uniform(start_range[0], start_range[1], size=dim)

    try:
        opt = opt_class(obj_func=obj_func, grad_func=grad_func, **opt_params)
        start_time = time.time()
        history, _ = opt.optimize(start_pos, iterations)
        runtime = time.time() - start_time

        final_pos = history[-1][0] if isinstance(history[-1], tuple) else history[-1]
        val_result = obj_func(final_pos)

        # Support legacy and new safe_scalar_func interfaces
        if isinstance(val_result, tuple):
            final_val, was_safe = val_result
        else:
            final_val = val_result
            was_safe = np.isfinite(final_val) and abs(final_val) < 1e10

        error = final_val - true_min_val if was_safe else "FAIL"

        return {
            "function_name": func_name_dim,
            "dim": dim,
            "optimizer": opt_name,
            "trial": trial_num,
            "final_val": final_val,
            "true_min_val": true_min_val,
            "error": error,
            "runtime_s": runtime,
            "iterations": iterations,
            "status": "PASS" if was_safe else "FAIL"
        }

    except Exception as e:
        return {
            "function_name": func_name_dim,
            "dim": dim,
            "optimizer": opt_name,
            "trial": trial_num,
            "final_val": "ERROR",
            "true_min_val": "ERROR",
            "error": str(e),
            "runtime_s": 0,
            "iterations": iterations,
            "status": "FAIL"
        }


def run_benchmark(optimizers_to_run=None, num_starts=10, num_workers=-1, output_filename=None):
    """Orchestrates the full benchmark run based on provided arguments."""
    logger = ConsoleLogger()
    logger.header("Benchmark Initializing")

    start_time = logger.start_timer("Preparing benchmark trials")
    trial_configs = []
    benchmark_configs = get_benchmark_configs()

    if optimizers_to_run:
        logger.info(f"Filtering to run only: {', '.join(optimizers_to_run)}")

    for func_name_dim, config in benchmark_configs.items():
        all_optimizers = config['optimizers']
        active_optimizers = {k: v for k, v in all_optimizers.items() if not optimizers_to_run or k in optimizers_to_run}

        for opt_name, opt_data in active_optimizers.items():
            if 'grad_func' in opt_data['class'].__init__.__code__.co_varnames and config['grad'] is None:
                continue

            for i in range(1, num_starts + 1):
                trial_configs.append({
                    "func_name_dim": func_name_dim,
                    "dim": config['dim'],
                    "opt_name": opt_name,
                    "trial_num": i,
                    "obj_func": opt_data.get("func", config["func"]),
                    "grad_func": config['grad'],
                    "true_min_val": config['true_min_val'],
                    "start_range": config['start_range'],
                    "opt_class": opt_data['class'],
                    "opt_params": opt_data['params'],
                    "iterations": opt_data.get('iterations')
                })

    logger.end_timer(start_time)

    if not trial_configs:
        logger.info("No trials to run based on the provided filters. Exiting.")
        return

    logger.info(f"Total trials to run: {len(trial_configs)}")

    logger.header("Running Benchmark")
    logger.info(f"Using {num_workers if num_workers > 0 else os.cpu_count()} CPU cores...")

    results = Parallel(n_jobs=num_workers)(
        delayed(run_single_trial)(cfg) for cfg in tqdm(trial_configs, desc="Overall Progress")
    )

    logger.header("Processing Results")
    start_time = logger.start_timer("Aggregating and saving results")

    results_df = pd.DataFrame([res for res in results if res is not None])

    results_dir = "benchmark_results"
    os.makedirs(results_dir, exist_ok=True)

    if output_filename is None:
        output_filename = f"benchmark_results_{int(time.time())}.csv"

    results_path = os.path.join(results_dir, output_filename)
    results_df.to_csv(results_path, index=False)

    logger.end_timer(start_time)
    logger.success(f"Results successfully saved to {results_path}")
    logger.header("Benchmark Complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the optimizer benchmark suite.")
    parser.add_argument('--optimizer', nargs='+', help='Run only the specified optimizer(s).')
    parser.add_argument('--num-starts', type=int, default=10, help='The number of independent trials.')
    parser.add_argument('--num-workers', type=int, default=-1, help='The number of CPU cores to use.')
    parser.add_argument('--output', type=str, default=None, help='Specify a custom name for the output CSV file.')

    args = parser.parse_args()

    run_benchmark(
        optimizers_to_run=args.optimizer,
        num_starts=args.num_starts,
        num_workers=args.num_workers,
        output_filename=args.output
    )
