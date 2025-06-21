# config.py
import numpy as np
from benchmark_functions import *
from optimizers import *
from safe_wrappers import safe_function_wrapper, safe_gradient_wrapper, safe_scalar_func


def get_iterations_for_dim(dim):
    """Returns a reasonable number of iterations for a given problem dimension."""
    if dim <= 2: return 200
    if dim <= 5: return 500
    if dim <= 10: return 1000
    return 2000

# === Optimizer Configurations ===
base_optimizer_params = {
    "LineSearchGD": {"class": LineSearchGDOptimizer, "params": {"initial_step_size": 1.0, "momentum_coeff": 0.9, "line_search_shrink": 0.5, "line_search_c": 0.1}},
    "Adaptive": {"class": AdaptiveOptimizer, "params": {"step_size": 0.05, "jump_threshold": 1e-3, "jump_frequency": 50, "jump_scale": 1.0, "anneal_threshold": 0.01, "anneal_factor": 0.95, "momentum_coeff": 0.9, "apply_rotation_to_dirs": True}},
    "SmarterAdaptive": {
        "class": SmarterAdaptiveOptimizer,
        "params": {
            "step_size": 0.05, "jump_threshold": 1e-3, "jump_frequency": 50, "jump_scale": 1.0,
            "anneal_threshold": 0.01, "anneal_factor": 0.95, "momentum_coeff": 0.9,
            "max_directions": 20, "min_directions": 5, "grad_norm_threshold": 5.0
        }
    },
    "EvenSmarterAdaptive": {
        "class": EvenSmarterAdaptiveOptimizer,
        "params": {
            "step_size": 0.05, "jump_threshold": 1e-3, "jump_frequency": 25, "jump_scale": 1.5,
            "anneal_threshold": 0.01, "anneal_factor": 0.95, "momentum_coeff": 0.9,
            "max_directions": 20, "min_directions": 5, "grad_norm_threshold": 5.0,
            "escape_probes": 5, "hessian_eps": 1e-4
        }
    },
    "GD": {"class": StandardGradientDescent, "params": {"learning_rate": 0.001, "anneal_threshold": 1e-3, "anneal_factor": 0.95, "momentum_coeff": 0.9}},
    "Adam": {"class": AdamOptimizer, "params": {"learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "anneal_threshold": 1e-3, "anneal_factor": 0.95}},
    "CMA-ES": {"class": CMAESOptimizer, "params": {"initial_sigma": 1.0}, "iterations": 200},
    "Nevergrad": {"class": NevergradOptimizer, "params": {"nevergrad_optimizer_name": "TwoPointsDE"}, "iterations": None},
    # === Essential Baselines (from Scipy & Nevergrad) ===
    "L-BFGS-B": {
        "class": ScipyOptimizerWrapper,
        "params": {"method": "L-BFGS-B"}
    },
    "Nelder-Mead": {
        "class": ScipyOptimizerWrapper,
        "params": {"method": "Nelder-Mead"}
    },
    "Powell": {
        "class": ScipyOptimizerWrapper,
        "params": {"method": "Powell"}
    },
    "COBYLA": {
        "class": ScipyOptimizerWrapper,
        "params": {"method": "COBYLA"}
    },
    "PSO": { # Particle Swarm Optimization
        "class": NevergradOptimizer,
        "params": {"nevergrad_optimizer_name": "PSO"}
    },
    "DE": { # Differential Evolution
        "class": NevergradOptimizer,
        "params": {"nevergrad_optimizer_name": "DE"}
    },
    # === High-Impact Research Optimizers (from Nevergrad) ===
    "BOBYQA": { # Powell's BOBYQA algorithm
        "class": NevergradOptimizer,
        "params": {"nevergrad_optimizer_name": "BOBYQA"}
    },
    "TwoPointsDE": { # A strong variant of Differential Evolution
        "class": NevergradOptimizer,
        "params": {"nevergrad_optimizer_name": "TwoPointsDE"}
    },
    "NGOpt": { # A good default for Bayesian/model-based optimization
        "class": NevergradOptimizer,
        "params": {"nevergrad_optimizer_name": "NGOpt"}
    },

}


# === Function Configurations ===
function_properties = {

    # === 🎯 Standard Benchmark Functions ===
    "Sphere": {
        "func": sphere_nd,
        "grad": grad_sphere_nd,
        "start_range": [-5.12, 5.12],
        "min_pos": 0.0,
        "min_val": 0.0
    },
    "Rastrigin": {
        "func": rastrigin_nd,
        "grad": grad_rastrigin_nd,
        "start_range": [-5.12, 5.12],
        "min_pos": 0.0,
        "min_val": 0.0
    },
    "Rosenbrock": {
        "func": rosenbrock_nd,
        "grad": grad_rosenbrock_nd,
        "start_range": [-2.0, 2.0],
        "min_pos": 1.0,
        "min_val": 0.0
    },

    # === 🌐 Complex / Multi-modal & Deceptive ===
    "Ackley": {
        "func": safe_function_wrapper(ackley_nd),
        "grad": safe_gradient_wrapper(grad_ackley_nd),
        "start_range": [-32.768, 32.768],
        "min_pos": 0.0,
        "min_val": 0.0
    },
    "Griewank": {
        "func": safe_function_wrapper(griewank_nd),
        "grad": safe_gradient_wrapper(grad_griewank_nd),
        "start_range": [-600, 600],
        "min_pos": 0.0,
        "min_val": 0.0
    },
    "Schwefel": {
        "func": safe_function_wrapper(schwefel_nd),
        "grad": safe_gradient_wrapper(grad_schwefel_nd),
        "start_range": [-500, 500],
        "min_pos": 420.9687,
        "min_val": 0.0
    },
    "Michalewicz": {
        "func": michalewicz_nd,
        "grad": grad_michalewicz_nd,
        "start_range": [0, np.pi],
        "min_pos": None,
        "min_val": -1.801  # for 2D
    },
    "DropWave": {
        "func": safe_function_wrapper(drop_wave_nd),
        "grad": safe_gradient_wrapper(grad_drop_wave_nd),
        "start_range": [-5.12, 5.12],
        "min_pos": 0.0,
        "min_val": -1.0
    },
    "LunacekBiRastrigin": {
        "func": safe_function_wrapper(lunacek_bi_rastrigin_nd),
        "grad": None,
        "start_range": [-5.12, 5.12],
        "min_pos": None,
        "min_val": None
    },

    # === 🪪 Shifted or Transformed Functions ===
    "ShiftedSphere": {
        "func": safe_function_wrapper(shifted_sphere_nd),
        "grad": safe_gradient_wrapper(grad_shifted_sphere_nd),
        "start_range": [0.0, 6.0],
        "min_pos": 3.0,
        "min_val": 0.0
    },
    "ShiftedRastrigin": {
        "func": safe_function_wrapper(shifted_rastrigin_nd),
        "grad": safe_gradient_wrapper(grad_shifted_rastrigin_nd),
        "start_range": [0.0, 6.0],
        "min_pos": 3.0,
        "min_val": 0.0
    },
    "HybridSphereRastrigin": {
        "func": hybrid_sphere_rastrigin_nd,
        "grad": grad_hybrid_sphere_rastrigin_nd,
        "start_range": [-5.12, 5.12],
        "min_pos": 0.0,
        "min_val": 0.0
    },

    # === 🧮 Ill-Conditioned & Pathological Functions ===
    "Ellipsoid": {
        "func": ellipsoid_nd,
        "grad": grad_ellipsoid_nd,
        "start_range": [-5.12, 5.12],
        "min_pos": 0.0,
        "min_val": 0.0
    },
    "Discus": {
        "func": discus_nd,
        "grad": grad_discus_nd,
        "start_range": [-5.12, 5.12],
        "min_pos": 0.0,
        "min_val": 0.0
    },
    "Zakharov": {
        "func": zakharov_nd,
        "grad": grad_zakharov_nd,
        "start_range": [-5, 10],
        "min_pos": 0.0,
        "min_val": 0.0
    },
    "StyblinskiTang": {
        "func": styblinski_tang_nd,
        "grad": grad_styblinski_tang_nd,
        "start_range": [-5, 5],
        "min_pos": -2.903534,
        "min_val_coeff": -39.166166
    },

    # === 📉 Non-Smooth & Stochastic ===
    "Step": {
        "func": step_nd,
        "grad": None,
        "start_range": [-5.12, 5.12],
        "min_pos": 0.0,
        "min_val": 0.0
    },
    "QuarticNoise": {
        "func": quartic_with_noise_nd,
        "grad": grad_quartic_with_noise_nd,
        "start_range": [-1.28, 1.28],
        "min_pos": 0.0,
        "min_val": 0.0
    },

    # === 🧭 Special Case: 2D Realistic Landscape ===
    "Branin2D": {
        "func": branin_2d,
        "grad": grad_branin_2d,
        "start_range": [-5, 10],
        "min_pos": None,
        "min_val": 0.397887
    }
}


def get_benchmark_configs():
    """
    This function programmatically builds the final benchmark configuration
    dictionary that the main script will loop over.
    """
    benchmark_configs = {}
    dimensions_to_test = [2, 5, 10, 20]

    for func_name, props in function_properties.items():
        for dim in dimensions_to_test:
            key = f"{func_name}_{dim}D"

            # Handle transformed function generation
            if func_name == "AffineTransformedRosenbrock":
                obj_func, grad_func = props["func"](dim)
            else:
                obj_func, grad_func = props["func"], props["grad"]

            # Handle dimensional scaling for some minima
            min_val = props.get("min_val_coeff", props.get("min_val", 0.0))
            if "min_val_coeff" in props:
                min_val *= dim

            # Store common properties first
            benchmark_configs[key] = {
                "func": obj_func,  # Will be overridden per optimizer if needed
                "grad": grad_func,
                "dim": dim,
                "true_min_pos": np.full(dim, props["min_pos"]),
                "true_min_val": min_val,
                "start_range": props["start_range"],
                "optimizers": {}
            }

            for opt_name, opt_conf in base_optimizer_params.items():
                cfg = opt_conf["params"].copy()
                iters_val = opt_conf.get("iterations") or get_iterations_for_dim(dim)

                # Adjust config for Nevergrad optimizers
                if opt_conf["class"] == NevergradOptimizer:
                    cfg['dim'] = dim
                    cfg['budget'] = iters_val
                    cfg['lower'], cfg['upper'] = props["start_range"]

                # Adjust config for Scipy optimizers
                if opt_conf["class"] == ScipyOptimizerWrapper:
                    padding = 1e-6  # Prevent boundary issues
                    lower, upper = props["start_range"]
                    cfg["bounds"] = [(lower + padding, upper - padding)] * dim

                # Apply safe_scalar_func if needed
                obj_func_wrapped = (
                    safe_scalar_func(obj_func, label=f"{key}_{opt_name}")
                    if opt_conf["class"] == ScipyOptimizerWrapper else obj_func
                )

                # Register optimizer entry
                benchmark_configs[key]["optimizers"][opt_name] = {
                    "class": opt_conf["class"],
                    "params": cfg,
                    "iterations": iters_val,
                    "func": obj_func_wrapped  # Used by run_benchmark
                }

    return benchmark_configs
