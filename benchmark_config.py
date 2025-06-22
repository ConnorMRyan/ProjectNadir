import numpy as np
from benchmark_functions import *
from optimizers import *
from safe_wrappers import safe_function_wrapper, safe_gradient_wrapper, safe_scalar_func

def get_iterations_for_dim(dim):
    if dim <= 2: return 200
    if dim <= 5: return 500
    if dim <= 10: return 1000
    return 2000

# === Optimizer Configurations ===
base_optimizer_params = {
    "LineSearchGD": {
        "class": LineSearchGDOptimizer,
        "params": {"initial_step_size": 1.0, "momentum_coeff": 0.9, "line_search_shrink": 0.5, "line_search_c": 0.1}
    },
    "Adaptive": {
        "class": AdaptiveOptimizer,
        "params": {
            "step_size": 0.05, "jump_threshold": 1e-3, "jump_frequency": 50, "jump_scale": 1.0,
            "anneal_threshold": 0.01, "anneal_factor": 0.95, "momentum_coeff": 0.9, "apply_rotation_to_dirs": True
        }
    },
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
    "GD": {
        "class": StandardGradientDescent,
        "params": {"learning_rate": 0.001, "anneal_threshold": 1e-3, "anneal_factor": 0.95, "momentum_coeff": 0.9}
    },
    "Adam": {
        "class": AdamOptimizer,
        "params": {"learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8,
                   "anneal_threshold": 1e-3, "anneal_factor": 0.95}
    },
    "CMA-ES": {
        "class": CMAESOptimizer,
        "params": {"initial_sigma": 1.0},
        "iterations": 200
    },
    "Nevergrad": {
        "class": NevergradOptimizer,
        "params": {"optimizer_name": "TwoPointsDE"},
        "iterations": None
    },
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
    "PSO": {
        "class": NevergradOptimizer,
        "params": {"optimizer_name": "PSO"}
    },
    "DE": {
        "class": NevergradOptimizer,
        "params": {"optimizer_name": "DE"}
    },
    "BOBYQA": {
        "class": NevergradOptimizer,
        "params": {"optimizer_name": "BOBYQA"}
    },
    "TwoPointsDE": {
        "class": NevergradOptimizer,
        "params": {"optimizer_name": "TwoPointsDE"}
    },
    "NGOpt": {
        "class": NevergradOptimizer,
        "params": {"optimizer_name": "NGOpt"}
    },
}

# === Function Configurations ===
function_properties = {
    "Sphere": {"func": sphere_nd, "grad": grad_sphere_nd, "start_range": [-5.12, 5.12], "min_pos": 0.0, "min_val": 0.0},
    "Rastrigin": {"func": rastrigin_nd, "grad": grad_rastrigin_nd, "start_range": [-5.12, 5.12], "min_pos": 0.0, "min_val": 0.0},
    "Rosenbrock": {"func": rosenbrock_nd, "grad": grad_rosenbrock_nd, "start_range": [-2.0, 2.0], "min_pos": 1.0, "min_val": 0.0},
    "Ackley": {"func": safe_function_wrapper(ackley_nd), "grad": safe_gradient_wrapper(grad_ackley_nd), "start_range": [-32.768, 32.768], "min_pos": 0.0, "min_val": 0.0},
    "Griewank": {"func": safe_function_wrapper(griewank_nd), "grad": safe_gradient_wrapper(grad_griewank_nd), "start_range": [-600, 600], "min_pos": 0.0, "min_val": 0.0},
    "Schwefel": {"func": safe_function_wrapper(schwefel_nd), "grad": safe_gradient_wrapper(grad_schwefel_nd), "start_range": [-500, 500], "min_pos": 420.9687, "min_val": 0.0},
    "Michalewicz": {"func": michalewicz_nd, "grad": grad_michalewicz_nd, "start_range": [0, np.pi], "min_pos": None, "min_val": -1.801},
    "DropWave": {"func": safe_function_wrapper(drop_wave_nd), "grad": safe_gradient_wrapper(grad_drop_wave_nd), "start_range": [-5.12, 5.12], "min_pos": 0.0, "min_val": -1.0},
    "LunacekBiRastrigin": {"func": safe_function_wrapper(lunacek_bi_rastrigin_nd), "grad": None, "start_range": [-5.12, 5.12], "min_pos": None, "min_val": None},
    "ShiftedSphere": {"func": safe_function_wrapper(shifted_sphere_nd), "grad": safe_gradient_wrapper(grad_shifted_sphere_nd), "start_range": [0.0, 6.0], "min_pos": 3.0, "min_val": 0.0},
    "ShiftedRastrigin": {"func": safe_function_wrapper(shifted_rastrigin_nd), "grad": safe_gradient_wrapper(grad_shifted_rastrigin_nd), "start_range": [0.0, 6.0], "min_pos": 3.0, "min_val": 0.0},
    "HybridSphereRastrigin": {"func": hybrid_sphere_rastrigin_nd, "grad": grad_hybrid_sphere_rastrigin_nd, "start_range": [-5.12, 5.12], "min_pos": 0.0, "min_val": 0.0},
    "Ellipsoid": {"func": ellipsoid_nd, "grad": grad_ellipsoid_nd, "start_range": [-5.12, 5.12], "min_pos": 0.0, "min_val": 0.0},
    "Discus": {"func": discus_nd, "grad": grad_discus_nd, "start_range": [-5.12, 5.12], "min_pos": 0.0, "min_val": 0.0},
    "Zakharov": {"func": zakharov_nd, "grad": grad_zakharov_nd, "start_range": [-5, 10], "min_pos": 0.0, "min_val": 0.0},
    "StyblinskiTang": {"func": styblinski_tang_nd, "grad": grad_styblinski_tang_nd, "start_range": [-5, 5], "min_pos": -2.903534, "min_val_coeff": -39.166166},
    "Step": {"func": step_nd, "grad": None, "start_range": [-5.12, 5.12], "min_pos": 0.0, "min_val": 0.0},
    "QuarticNoise": {"func": quartic_with_noise_nd, "grad": grad_quartic_with_noise_nd, "start_range": [-1.28, 1.28], "min_pos": 0.0, "min_val": 0.0},
    "Branin2D": {"func": branin_2d, "grad": grad_branin_2d, "start_range": [-5, 10], "min_pos": None, "min_val": 0.397887}
}

def get_benchmark_configs():
    benchmark_configs = {}
    dimensions_to_test = [2, 5, 10, 20]

    for func_name, props in function_properties.items():
        for dim in dimensions_to_test:
            key = f"{func_name}_{dim}D"
            obj_func_raw, grad_func = props["func"], props["grad"]
            min_val = props.get("min_val_coeff", props.get("min_val", 0.0)) or 0.0
            if "min_val_coeff" in props:
                min_val *= dim

            min_pos = props.get("min_pos")
            true_min_pos = np.full(dim, min_pos) if min_pos is not None else None

            benchmark_configs[key] = {
                "func": obj_func_raw,
                "grad": grad_func,
                "dim": dim,
                "true_min_pos": true_min_pos,
                "true_min_val": min_val,
                "start_range": props["start_range"],
                "optimizers": {}
            }

            for opt_name, opt_conf in base_optimizer_params.items():
                cfg = opt_conf["params"].copy()
                iters_val = opt_conf.get("iterations") or get_iterations_for_dim(dim)

                if opt_conf["class"] in [NevergradOptimizer, CMAESOptimizer]:
                    cfg["dim"] = dim
                    if opt_conf["class"] == NevergradOptimizer:
                        cfg["budget"] = int(iters_val * 6)
                        cfg["lower"], cfg["upper"] = props["start_range"]

                if opt_conf["class"] == ScipyOptimizerWrapper:
                    lower, upper = props["start_range"]
                    padding = 1e-6
                    cfg["bounds"] = [[float(lower + padding), float(upper - padding)] for _ in range(dim)]

                # Enforce scalar output for certain optimizers
                requires_scalar = opt_conf["class"] in [ScipyOptimizerWrapper, NevergradOptimizer, CMAESOptimizer]
                if requires_scalar:
                    label = f"{key}_{opt_name}"
                    obj_func_wrapped = safe_scalar_func(obj_func_raw, label=label)
                else:
                    obj_func_wrapped = obj_func_raw

                benchmark_configs[key]["optimizers"][opt_name] = {
                    "class": opt_conf["class"],
                    "params": cfg,
                    "iterations": iters_val,
                    "func": obj_func_wrapped
                }

    return benchmark_configs
