import numpy as np
import logging

# Set up logger for safety diagnostics
safety_logger = logging.getLogger("safety_logger")
safety_logger.setLevel(logging.INFO)
if not safety_logger.handlers:
    handler = logging.FileHandler("overflow_safety.log")
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    safety_logger.addHandler(handler)

def safe_function_wrapper(func, clip_range=(-1e4, 1e4), penalty=np.inf, verbose=False):
    def safe_func(x):
        x_clipped = np.clip(x, *clip_range)
        try:
            val = func(x_clipped)
            if not np.isfinite(val):
                msg = f"[safe_func] Non-finite value: f(x)={val} | x.shape={x.shape}"
                if verbose:
                    print(msg)
                safety_logger.warning(msg)
                return penalty
            return val
        except Exception as e:
            msg = f"[safe_func] Exception: {e} | x.shape={x.shape}"
            if verbose:
                print(msg)
            safety_logger.error(msg)
            return penalty
    return safe_func

def safe_gradient_wrapper(grad_func, clip_range=(-1e4, 1e4), verbose=False):
    def safe_grad(x):
        x_clipped = np.clip(x, *clip_range)
        try:
            grad = grad_func(x_clipped)
            if not np.all(np.isfinite(grad)):
                msg = f"[safe_grad] Invalid gradient: grad={grad} | x.shape={x.shape}"
                if verbose:
                    print(msg)
                safety_logger.warning(msg)
                return np.zeros_like(x)
            return grad
        except Exception as e:
            msg = f"[safe_grad] Exception: {e} | x.shape={x.shape}"
            if verbose:
                print(msg)
            safety_logger.error(msg)
            return np.zeros_like(x)
    return safe_grad

def safe_scalar_func(func, max_abs_x=1e6, max_val=1e10, label="UnknownFunc"):
    def wrapped(x):
        x = np.clip(x, -max_abs_x, max_abs_x)
        if not np.all(np.isfinite(x)):
            safety_logger.warning(f"[{label}] Input has non-finite values: {x}")
            return np.inf, False
        try:
            val = func(x)
            if not np.isfinite(val):
                safety_logger.warning(f"[{label}] Output is non-finite: f(x) = {val} for x = {x}")
                return np.inf, False
            if np.abs(val) > max_val:
                safety_logger.warning(f"[{label}] Output exceeded max_val={max_val}: f(x) = {val} for x = {x}")
                return np.inf, False
            return val, True
        except Exception as e:
            safety_logger.error(f"[{label}] Exception during evaluation: {e} at x = {x}")
            return np.inf, False
    return wrapped
