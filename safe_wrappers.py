import numpy as np
import logging

# -----------------------------
# Logger Setup for Safety Layer
# -----------------------------
safety_logger = logging.getLogger("safety_logger")
safety_logger.setLevel(logging.INFO)

if not safety_logger.handlers:
    handler = logging.FileHandler("overflow_safety.log")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    safety_logger.addHandler(handler)

# -----------------------------
# Safe Function Wrapper
# -----------------------------
def safe_function_wrapper(func, clip_range=(-1e4, 1e4), penalty=np.inf, verbose=False):
    """
    Wraps an objective function to guard against NaNs, Infs, and numerical errors.
    Supports batched inputs.
    """
    def safe_func(x):
        x = np.asarray(x)
        if x.ndim > 1:
            return np.array([safe_func(xi) for xi in x])

        x_clipped = np.clip(x, *clip_range)
        try:
            val = func(x_clipped)
            val = np.asarray(val).flatten()
            if val.size != 1 or not np.isfinite(val[0]):
                msg = f"[safe_func] Invalid output: f(x)={val} | x.shape={x.shape}"
                if verbose: print(msg)
                safety_logger.warning(msg)
                return penalty
            return float(val[0])
        except Exception as e:
            msg = f"[safe_func] Exception: {e} | x.shape={x.shape}"
            if verbose: print(msg)
            safety_logger.error(msg)
            return penalty
    return safe_func

# -----------------------------
# Safe Gradient Wrapper
# -----------------------------
def safe_gradient_wrapper(grad_func, clip_range=(-1e4, 1e4), verbose=False):
    """
    Wraps a gradient function to avoid propagating invalid gradients.
    """
    def safe_grad(x):
        x = np.asarray(x).flatten()
        x_clipped = np.clip(x, *clip_range)
        try:
            grad = grad_func(x_clipped)
            grad = np.asarray(grad).flatten()
            if not np.all(np.isfinite(grad)):
                msg = f"[safe_grad] Invalid gradient: grad={grad} | x.shape={x.shape}"
                if verbose: print(msg)
                safety_logger.warning(msg)
                return np.zeros_like(x)
            return grad
        except Exception as e:
            msg = f"[safe_grad] Exception: {e} | x.shape={x.shape}"
            if verbose: print(msg)
            safety_logger.error(msg)
            return np.zeros_like(x)
    return safe_grad

# -----------------------------
# Safe Scalar Function Wrapper
# -----------------------------
def safe_scalar_func(func, max_abs_x=1e6, max_val=1e6, label="UnknownFunc"):
    """
    Wraps a function to ensure it always returns a valid scalar float.
    Enforces 1D inputs. Use for scalar-accepting optimizers like Scipy.
    """
    def wrapped(x):
        x = np.asarray(x).flatten()

        if x.ndim != 1:
            safety_logger.error(f"[{label}] Expected 1D input, got shape {x.shape}")
            return np.inf

        if not np.all(np.isfinite(x)):
            safety_logger.warning(f"[{label}] Input has non-finite values: {x}")
            return np.inf

        try:
            x_clipped = np.clip(x, -max_abs_x, max_abs_x)
            val = func(x_clipped)

            val = np.asarray(val).flatten()
            if val.size != 1:
                raise ValueError(f"[{label}] Non-scalar return from func: shape {val.shape}")
            val = float(val[0])

            if not np.isfinite(val):
                safety_logger.warning(f"[{label}] Output is non-finite: f(x) = {val} for x = {x}")
                return np.inf

            if abs(val) > max_val:
                safety_logger.warning(f"[{label}] Output exceeded max_val={max_val}: f(x) = {val} for x = {x}")
                return np.inf

            return val

        except Exception as e:
            safety_logger.error(f"[{label}] Exception during evaluation: {e} at x = {x}")
            return np.inf
    return wrapped
