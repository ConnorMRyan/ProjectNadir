import numpy as np

# === Benchmark Functions ===

def sphere_nd(x): return np.sum(x**2)
def grad_sphere_nd(x): return 2 * x

def rastrigin_nd(x): return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
def grad_rastrigin_nd(x): return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)

def rosenbrock_nd(x): return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
def grad_rosenbrock_nd(x):
    grad = np.zeros_like(x)
    grad[:-1] += -2 * (1 - x[:-1]) - 400 * x[:-1] * (x[1:] - x[:-1]**2)
    grad[1:] += 200 * (x[1:] - x[:-1]**2)
    return grad

def ackley_nd(x):
    d = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + 20 + np.e

def grad_ackley_nd(x):
    d = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    sqrt_term = np.sqrt(sum_sq / d + 1e-10)
    term1 = (20 * 0.2 * x) / (d * sqrt_term) * np.exp(-0.2 * sqrt_term)
    term2 = (2 * np.pi / d) * np.sin(2 * np.pi * x) * np.exp(sum_cos / d)
    return term1 + term2

def griewank_nd(x):
    sum_sq = np.sum(x**2)
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + sum_sq / 4000 - prod_cos

def grad_griewank_nd(x):
    i = np.arange(1, len(x) + 1)
    sqrt_i = np.sqrt(i)
    cos_terms = np.cos(x / sqrt_i)
    sin_terms = np.sin(x / sqrt_i)
    grad = x / 2000.0
    prod_cos = np.prod(cos_terms)
    for j in range(len(x)):
        if np.abs(cos_terms[j]) < 1e-10:
            continue
        dprod = -prod_cos * (sin_terms[j] / cos_terms[j]) / sqrt_i[j]
        grad[j] += dprod
    return grad

def schwefel_nd(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def grad_schwefel_nd(x):
    sqrt_abs_x = np.sqrt(np.abs(x))
    return -(np.sin(sqrt_abs_x) + x * np.cos(sqrt_abs_x) * 0.5 / (sqrt_abs_x + 1e-10) * np.sign(x))

def zakharov_nd(x):
    i = np.arange(1, len(x)+1)
    return np.sum(x**2) + (0.5 * np.sum(i * x))**2 + (0.5 * np.sum(i * x))**4

def grad_zakharov_nd(x):
    i = np.arange(1, len(x)+1)
    sum2 = 0.5 * np.sum(i * x)
    return 2 * x + 2 * sum2 * 0.5 * i + 4 * (sum2**3) * 0.5 * i

def styblinski_tang_nd(x):
    return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)

def grad_styblinski_tang_nd(x):
    return 0.5 * (4 * x**3 - 32 * x + 5)

# === Rotation Utilities ===

def get_random_rotation_matrix(dim):
    if dim == 1: return np.eye(1)
    q, _ = np.linalg.qr(np.random.randn(dim, dim))
    return q

def create_transformed_rosenbrock(dim):
    transform_matrix = get_random_rotation_matrix(dim)
    def transformed_rosenbrock_nd(x):
        return rosenbrock_nd(transform_matrix @ x)
    def grad_transformed_rosenbrock_nd(x):
        y = transform_matrix @ x
        return transform_matrix.T @ grad_rosenbrock_nd(y)
    return transformed_rosenbrock_nd, grad_transformed_rosenbrock_nd

# --- 🧠 Ill-Conditioned & Plateaus ---

def ellipsoid_nd(x):
    d = len(x)
    if d == 1: return x[0]**2
    conditioning = np.power(10, 6 * np.arange(d) / (d - 1.0))
    return np.sum(conditioning * x**2)

def grad_ellipsoid_nd(x):
    d = len(x)
    if d == 1: return 2 * x
    conditioning = np.power(10, 6 * np.arange(d) / (d - 1.0))
    return 2 * conditioning * x

def discus_nd(x):
    return 1e6 * x[0]**2 + np.sum(x[1:]**2)

def grad_discus_nd(x):
    grad = 2 * x
    grad[0] *= 1e6
    return grad

# --- 🌋 Deceptive or Pathological ---

def michalewicz_nd(x, m=10):
    d = len(x)
    i = np.arange(1, d + 1)
    return -np.sum(np.sin(x) * (np.sin(i * x**2 / np.pi))**(2 * m))

def grad_michalewicz_nd(x, m=10):
    d = len(x)
    i = np.arange(1, d + 1)
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_term = np.sin(i * x**2 / np.pi)
    cos_term = np.cos(i * x**2 / np.pi)

    term1 = -cos_x * (sin_term)**(2 * m)

    term2_factor = -sin_x * (2 * m) * (sin_term)**(2 * m - 1)
    term2_inner_grad = cos_term * (2 * i * x / np.pi)
    term2 = term2_factor * term2_inner_grad

    return term1 + term2

def drop_wave_nd(x):
    # Typically used in 2D, but generalized here
    sum_sq = np.sum(x**2)
    if sum_sq == 0: return -1.0
    return - (1 + np.cos(12 * np.sqrt(sum_sq))) / (0.5 * sum_sq + 2)

def grad_drop_wave_nd(x):
    sum_sq = np.sum(x**2)
    if sum_sq == 0: return np.zeros_like(x)
    sqrt_sum_sq = np.sqrt(sum_sq)

    numerator = 1 + np.cos(12 * sqrt_sum_sq)
    denominator = 0.5 * sum_sq + 2

    # Derivative of numerator
    d_num = 12 * np.sin(12 * sqrt_sum_sq) * (x / sqrt_sum_sq)
    # Derivative of denominator
    d_den = x

    # Apply quotient rule: (u'v - uv') / v^2
    grad = (d_num * denominator - numerator * d_den) / (denominator**2)
    return -grad

# --- 🎲 Noise and Non-Smoothness ---

def step_nd(x):
    # Non-differentiable
    return np.sum(np.floor(x + 0.5)**2)
# No gradient for Step function, so grad=None

def quartic_with_noise_nd(x):
    d = len(x)
    i = np.arange(1, d + 1)
    return np.sum(i * x**4) + np.random.uniform(0, 1)

def grad_quartic_with_noise_nd(x):
    # Gradient is calculated for the deterministic part only
    d = len(x)
    i = np.arange(1, d + 1)
    return 4 * i * x**3

# === Hybrid Functions ===

def hybrid_sphere_rastrigin_nd(x):
    d = len(x)
    return np.sum(x[:d//2]**2) + 10*(d - d//2) + np.sum(x[d//2:]**2 - 10 * np.cos(2 * np.pi * x[d//2:]))

def grad_hybrid_sphere_rastrigin_nd(x):
    d = len(x)
    grad = np.zeros_like(x)
    grad[:d//2] = 2 * x[:d//2]
    grad[d//2:] = 2 * x[d//2:] + 20 * np.pi * np.sin(2 * np.pi * x[d//2:])
    return grad

# === Function Rotation Utility ===

def create_rotated_function(base_func, base_grad, dim):
    R = get_random_rotation_matrix(dim)
    def f_rotated(x): return base_func(R @ x)
    def grad_rotated(x): return R.T @ base_grad(R @ x)
    return f_rotated, grad_rotated

# === Deceptive Function ===

def lunacek_bi_rastrigin_nd(x, mu1=2.5, d=1.0, s=1.0):
    n = len(x)
    mu2 = -np.sqrt((mu1**2 - d) / s)
    term1 = np.sum((x - mu1)**2)
    term2 = d * n + s * np.sum((x - mu2)**2)
    term3 = 10 * (n - np.sum(np.cos(2 * np.pi * x)))
    return min(term1, term2) + term3
# No reliable gradient; treat as derivative-free

# === Surrogate / Realistic Landscape ===

def branin_2d(x):
    x1, x2 = x[0], x[1]
    a = 1
    b = 5.1 / (4*np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8*np.pi)
    return a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1 - t)*np.cos(x1) + s

def grad_branin_2d(x):
    x1, x2 = x[0], x[1]
    a = 1
    b = 5.1 / (4*np.pi**2)
    c = 5 / np.pi
    t = 1 / (8*np.pi)
    dx1 = -2*a*(x2 - b*x1**2 + c*x1 - 6)*(2*b*x1 - c) + 10*(1 - t)*np.sin(x1)
    dx2 = 2*a*(x2 - b*x1**2 + c*x1 - 6)
    return np.array([dx1, dx2])

# === Shifted Variants ===

def shifted_sphere_nd(x, shift=None):
    if shift is None:
        shift = np.ones_like(x) * 3.0
    z = x - shift
    return np.sum(z**2)

def grad_shifted_sphere_nd(x, shift=None):
    if shift is None:
        shift = np.ones_like(x) * 3.0
    return 2 * (x - shift)

def shifted_rastrigin_nd(x, shift=None):
    if shift is None:
        shift = np.ones_like(x) * 3.0
    z = x - shift
    return 10 * len(x) + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

def grad_shifted_rastrigin_nd(x, shift=None):
    if shift is None:
        shift = np.ones_like(x) * 3.0
    z = x - shift
    return 2 * z + 20 * np.pi * np.sin(2 * np.pi * z)