import numpy as np
import scipy.optimize


# === Base Class ===
class BaseOptimizer:
    def __init__(self, obj_func, grad_func=None, **kwargs):
        self.obj_func = obj_func
        self.grad_func = grad_func
        self.learning_rate = kwargs.get('learning_rate', 0.01)
        self.momentum_coeff = kwargs.get('momentum_coeff', 0.9)

    def optimize(self, start_pos, iterations):
        raise NotImplementedError("Subclasses must implement optimize()")


# === Scipy Wrapper ===
class ScipyOptimizerWrapper(BaseOptimizer):
    def __init__(self, obj_func, grad_func=None, method="Powell", bounds=None, **kwargs):
        super().__init__(obj_func, grad_func)
        self.method = method
        self.bounds = bounds

    def optimize(self, start_pos, iterations=None):
        minimize_kwargs = {
            "fun": lambda x: float(np.ravel(self.obj_func(x))[0]),
            "x0": start_pos,
            "method": self.method,
            "bounds": self.bounds,
            "options": {"maxiter": iterations} if iterations else {}
        }
        if self.grad_func and self.method not in ["Powell", "COBYLA", "Nelder-Mead"]:
            minimize_kwargs["jac"] = self.grad_func

        result = scipy.optimize.minimize(**minimize_kwargs)
        return [(result.x, None)], [result.fun]


# === CMA-ES ===
class CMAESOptimizer(BaseOptimizer):
    def __init__(self, obj_func, dim, initial_sigma=0.5, **kwargs):
        super().__init__(obj_func)
        self.dim = dim
        self.sigma = initial_sigma
        self.lambda_ = kwargs.get('lambda_', 4 + int(3 * np.log(dim)))
        self.mu = kwargs.get('mu', self.lambda_ // 2)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights**2)

        self.c_c = (4 + self.mu_eff / dim) / (dim + 4 + 2 * self.mu_eff / dim)
        self.c_sigma = (self.mu_eff + 2) / (dim + self.mu_eff + 5)
        self.c_1 = 2 / ((dim + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1,
                        2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((dim + 2)**2 + 2 * self.mu_eff / 2))
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (dim + 1)) - 1) + self.c_sigma

    def optimize(self, start_mean, iterations):
        mean = np.array(start_mean, dtype=float)
        C, ps, pc = np.eye(self.dim), np.zeros(self.dim), np.zeros(self.dim)
        history, f_values = [mean.copy()], []

        for gen in range(1, iterations + 1):
            D, B = np.linalg.eigh(C)
            D = np.maximum(D, 1e-10)
            D_sqrt = np.sqrt(D)
            y = (B @ np.diag(D_sqrt) @ np.random.randn(self.dim, self.lambda_)).T
            population = mean + self.sigma * y
            fitness = []
            for p in population:
                try:
                    val = self.obj_func(p)
                    val = float(np.ravel(val)[0])
                    fitness.append(val if np.isfinite(val) else np.inf)
                except Exception:
                    fitness.append(np.inf)
            fitness = np.array(fitness)

            idx = np.argsort(fitness)
            best_y = y[idx[:self.mu]]
            y_w = np.sum(best_y * self.weights[:, None], axis=0)
            mean += self.sigma * y_w

            C_inv_sqrt = B @ np.diag(1 / D_sqrt) @ B.T
            ps = (1 - self.c_sigma) * ps + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * C_inv_sqrt @ y_w
            h_sigma = np.linalg.norm(ps) / np.sqrt(1 - (1 - self.c_sigma)**(2 * gen)) / self.dim < 1.4 + 2 / (self.dim + 1)
            pc = (1 - self.c_c) * pc + h_sigma * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y_w

            C = (1 - self.c_1 - self.c_mu) * C + \
                self.c_1 * (np.outer(pc, pc) + (1 - h_sigma) * self.c_c * (2 - self.c_c) * C) + \
                self.c_mu * (best_y.T @ np.diag(self.weights) @ best_y)

            self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (np.linalg.norm(ps) / np.sqrt(self.dim) - 1))

            history.append(mean.copy())
            f_values.append(fitness[idx[0]])

        return np.array(history), f_values


# === Nevergrad Wrapper ===
class NevergradOptimizer(BaseOptimizer):
    def __init__(self, obj_func, dim, budget, lower=None, upper=None, optimizer_name="NGOpt", **kwargs):
        super().__init__(obj_func)
        self.dim = dim
        self.budget = budget
        self.optimizer_name = optimizer_name
        self.lower = lower
        self.upper = upper

    def optimize(self, start_pos, iterations=None):
        import nevergrad as ng

        if self.optimizer_name not in ng.optimizers.registry:
            raise ValueError(f"Optimizer '{self.optimizer_name}' not found in Nevergrad registry.")

        param = ng.p.Array(shape=(self.dim,), lower=self.lower, upper=self.upper)
        optimizer = ng.optimizers.registry[self.optimizer_name](parametrization=param, budget=self.budget)

        if start_pos is not None:
            try:
                optimizer.suggest(start_pos)
            except Exception:
                pass

        history, f_values = [], []
        for _ in range(self.budget):
            x = optimizer.ask()
            try:
                val = self.obj_func(x.value)
                val = float(np.ravel(val)[0])
                if not np.isfinite(val): val = np.inf
            except Exception:
                val = np.inf
            optimizer.tell(x, val)
            rec = optimizer.recommend()
            history.append(rec.value.copy())
            f_values.append(rec.loss)

        return np.array(history), f_values

# === Gradient Descent Variants ===

class StandardGradientDescent(BaseOptimizer):
    def __init__(self, obj_func, grad_func, learning_rate=0.01, anneal_threshold=1e-4, anneal_factor=0.5, momentum_coeff=0.9, **kwargs):
        super().__init__(obj_func, grad_func, learning_rate=learning_rate, momentum_coeff=momentum_coeff)
        self.anneal_threshold = anneal_threshold
        self.anneal_factor = anneal_factor

    def optimize(self, start_pos, iterations):
        pos = np.array(start_pos, dtype=float)
        prev_pos = pos.copy()
        path, f_values = [pos.copy()], [self.obj_func(pos)]
        lr = self.learning_rate

        for _ in range(iterations):
            grad = self.grad_func(pos)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < self.anneal_threshold:
                lr *= self.anneal_factor
            velocity = pos - prev_pos
            direction = -grad / (grad_norm + 1e-10)
            new_pos = pos + lr * direction + self.momentum_coeff * velocity
            path.append(pos.copy())
            f_values.append(self.obj_func(pos))
            prev_pos, pos = pos.copy(), new_pos

        return np.array(path), f_values


class AdamOptimizer(BaseOptimizer):
    def __init__(self, obj_func, grad_func, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
        super().__init__(obj_func, grad_func, learning_rate=learning_rate)
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon

    def optimize(self, start_pos, iterations):
        pos = np.array(start_pos, dtype=float)
        m, v = np.zeros_like(pos), np.zeros_like(pos)
        path, f_values = [pos.copy()], [self.obj_func(pos)]

        for t in range(1, iterations + 1):
            grad = self.grad_func(pos)
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            pos -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            path.append(pos.copy())
            f_values.append(self.obj_func(pos))

        return np.array(path), f_values


class LineSearchGDOptimizer(BaseOptimizer):
    def __init__(self, obj_func, grad_func, initial_step_size=1.0, momentum_coeff=0.9, shrink=0.5, c=0.1, **kwargs):
        super().__init__(obj_func, grad_func, momentum_coeff=momentum_coeff)
        self.initial_step_size = initial_step_size
        self.shrink = shrink
        self.c = c

    def _backtracking(self, pos, grad, f_val):
        step = self.initial_step_size
        direction = -grad
        while self.obj_func(pos + step * direction) > f_val + self.c * step * np.dot(grad, direction):
            step *= self.shrink
            if step < 1e-10:
                return 0.0
        return step

    def optimize(self, start_pos, iterations):
        pos = np.array(start_pos, dtype=float)
        velocity = np.zeros_like(pos)
        history, f_values = [(pos.copy(), None)], [self.obj_func(pos)]

        for _ in range(iterations):
            f_val = f_values[-1]
            grad = self.grad_func(pos)
            step = self._backtracking(pos, grad, f_val)
            velocity = self.momentum_coeff * velocity - step * grad
            pos += velocity
            history.append((pos.copy(), None))
            f_values.append(self.obj_func(pos))

        return history, f_values

# === Helper Functions for Directional Methods ===

def generate_directions(dim, n=20):
    dirs = np.random.randn(n, dim)
    return dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-10)

def generate_smarter_directions(dim, n, grad, momentum=None, exploration_ratio=0.5):
    directions = []
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 1e-8:
        directions.append(-grad / grad_norm)
    if momentum is not None:
        momentum_norm = np.linalg.norm(momentum)
        if momentum_norm > 1e-8:
            momentum_dir = momentum / momentum_norm
            if not directions or np.dot(directions[0], momentum_dir) < 0.99:
                directions.append(momentum_dir)
    num_random = n - len(directions)
    if num_random > 0:
        rand_dirs = np.random.randn(num_random, dim)
        rand_dirs /= (np.linalg.norm(rand_dirs, axis=1, keepdims=True) + 1e-10)
        directions.extend(rand_dirs)
    return np.array(directions)

def adaptive_k(grad_norm, entropy, total_dirs):
    score = (20 / (grad_norm + 0.1)) * (1 + entropy / (np.log(total_dirs) + 1e-10))
    return int(np.clip(np.ceil(score), 1, total_dirs))


# === Adaptive Directional Optimizers ===

class AdaptiveOptimizer(BaseOptimizer):
    def __init__(self, obj_func, grad_func, step_size=0.1, jump_threshold=1e-3, jump_frequency=5, jump_scale=0.1,
                 anneal_threshold=1e-4, anneal_factor=0.5, momentum_coeff=0.9, **kwargs):
        super().__init__(obj_func, grad_func, learning_rate=step_size, momentum_coeff=momentum_coeff)
        self.jump_threshold = jump_threshold
        self.jump_frequency = jump_frequency
        self.jump_scale = jump_scale
        self.anneal_threshold = anneal_threshold
        self.anneal_factor = anneal_factor

    def optimize(self, start_pos, iterations):
        pos = np.array(start_pos, dtype=float)
        prev_pos = pos.copy()
        history, f_values = [], []
        stagnation, step_size = 0, self.learning_rate

        for _ in range(iterations):
            grad = self.grad_func(pos)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < self.anneal_threshold:
                step_size *= self.anneal_factor
            if grad_norm < self.jump_threshold:
                stagnation += 1
            else:
                stagnation = 0
            if stagnation >= self.jump_frequency:
                pos += np.random.uniform(-self.jump_scale, self.jump_scale, size=pos.shape)
                stagnation = 0

            dirs = generate_directions(len(pos))
            probed = pos + step_size * dirs
            values = np.array([self.obj_func(p) for p in probed])
            probs = np.abs(values)
            probs = probs / np.sum(probs) if np.sum(probs) > 1e-10 else np.ones_like(probs) / len(probs)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            k = adaptive_k(grad_norm, entropy, len(dirs))
            best_dir = dirs[np.argmin(values[:k])]
            velocity = pos - prev_pos
            new_pos = pos + step_size * best_dir + self.momentum_coeff * velocity

            history.append((pos.copy(), k))
            f_values.append(self.obj_func(pos))
            prev_pos, pos = pos.copy(), new_pos

        return history, f_values


class SmarterAdaptiveOptimizer(AdaptiveOptimizer):
    def __init__(self, obj_func, grad_func, max_directions=20, min_directions=5, grad_norm_threshold=5.0, **kwargs):
        super().__init__(obj_func, grad_func, **kwargs)
        self.max_directions = max_directions
        self.min_directions = min_directions
        self.grad_norm_threshold = grad_norm_threshold

    def optimize(self, start_pos, iterations):
        pos = np.array(start_pos, dtype=float)
        prev_pos = pos.copy()
        history, f_values = [], []
        stagnation, step_size = 0, self.learning_rate

        for _ in range(iterations):
            grad = self.grad_func(pos)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < self.anneal_threshold:
                step_size *= self.anneal_factor
            if grad_norm < self.jump_threshold:
                stagnation += 1
            else:
                stagnation = 0
            if stagnation >= self.jump_frequency:
                pos += np.random.uniform(-self.jump_scale, self.jump_scale, size=pos.shape)
                stagnation = 0

            ratio = min(1.0, grad_norm / self.grad_norm_threshold)
            num_dirs = int(self.max_directions - (self.max_directions - self.min_directions) * ratio)
            velocity = pos - prev_pos
            dirs = generate_smarter_directions(len(pos), n=num_dirs, grad=grad, momentum=velocity)
            probed = pos + step_size * dirs
            values = np.array([self.obj_func(p) for p in probed])
            best_dir = dirs[np.argmin(values)]
            new_pos = pos + step_size * best_dir + self.momentum_coeff * velocity

            history.append((pos.copy(), num_dirs))
            f_values.append(self.obj_func(pos))
            prev_pos, pos = pos.copy(), new_pos

        return history, f_values

class EvenSmarterAdaptiveOptimizer(SmarterAdaptiveOptimizer):
    def __init__(self, obj_func, grad_func, escape_probes=5, hessian_eps=1e-4, **kwargs):
        super().__init__(obj_func, grad_func, **kwargs)
        self.escape_probes = escape_probes
        self.hessian_eps = hessian_eps

    def _find_escape_direction(self, pos, grad):
        for _ in range(self.escape_probes):
            v = np.random.randn(len(pos))
            v /= np.linalg.norm(v)
            grad_new = self.grad_func(pos + self.hessian_eps * v)
            Hv = (grad_new - grad) / self.hessian_eps
            curvature = np.dot(v, Hv)
            if curvature < 0:
                return -v if np.dot(grad, v) > 0 else v
        return None

    def optimize(self, start_pos, iterations):
        pos = np.array(start_pos, dtype=float)
        prev_pos = pos.copy()
        history, f_values = [], []
        stagnation, step_size = 0, self.learning_rate

        for _ in range(iterations):
            grad = self.grad_func(pos)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < self.anneal_threshold:
                step_size *= self.anneal_factor
            if grad_norm < self.jump_threshold:
                stagnation += 1
            else:
                stagnation = 0

            if stagnation >= self.jump_frequency:
                escape = self._find_escape_direction(pos, grad)
                if escape is not None:
                    pos += self.jump_scale * 1.5 * escape
                else:
                    pos += np.random.uniform(-self.jump_scale / 2, self.jump_scale / 2, size=pos.shape)
                stagnation = 0

            ratio = min(1.0, grad_norm / self.grad_norm_threshold)
            num_dirs = int(self.max_directions - (self.max_directions - self.min_directions) * ratio)
            velocity = pos - prev_pos
            dirs = generate_smarter_directions(len(pos), n=num_dirs, grad=grad, momentum=velocity)
            probed = pos + step_size * dirs
            values = np.array([self.obj_func(p) for p in probed])
            best_dir = dirs[np.argmin(values)]
            new_pos = pos + step_size * best_dir + self.momentum_coeff * velocity

            history.append((pos.copy(), num_dirs))
            f_values.append(self.obj_func(pos))
            prev_pos, pos = pos.copy(), new_pos

        return history, f_values
