# optimizers.py
import numpy as np
from scipy.optimize import minimize


# === Helper Functions for Optimizers ===

def generate_directions(dim, n=20, jitter=0.0, apply_rotation=False):
    if dim == 2:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        dirs = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    else:
        dirs = np.random.randn(n, dim)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-10
    if apply_rotation and dim == 2:
        angle_offset = np.random.uniform(0, 2 * np.pi)
        rot = np.array([[np.cos(angle_offset), -np.sin(angle_offset)],[np.sin(angle_offset),  np.cos(angle_offset)]])
        dirs = dirs @ rot.T
    if jitter > 0:
        dirs += np.random.randn(*dirs.shape) * jitter
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-10
    return dirs

def generate_smarter_directions(dim, n, grad, momentum=None, exploration_ratio=0.5):
    directions = []
    grad_norm = np.linalg.norm(grad)
    momentum_norm = np.linalg.norm(momentum) if momentum is not None else 0
    if grad_norm > 1e-8:
        grad_dir = -grad / grad_norm
        directions.append(grad_dir)
    if momentum is not None and momentum_norm > 1e-8:
        momentum_dir = momentum / momentum_norm
        if len(directions) == 0 or (len(directions) > 0 and np.dot(grad_dir, momentum_dir) < 0.99):
            directions.append(momentum_dir)
    num_random_dirs = int(n * exploration_ratio)
    num_guided_dirs = n - len(directions) - num_random_dirs
    if len(directions) > 0 and num_guided_dirs > 0:
        base_dir = directions[0]
        for _ in range(num_guided_dirs):
            noise = np.random.randn(dim) * 0.4
            new_dir = base_dir + noise
            directions.append(new_dir / (np.linalg.norm(new_dir) + 1e-10))
    num_to_fill = n - len(directions)
    if num_to_fill > 0:
        random_dirs = np.random.randn(num_to_fill, dim)
        random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True) + 1e-10
        directions.extend(random_dirs)
    return np.array(directions)

def adaptive_k(grad_norm, step_vals_entropy, all_dirs_len):
    k_val = (20 / (grad_norm + 0.1)) * (1 + step_vals_entropy / (np.log(all_dirs_len) + 1e-10))
    return int(np.clip(np.ceil(k_val), 1, all_dirs_len))


# === Base Class ===
class BaseOptimizer:
    def __init__(self, obj_func, grad_func=None, **kwargs):
        self.obj_func = obj_func
        self.grad_func = grad_func
        self.learning_rate = kwargs.get('learning_rate', 0.01)
        self.momentum_coeff = kwargs.get('momentum_coeff', 0.9)
    def optimize(self, start_pos, iterations):
        raise NotImplementedError("Subclasses must implement optimize()")

# === Optimizer Implementations ===

class StandardGradientDescent(BaseOptimizer):
    def __init__(self, obj_func, grad_func, learning_rate, anneal_threshold, anneal_factor, momentum_coeff, **kwargs):
        super().__init__(obj_func, grad_func, learning_rate=learning_rate, momentum_coeff=momentum_coeff)
        self.anneal_threshold = anneal_threshold
        self.anneal_factor = anneal_factor
    def optimize(self, start_pos, iterations):
        pos = np.array(start_pos, dtype=float)
        path, f_values_history = [pos.copy()], [self.obj_func(pos)]
        current_lr = self.learning_rate
        prev_pos = pos.copy()
        for _ in range(iterations):
            grad = self.grad_func(pos)
            grad_magnitude = np.linalg.norm(grad)
            if grad_magnitude < self.anneal_threshold: current_lr *= self.anneal_factor
            velocity = pos - prev_pos
            direction = -grad / (grad_magnitude + 1e-10)
            new_pos = pos + current_lr * direction + self.momentum_coeff * velocity
            path.append(pos.copy())
            f_values_history.append(self.obj_func(pos))
            prev_pos, pos = pos.copy(), new_pos
        return np.array(path), f_values_history

class AdamOptimizer(BaseOptimizer):
    def __init__(self, obj_func, grad_func, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
        super().__init__(obj_func, grad_func, learning_rate=learning_rate)
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
    def optimize(self, start_pos, iterations):
        pos = np.array(start_pos, dtype=float)
        path, f_values_history = [pos.copy()], [self.obj_func(pos)]
        m, v = np.zeros_like(pos), np.zeros_like(pos)
        for t in range(1, iterations + 1):
            grad = self.grad_func(pos)
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            m_hat, v_hat = m / (1 - self.beta1 ** t), v / (1 - self.beta2 ** t)
            pos -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            path.append(pos.copy())
            f_values_history.append(self.obj_func(pos))
        return np.array(path), f_values_history

class LineSearchGDOptimizer(BaseOptimizer):
    def __init__(self, obj_func, grad_func, initial_step_size=1.0, momentum_coeff=0.9, line_search_shrink=0.5, line_search_c=0.1, **kwargs):
        super().__init__(obj_func, grad_func, momentum_coeff=momentum_coeff)
        self.initial_step_size, self.line_search_shrink, self.line_search_c = initial_step_size, line_search_shrink, line_search_c
    def _backtracking_line_search(self, pos, grad, f_val):
        step_size, search_dir = self.initial_step_size, -grad
        while self.obj_func(pos + step_size * search_dir) > f_val + self.line_search_c * step_size * np.dot(grad, search_dir):
            step_size *= self.line_search_shrink
            if step_size < 1e-10: return 0
        return step_size
    def optimize(self, start_pos, iterations):
        pos = np.array(start_pos, dtype=float)
        history, f_values_history = [(pos.copy(), None)], [self.obj_func(pos)]
        velocity = np.zeros_like(pos)
        for _ in range(iterations):
            f_val = f_values_history[-1]
            g = self.grad_func(pos)
            step_size = self._backtracking_line_search(pos, g, f_val)
            velocity = (self.momentum_coeff * velocity) - (step_size * g)
            pos += velocity
            history.append((pos.copy(), None))
            f_values_history.append(self.obj_func(pos))
        return history, f_values_history

class AdaptiveOptimizer(BaseOptimizer):
    def __init__(self, obj_func, grad_func, step_size, jump_threshold, jump_frequency, jump_scale, anneal_threshold, anneal_factor, momentum_coeff, apply_rotation_to_dirs=False, **kwargs):
        super().__init__(obj_func, grad_func, learning_rate=step_size, momentum_coeff=momentum_coeff)
        self.step_size, self.jump_threshold, self.jump_frequency, self.jump_scale = step_size, jump_threshold, jump_frequency, jump_scale
        self.anneal_threshold, self.anneal_factor, self.apply_rotation_to_dirs = anneal_threshold, anneal_factor, apply_rotation_to_dirs
    def optimize(self, start_pos, iterations):
        pos, prev_pos = np.array(start_pos, dtype=float), np.array(start_pos, dtype=float)
        history, f_values_history = [], []
        stagnation_counter, current_step_size = 0, self.step_size
        for i in range(iterations):
            g = self.grad_func(pos)
            grad_magnitude = np.linalg.norm(g)
            if grad_magnitude < self.anneal_threshold: current_step_size *= self.anneal_factor
            if grad_magnitude < self.jump_threshold: stagnation_counter += 1
            else: stagnation_counter = 0
            if stagnation_counter >= self.jump_frequency:
                pos += np.random.uniform(-self.jump_scale, self.jump_scale, size=pos.shape)
                stagnation_counter = 0
            all_dirs = generate_directions(len(pos), apply_rotation=self.apply_rotation_to_dirs)
            steps_probed = pos + current_step_size * all_dirs
            step_values_probed = np.array([self.obj_func(s) for s in steps_probed])
            probs = np.abs(step_values_probed)
            probs = probs / np.sum(probs) if np.sum(probs) > 1e-10 else np.ones_like(probs) / len(probs)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            k = adaptive_k(grad_magnitude, entropy, len(all_dirs))
            best_k_indices = np.argsort(step_values_probed)[:k]
            best_direction = all_dirs[np.argmin(step_values_probed[best_k_indices])] if best_k_indices.size > 0 else np.zeros_like(pos)
            velocity = pos - prev_pos
            new_pos = pos + current_step_size * best_direction + self.momentum_coeff * velocity
            history.append((pos.copy(), k))
            f_values_history.append(self.obj_func(pos))
            prev_pos, pos = pos.copy(), new_pos
        return history, f_values_history

class SmarterAdaptiveOptimizer(BaseOptimizer):
    def __init__(self, obj_func, grad_func, step_size, jump_threshold, jump_frequency, jump_scale, anneal_threshold, anneal_factor, momentum_coeff, max_directions=20, min_directions=5, grad_norm_threshold=5.0, **kwargs):
        super().__init__(obj_func, grad_func, **kwargs)
        self.step_size, self.jump_threshold, self.jump_frequency, self.jump_scale = step_size, jump_threshold, jump_frequency, jump_scale
        self.anneal_threshold, self.anneal_factor, self.momentum_coeff = anneal_threshold, anneal_factor, momentum_coeff
        self.max_directions, self.min_directions, self.grad_norm_threshold = max_directions, min_directions, grad_norm_threshold
    def optimize(self, start_pos, iterations):
        pos, prev_pos = np.array(start_pos, dtype=float), np.array(start_pos, dtype=float)
        history, f_values_history = [], []
        stagnation_counter, current_step_size = 0, self.step_size
        for i in range(iterations):
            g = self.grad_func(pos)
            grad_magnitude = np.linalg.norm(g)
            if grad_magnitude < self.anneal_threshold: current_step_size *= self.anneal_factor
            if grad_magnitude < self.jump_threshold: stagnation_counter += 1
            else: stagnation_counter = 0
            if stagnation_counter >= self.jump_frequency:
                pos += np.random.uniform(-self.jump_scale, self.jump_scale, size=pos.shape)
                stagnation_counter = 0
            grad_ratio = min(1.0, grad_magnitude / self.grad_norm_threshold)
            num_directions = int(self.max_directions - (self.max_directions - self.min_directions) * grad_ratio)
            velocity = pos - prev_pos
            all_dirs = generate_smarter_directions(len(pos), n=num_directions, grad=g, momentum=velocity)
            steps_probed = pos + current_step_size * all_dirs
            step_values_probed = np.array([self.obj_func(s) for s in steps_probed])
            best_direction = all_dirs[np.argmin(step_values_probed)] if step_values_probed.size > 0 else np.zeros_like(pos)
            new_pos = pos + current_step_size * best_direction + self.momentum_coeff * velocity
            history.append((pos.copy(), num_directions))
            f_values_history.append(self.obj_func(pos))
            prev_pos, pos = pos.copy(), new_pos
        return history, f_values_history

class EvenSmarterAdaptiveOptimizer(SmarterAdaptiveOptimizer):
    def __init__(self, obj_func, grad_func, escape_probes=5, hessian_eps=1e-4, **kwargs):
        super().__init__(obj_func, grad_func, **kwargs)
        self.escape_probes, self.hessian_eps = escape_probes, hessian_eps
    def _find_escape_direction(self, pos, current_grad):
        for _ in range(self.escape_probes):
            v = np.random.randn(len(pos))
            v /= np.linalg.norm(v)
            grad_at_new_pos = self.grad_func(pos + self.hessian_eps * v)
            Hv = (grad_at_new_pos - current_grad) / self.hessian_eps
            curvature = np.dot(v, Hv)
            if curvature < 0: return -v if np.dot(current_grad, v) > 0 else v
        return None
    def optimize(self, start_pos, iterations):
        pos, prev_pos = np.array(start_pos, dtype=float), np.array(start_pos, dtype=float)
        history, f_values_history = [], []
        stagnation_counter, current_step_size = 0, self.step_size
        for i in range(iterations):
            g = self.grad_func(pos)
            grad_magnitude = np.linalg.norm(g)
            if grad_magnitude < self.anneal_threshold: current_step_size *= self.anneal_factor
            if grad_magnitude < self.jump_threshold: stagnation_counter += 1
            else: stagnation_counter = 0
            if stagnation_counter >= self.jump_frequency:
                escape_direction = self._find_escape_direction(pos, g)
                if escape_direction is not None:
                    pos += self.jump_scale * 1.5 * escape_direction
                else:
                    pos += np.random.uniform(-self.jump_scale / 2, self.jump_scale / 2, size=pos.shape)
                stagnation_counter = 0
            grad_ratio = min(1.0, grad_magnitude / self.grad_norm_threshold)
            num_directions = int(self.max_directions - (self.max_directions - self.min_directions) * grad_ratio)
            velocity = pos - prev_pos
            all_dirs = generate_smarter_directions(len(pos), n=num_directions, grad=g, momentum=velocity)
            steps_probed = pos + current_step_size * all_dirs
            step_values_probed = np.array([self.obj_func(s) for s in steps_probed])
            best_direction = all_dirs[np.argmin(step_values_probed)] if step_values_probed.size > 0 else np.zeros_like(pos)
            new_pos = pos + current_step_size * best_direction + self.momentum_coeff * velocity
            history.append((pos.copy(), num_directions))
            f_values_history.append(self.obj_func(pos))
            prev_pos, pos = pos.copy(), new_pos
        return history, f_values_history

class CMAESOptimizer(BaseOptimizer):
    def __init__(self, obj_func, dim, initial_sigma, lambda_=None, mu=None, **kwargs):
        super().__init__(obj_func)
        self.dim, self.sigma = dim, initial_sigma
        self.lambda_ = lambda_ or 4 + int(3 * np.log(dim))
        self.mu = mu or self.lambda_ // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights**2)
        self.c_c, self.c_sigma = (4 + self.mu_eff/self.dim) / (self.dim+4+2*self.mu_eff/self.dim), (self.mu_eff+2)/(self.dim+self.mu_eff+5)
        self.c_1, self.c_mu = 2/((self.dim+1.3)**2+self.mu_eff), min(1-self.c_1, 2*(self.mu_eff-2+1/self.mu_eff)/((self.dim+2)**2+2*self.mu_eff/2))
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff-1)/(self.dim+1)) - 1) + self.c_sigma
    def optimize(self, start_mean, iterations):
        mean = np.array(start_mean, dtype=float)
        C, ps, pc = np.eye(self.dim), np.zeros(self.dim), np.zeros(self.dim)
        history, f_values_history = [mean.copy()], [self.obj_func(mean)]
        for gen in range(1, iterations + 1):
            D, B = np.linalg.eigh(C)
            D_sqrt = np.sqrt(D)
            y = (B @ np.diag(D_sqrt) @ np.random.randn(self.dim, self.lambda_)).T
            population = mean + self.sigma * y
            fitness = np.array([self.obj_func(p) for p in population])
            idx = np.argsort(fitness)
            best_y, y_w = y[idx[:self.mu]], np.sum(best_y * self.weights[:, np.newaxis], axis=0)
            mean += self.sigma * y_w
            C_inv_sqrt = B @ np.diag(1/D_sqrt) @ B.T
            ps = (1-self.c_sigma)*ps + np.sqrt(self.c_sigma*(2-self.c_sigma)*self.mu_eff) * C_inv_sqrt @ y_w
            h_sigma = np.linalg.norm(ps)/np.sqrt(1-(1-self.c_sigma)**(2*gen))/self.dim < 1.4 + 2/(self.dim+1)
            pc = (1-self.c_c)*pc + h_sigma * np.sqrt(self.c_c*(2-self.c_c)*self.mu_eff) * y_w
            C = (1-self.c_1-self.c_mu)*C + self.c_1*(np.outer(pc,pc)+(1-h_sigma)*self.c_c*(2-self.c_c)*C) + self.c_mu*(best_y.T@np.diag(self.weights)@best_y)
            self.sigma *= np.exp((self.c_sigma/self.d_sigma)*(np.linalg.norm(ps)/np.sqrt(self.dim) - 1))
            history.append(mean.copy())
            f_values_history.append(fitness[idx[0]])
        return np.array(history), f_values_history

class ScipyOptimizerWrapper(BaseOptimizer):
    def __init__(self, obj_func, grad_func=None, method="Powell", bounds=None, **kwargs):
        self.obj_func = obj_func
        self.grad_func = grad_func
        self.method = method
        self.bounds = bounds
        self.kwargs = kwargs

    def optimize(self, start_pos, iterations=None):
        result = opt.minimize(
            self.obj_func,
            start_pos,
            method=self.method,
            jac=self.grad_func if self.grad_func else None,
            bounds=self.bounds,
            options={"maxiter": iterations} if iterations else {},
        )
        return [(start_pos, None)], [result.fun]  # Return dummy history + final value

class NevergradOptimizer(BaseOptimizer):
    def __init__(self, obj_func, dim, budget, parametrization_type="Array", lower=None, upper=None, **kwargs):
        super().__init__(obj_func)
        self.dim, self.budget, self.parametrization_type = dim, budget, parametrization_type
        self.optimizer_name = kwargs.get("nevergrad_optimizer_name", "NGO")
        self.lower, self.upper = lower, upper

    def optimize(self, start_pos, iterations=None):
        # --- THIS IS THE FIX ---
        # Import nevergrad here, only when a Nevergrad trial is actually running.
        import nevergrad as ng
        # --------------------

        param = ng.p.Array(shape=(self.dim,), lower=self.lower, upper=self.upper)
        optimizer = ng.optimizers.registry[self.optimizer_name](parametrization=param, budget=self.budget)

        if start_pos is not None:
            try:
                optimizer.suggest(start_pos)
            except Exception:
                pass # Fail silently if suggestion is not supported

        history, f_values_history = [], []
        for _ in range(self.budget):
            x = optimizer.ask()
            value = self.obj_func(x.value)
            optimizer.tell(x, value)
            rec = optimizer.recommend()
            history.append(rec.value.copy())
            f_values_history.append(rec.loss)

        return np.array(history), f_values_history