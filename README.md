# ProjectNadir

**ProjectNadir** is a modular benchmarking framework for testing a wide variety of optimization algorithms against a broad set of classical objective functions.

It emphasizes safety, reliability, and extensibility in a research or development context.

---

## 🔧 Features

- 🌐 Benchmark dozens of classical optimization functions (Sphere, Rastrigin, Rosenbrock, etc.)
- 🤖 Plug-and-play support for:
  - Gradient Descent variants (GD, Adam, LineSearchGD)
  - Adaptive directional optimizers (Smarter, EvenSmarter)
  - Scipy optimizers (L-BFGS-B, COBYLA, etc.)
  - Nevergrad optimizers (NGOpt, DE, PSO, etc.)
  - CMA-ES
- 🔍 Fine-grained safety logging for numerical overflows, invalid gradients, and convergence anomalies
- 📈 Extensible configuration system for tuning hyperparameters and adding new optimizers or functions

---

## 📁 Directory Structure

```
ProjectNadir/
├── benchmark_functions.py     # Objective functions and gradients
├── config.py                  # Benchmark configurations
├── run_benchmark.py           # Benchmark entry point
├── optimizers.py              # Optimizer implementations
├── safe_wrappers.py           # Safe guards for numerical stability
├── logs/                      # Generated log files
└── results/                   # Output from benchmark runs
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ConnorMRyan/ProjectNadir.git
cd ProjectNadir
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> You may also need:
> ```bash
> pip install nevergrad scipy numpy
> ```

---

## ✅ Running the Benchmark

```bash
python run_benchmark.py
```

Or use filters to test specific cases:

```bash
python run_benchmark.py --optimizer Adam --function Rastrigin --dimensions 10
```

---

## 🧪 Example Output

The framework will automatically:
- Evaluate all combinations of optimizers and functions across dimensions
- Log progress to console and log files
- Save results in structured files for downstream analysis

---

## 🧱 Extending the Framework

### Adding a New Optimizer:

1. Create a subclass of `BaseOptimizer` in `optimizers.py`
2. Register it in `config.py` under `base_optimizer_params`

### Adding a New Benchmark Function:

1. Implement function and gradient in `benchmark_functions.py`
2. Register it in `function_properties` in `config.py`

---

## 🧯 Safety & Stability

ProjectNadir includes:
- Safe wrappers to guard against numerical issues (overflow, NaNs, Inf)
- Automatic clipping of input domains
- Logging of every dangerous evaluation to `overflow_safety.log`

---

## 📊 Benchmark Coverage

| Function             | Gradients | Safeguards | Notes                          |
|----------------------|-----------|------------|--------------------------------|
| Sphere               | ✅        | ✅         |                                |
| Rastrigin            | ✅        | ✅         |                                |
| Rosenbrock           | ✅        | ✅         |                                |
| Ackley               | ✅        | ✅         | Wrapped with overflow protection |
| DropWave             | ✅        | ✅         |                                |
| LunacekBiRastrigin   | ❌        | ✅         | No gradient available          |
| ShiftedSphere        | ✅        | ✅         |                                |

---

## 📚 References

- [Nevergrad](https://github.com/facebookresearch/nevergrad)
- [CMA-ES](https://cma-es.github.io/)
- [Scipy Optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html)

---

## 🛠 Requirements

- Python 3.8+
- NumPy
- SciPy
- Nevergrad

---

## 👤 Author

Connor M. Ryan  
GitHub: [@ConnorMRyan](https://github.com/ConnorMRyan)

---

## 📄 License

MIT License © 2025 Connor M. Ryan
