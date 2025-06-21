import pandas as pd
import os
import glob

# --- Step 1: Find the most recent CSV file ---
result_files = glob.glob("benchmark_results/benchmark_results_*.csv")
if not result_files:
    raise FileNotFoundError("No benchmark result files found in 'benchmark_results/'.")

latest_file = max(result_files, key=os.path.getmtime)
print(f"Loading latest benchmark results: {latest_file}")

# --- Step 2: Load and check failures ---
df = pd.read_csv(latest_file)

fail_mask = (
        df.apply(lambda row: row.astype(str).str.contains("ERROR|Exception|must return a scalar", case=False).any(), axis=1)
        | (df["status"].astype(str).str.upper() == "FAIL")
)

failures_df = df[fail_mask]
top_errors = failures_df["error"].value_counts().head(10)

# --- Step 3: Display results ---
print("\nTop error messages:")
print(top_errors)

print(f"\nTotal failures: {len(failures_df)}")
print("\nSample failing rows:")
print(failures_df.head(10))
