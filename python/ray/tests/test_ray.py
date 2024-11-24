import ray
import time
import os

storage_dir = os.path.abspath("ray_cache")
storage_path = f"file://{storage_dir}"
ray.init(storage=storage_path, include_dashboard=False)  
print(f"[TEST] Ray initialized with storage at: {storage_path}")

print("\n[TEST] About to define remote function with incremental=True")
@ray.remote(incremental=True)
def compute_heavy(x):
    print(f"Computing heavy operation for {x}")
    time.sleep(2)
    return x * x

print("[TEST] Remote function defined")

print("\nFirst call starting...")
start = time.time()
result1 = ray.get(compute_heavy.remote(5))
duration1 = time.time() - start
print(f"Result 1: {result1}, Duration: {duration1:.2f}s")

print("\nSecond call starting...")
start = time.time()
result2 = ray.get(compute_heavy.remote(5))
duration2 = time.time() - start
print(f"Result 2: {result2}, Duration: {duration2:.2f}s")

print("\nDurations similar:", abs(duration2 - duration1) < 0.1)