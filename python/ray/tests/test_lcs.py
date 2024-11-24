import time
import logging
import os
import unittest
import ray
import random
import string
import psutil


class TestLCS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """One-time setup for all tests"""
        cls.bsize = 50  # 50x50 blocks = 20x20 grid
        cls.string_length = 1000
        
        # Generate base strings once for all tests
        cls.S1 = "".join(random.choices(string.ascii_uppercase, k=cls.string_length))
        cls.S2 = "".join(random.choices(string.ascii_uppercase, k=cls.string_length))

    def setUp(self):
        """Setup before each test"""
        # Initialize Ray before each test
        storage_dir = os.path.abspath("ray_cache")
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        storage_path = f"file://{storage_dir}"
        ray.init(storage=storage_path, include_dashboard=False)
        print(f"\nRay initialized with storage at: {storage_path}")

        # Set up logging for this test
        self.log_dir = "logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        log_filename = f'lcs_{self.string_length}x{self.string_length}_bsize{self.bsize}.log'
        log_path = os.path.join(self.log_dir, log_filename)
        
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )

        # Initialize LCS function
        self.lcs = self.init_lcs()
        
        # Set up dimensions
        self.m = len(self.S1)
        self.n = len(self.S2)
        
        # Verify grid size requirements
        self.assertTrue(self.m % self.bsize == 0)
        self.assertTrue(self.n % self.bsize == 0)
        print(f"Grid size will be: {self.m//self.bsize} x {self.n//self.bsize} blocks")

    def tearDown(self):
        """Cleanup after each test"""
        ray.shutdown()
        print("Ray shutdown completed")

    def init_lcs(self):
        """Initialize the LCS function"""
        @ray.remote(incremental=True)
        def lcs(X, Y, bleft, bup, bsize, Lleft, Lup, Ldiag):
            """
            LCS function that calculates longest common subsequence for a block
            """
            logging.info(f"Starting block calculation: (bleft={bleft}, bup={bup})")
            L = [[None] * bsize for i in range(bsize)]

            def l(i, j):
                if i >= 0 and j >= 0:
                    return L[i][j]
                if i < 0 and j < 0:
                    val = Ldiag[bsize+i][bsize+j] if Ldiag is not None else 0
                    logging.info(f"  Using diagonal value at ({i},{j}): {val}")
                    return val
                if i < 0:
                    val = Lleft[bsize+i][j] if Lleft is not None else 0
                    logging.info(f"  Using left value at ({i},{j}): {val}")
                    return val
                val = Lup[i][bsize+j] if Lup is not None else 0
                logging.info(f"  Using up value at ({i},{j}): {val}")
                return val

            for i in range(bsize):
                for j in range(bsize):
                    left = bleft*bsize + i
                    up = bup*bsize + j
                    if left == 0 or up == 0:
                        L[i][j] = 0
                        logging.info(f"  Setting boundary value L[{i}][{j}] = 0")
                    elif X[left-1] == Y[up-1]:
                        L[i][j] = l(i-1, j-1) + 1
                        logging.info(f"  Characters match at ({left-1},{up-1}): L[{i}][{j}] = {L[i][j]}")
                    else:
                        L[i][j] = max(l(i-1, j), l(i, j-1))
                        logging.info(f"  Characters don't match: L[{i}][{j}] = {L[i][j]}")

            logging.info(f"Completed block (bleft={bleft}, bup={bup})")
            return L
        
        return lcs

    def get_memory_stats(self):
        """Get current memory statistics"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        ray_memory = ray.available_resources()
        return {
            'process_memory': memory_info.rss / 1024 / 1024,  # MB
            'object_store_memory': ray_memory.get('object_store_memory', 0) / 1024 / 1024  # MB
        }

    def print_stats(self, phase=""):
        """Print cache and memory statistics"""
        print(f"\nStatistics {phase}:")
        print(f"Cache Stats:")
        total_ops = self.lcs._cache_hits + self.lcs._cache_misses
        if total_ops > 0:
            hit_ratio = self.lcs._cache_hits / total_ops * 100
            print(f"  - Cache hits: {self.lcs._cache_hits}")
            print(f"  - Cache misses: {self.lcs._cache_misses}")
            print(f"  - Hit ratio: {hit_ratio:.2f}%")
            print(f"  - Recomputation ratio: {100 - hit_ratio:.2f}%")
        
        mem_stats = self.get_memory_stats()
        print(f"Memory Stats:")
        print(f"  - Process memory: {mem_stats['process_memory']:.2f} MB")
        print(f"  - Available object store memory: {mem_stats['object_store_memory']:.2f} MB")

    def run_lcs(self, X, Y):
        """Helper method to run LCS calculation"""
        f = [[None] * (self.n//self.bsize + 1) for i in range(self.m//self.bsize + 1)]
        start_time = time.time()
        
        for bleft in range(0, self.m//self.bsize):
            for bup in range(0, self.n//self.bsize):
                fleft = f[bleft-1][bup] if bleft > 0 else None
                fup = f[bleft][bup-1] if bup > 0 else None
                fdiag = f[bleft-1][bup-1] if bleft > 0 and bup > 0 else None
                f[bleft][bup] = self.lcs.remote(X, Y, bleft, bup, self.bsize, fleft, fup, fdiag)

        result = ray.get(f[self.m//self.bsize-1][self.n//self.bsize-1])
        elapsed = time.time() - start_time
        return result, elapsed

    def test_caching_behavior(self):
        """Test basic caching behavior with two identical runs"""
        print("\nTesting caching behavior...")
        
        # First run
        print("Running initial calculation...")
        self.X = ray.put(self.S1)
        self.Y = ray.put(self.S2)
        L1, elapsed1 = self.run_lcs(self.X, self.Y)
        initial_result = L1[self.bsize-1][self.bsize-1]
        print(f"Initial run result: {initial_result}, Duration: {elapsed1:.2f}s")
        self.print_stats("after initial run")
        
        # Second run (should use cache)
        print("\nRunning cached calculation...")
        L2, elapsed2 = self.run_lcs(self.X, self.Y)
        cached_result = L2[self.bsize-1][self.bsize-1]
        print(f"Cached run result: {cached_result}, Duration: {elapsed2:.2f}s")
        self.print_stats("after cached run")
        
        print(f"\nCache speedup factor: {elapsed1/elapsed2:.2f}x")
        
        # Verify results
        self.assertEqual(initial_result, cached_result)
        self.assertLess(elapsed2, elapsed1)

    def test_suffix_modification(self):
        """Test with modification pattern: 1000x(800+200*)"""
        print("\nTesting 1000x(800+200*) modification...")
        
        # Original string for first run
        self.X = ray.put(self.S1)
        self.Y = ray.put(self.S2)
        L1, elapsed1 = self.run_lcs(self.X, self.Y)
        print(f"Original run completed in {elapsed1:.2f}s")
        self.print_stats("after original run")
        
        # Modified string: last 200 chars changed
        S1_mod = self.S1[:800] + "".join(random.choices(string.ascii_uppercase, k=200))
        X_modified = ray.put(S1_mod)
        L2, elapsed2 = self.run_lcs(X_modified, self.Y)
        print(f"Modified run completed in {elapsed2:.2f}s")
        self.print_stats("after modified run")
        
        print(f"\nSpeedup factor: {elapsed1/elapsed2:.2f}x")

    def test_middle_split_modification(self):
        """Test with modification pattern: 1000x(500+500*)"""
        print("\nTesting 1000x(500+500*) modification...")
        
        # Original string for first run
        self.X = ray.put(self.S1)
        self.Y = ray.put(self.S2)
        L1, elapsed1 = self.run_lcs(self.X, self.Y)
        print(f"Original run completed in {elapsed1:.2f}s")
        self.print_stats("after original run")
        
        # Modified string: last 500 chars changed
        S1_mod = self.S1[:500] + "".join(random.choices(string.ascii_uppercase, k=500))
        X_modified = ray.put(S1_mod)
        L2, elapsed2 = self.run_lcs(X_modified, self.Y)
        print(f"Modified run completed in {elapsed2:.2f}s")
        self.print_stats("after modified run")
        
        print(f"\nSpeedup factor: {elapsed1/elapsed2:.2f}x")

    def test_corner_modification(self):
        """Test with modification pattern: (900+100*)x(900+100*)"""
        print("\nTesting (900+100*)x(900+100*) modification...")
        
        # Original string for first run
        self.X = ray.put(self.S1)
        self.Y = ray.put(self.S2)
        L1, elapsed1 = self.run_lcs(self.X, self.Y)
        print(f"Original run completed in {elapsed1:.2f}s")
        self.print_stats("after original run")
        
        # Modified strings: last 100 chars changed in both
        S1_mod = self.S1[:900] + "".join(random.choices(string.ascii_uppercase, k=100))
        S2_mod = self.S2[:900] + "".join(random.choices(string.ascii_uppercase, k=100))
        X_modified = ray.put(S1_mod)
        Y_modified = ray.put(S2_mod)
        L2, elapsed2 = self.run_lcs(X_modified, Y_modified)
        print(f"Modified run completed in {elapsed2:.2f}s")
        self.print_stats("after modified run")
        
        print(f"\nSpeedup factor: {elapsed1/elapsed2:.2f}x")

    def test_prefix_modification(self):
        """Test with modification pattern: (100*+900)x(100*+900)"""
        print("\nTesting (100*+900)x(100*+900) modification...")
        
        # Original string for first run
        self.X = ray.put(self.S1)
        self.Y = ray.put(self.S2)
        L1, elapsed1 = self.run_lcs(self.X, self.Y)
        print(f"Original run completed in {elapsed1:.2f}s")
        self.print_stats("after original run")
        
        # Modified strings: first 100 chars changed in both
        S1_mod = "".join(random.choices(string.ascii_uppercase, k=100)) + self.S1[100:]
        S2_mod = "".join(random.choices(string.ascii_uppercase, k=100)) + self.S2[100:]
        X_modified = ray.put(S1_mod)
        Y_modified = ray.put(S2_mod)
        L2, elapsed2 = self.run_lcs(X_modified, Y_modified)
        print(f"Modified run completed in {elapsed2:.2f}s")
        self.print_stats("after modified run")
        
        print(f"\nSpeedup factor: {elapsed1/elapsed2:.2f}x")


if __name__ == '__main__':
    unittest.main()