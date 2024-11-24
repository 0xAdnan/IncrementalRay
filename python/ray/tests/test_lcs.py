import time
import logging
import os

import ray

storage_dir = os.path.abspath("ray_cache")
storage_path = f"file://{storage_dir}"
ray.init(storage=storage_path, include_dashboard=False)
print(f" Ray initialized with storage at: {storage_path}")

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# S1 = "CLASS"
# S2 = "LABS"
S1 = "A"*30
S2 = "A"*30
bsize = 10

# Create log filename with dimensions and block size
log_filename = f'lcs_{len(S1)}x{len(S2)}_bsize{bsize}.log'
log_path = os.path.join(log_dir, log_filename)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

@ray.remote(incremental=True)
def lcs(X, Y, bleft, bup, bsize, Lleft, Lup, Ldiag):
    """
        :param X, Y: input strings
        :param (bleft, bup): block index
        :param bsize: size of the block: bsize*bsize
        :param Lleft: left block
        :param Lup: up block
        :param Ldiag: diagonal block

        :return: outputs block at the input block index
    """
    logging.info(f"Starting block calculation: (bleft={bleft}, bup={bup})")
    # Declaring the array for storing the dp values
    L = [[None] * bsize for i in range(bsize)]

    def l(i, j):
        if i >= 0 and j >= 0:
            return L[i][j] #current block
        if i < 0 and j < 0:
            val = Ldiag[bsize+i][bsize+j] if Ldiag is not None else 0 #diagonal block
            logging.info(f"  Using diagonal value at ({i},{j}): {val}")
            return val
        if i < 0:
            val = Lleft[bsize+i][j] if Lleft is not None else 0 #left block
            logging.info(f"  Using left value at ({i},{j}): {val}")
            return val
        val = Lup[i][bsize+j] if Lup is not None else 0 #up block
        logging.info(f"  Using up value at ({i},{j}): {val}")
        return val

    # Following steps build L[m+1][n+1] in bottom up fashion
    # Note: L[i][j] contains length of LCS of X[0..i-1]
    # and Y[0..j-1]
    #main algo
    for i in range(bsize):
        for j in range(bsize):
            left = bleft*bsize + i
            up = bup*bsize + j
            if left == 0 or up == 0:
                L[i][j] = 0
                logging.info(f"  Setting boundary value L[{i}][{j}] = 0")
            if X[left-1] == Y[up-1]:
                L[i][j] = l(i-1, j-1) + 1
                logging.info(f"  Characters match at ({left-1},{up-1}): L[{i}][{j}] = {L[i][j]}")
            else:
                L[i][j] = max(l(i-1, j), l(i, j-1))
                logging.info(f"  Characters don't match: L[{i}][{j}] = {L[i][j]}")

    logging.info(f"Completed block (bleft={bleft}, bup={bup})")
    return L


if __name__ == '__main__':
    print(" Starting LCS calculation...")
    logging.info("Starting LCS calculation...")
    logging.info(f"Input strings: length(S1)={len(S1)}, length(S2)={len(S2)}")

    m = len(S1)
    n = len(S2)
    
    assert m % bsize == 0
    assert n % bsize == 0
    print(f"Grid size will be: {m//bsize} x {n//bsize} blocks")

    X = ray.put(S1)
    Y = ray.put(S2)

    # First run
    print("\nFirst run starting...")
    f = [[None] * (n//bsize + 1) for i in range(m//bsize + 1)]
    start_time1 = time.time()
    
    for bleft in range(0, m//bsize):
        for bup in range(0, n//bsize):
            fleft = f[bleft-1][bup] if bleft > 0 else None
            fup = f[bleft][bup-1] if bup > 0 else None
            fdiag = f[bleft-1][bup-1] if bleft > 0 and bup > 0 else None
            f[bleft][bup] = lcs.remote(X, Y, bleft, bup, bsize, fleft, fup, fdiag)

    L1 = ray.get(f[m//bsize-1][n//bsize-1])
    elapsed1 = time.time() - start_time1
    print(f" First run result: {L1[bsize-1][bsize-1]}, Duration: {elapsed1:.2f}s")

    # Second run (should be using the cached results..hopefully)
    print("\nSecond run starting...")
    f = [[None] * (n//bsize + 1) for i in range(m//bsize + 1)]
    start_time2 = time.time()
    
    for bleft in range(0, m//bsize):
        for bup in range(0, n//bsize):
            fleft = f[bleft-1][bup] if bleft > 0 else None
            fup = f[bleft][bup-1] if bup > 0 else None
            fdiag = f[bleft-1][bup-1] if bleft > 0 and bup > 0 else None
            f[bleft][bup] = lcs.remote(X, Y, bleft, bup, bsize, fleft, fup, fdiag)

    L2 = ray.get(f[m//bsize-1][n//bsize-1])
    elapsed2 = time.time() - start_time2
    print(f" Second run result: {L2[bsize-1][bsize-1]}, Duration: {elapsed2:.2f}s")

    results_match = L1[bsize-1][bsize-1] == L2[bsize-1][bsize-1]
    speedup = elapsed1 / elapsed2 if elapsed2 > 0 else float('inf')
    
    print(f"\n Results match: {results_match}")
    print(f" Speedup factor: {speedup:.2f}x")

    print("\n Third run starting (with modified input)...")
    # Change the middle character of S1
    S1_modified = S1[:15] + 'B' + S1[16:]  
    X_modified = ray.put(S1_modified)
    
    f = [[None] * (n//bsize + 1) for i in range(m//bsize + 1)]
    start_time3 = time.time()
    
    for bleft in range(0, m//bsize):
        for bup in range(0, n//bsize):
            fleft = f[bleft-1][bup] if bleft > 0 else None
            fup = f[bleft][bup-1] if bup > 0 else None
            fdiag = f[bleft-1][bup-1] if bleft > 0 and bup > 0 else None
            f[bleft][bup] = lcs.remote(X_modified, Y, bleft, bup, bsize, fleft, fup, fdiag)

    L3 = ray.get(f[m//bsize-1][n//bsize-1])
    elapsed3 = time.time() - start_time3
    print(f" Third run result: {L3[bsize-1][bsize-1]}, Duration: {elapsed3:.2f}s")
    print(f" Change in result: {L2[bsize-1][bsize-1] - L3[bsize-1][bsize-1]}")
    print(f" Speedup vs first run: {elapsed1/elapsed3:.2f}x")