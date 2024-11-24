# Incremental Implementation of LCS with Ray

This document provides a detailed explanation of the incremental implementation of the Longest Common Subsequence (LCS) algorithm using Ray. The goal is to achieve significant performance benefits by caching and reusing intermediate results, especially when the input changes slightly.

## Overview

The LCS algorithm is implemented in a block-wise manner, where each block is computed independently and can be cached. This allows us to recompute only the necessary blocks when the input changes, leading to substantial performance improvements.

## Block-wise LCS Calculation

The input strings `S1` and `S2` are divided into blocks of size `bsize`. Each block is identified by its indices `(bleft, bup)`. The LCS calculation for each block depends on the values from the left, up, and diagonal blocks.

### Block Diagram
```
Before Change (all A's):
┌────────────┬────────────┬────────────┐
│ 0-9        │ 10-19      │ 20-29      │
│            │            │            │
│   Block 0,0│   Block 0,1│   Block 0,2│
├────────────┼────────────┼────────────┤
│            │            │            │
│   Block 1,0│   Block 1,1│   Block 1,2│
├────────────┼────────────┼────────────┤
│            │            │            │
│   Block 2,0│   Block 2,1│   Block 2,2│
└────────────┴────────────┴────────────┘
```
```
After Change (B at position 15):
┌────────────┬────────────┬────────────┐
│ UNCHANGED  │ UNCHANGED  │ UNCHANGED  │
│            │            │            │
│    CACHE   │    CACHE   │    CACHE   │
│     HIT    │     HIT    │     HIT    │
├────────────┼────────────┼────────────┤
│  CHANGED   │  CHANGED   │  CHANGED   │
│ Contains B │Depends on B│Depends on B│
│    MISS    │    MISS    │    MISS    │
├────────────┼────────────┼────────────┤
│ Depends on │ Depends on │ Depends on │
│  above     │  above     │  above     │
│    HIT     │    MISS    │    MISS    │
└────────────┴────────────┴────────────┘
```
### Change Propagation

When the input changes, only the blocks that depend on the changed part need to be recomputed. This is illustrated in the following diagram:
```
Change Propagation:
┌────────────┬────────────┬────────────┐
│            │            │            │
│            │            │            │
│            │     B      │            │
├────────────┼────────────┼────────────┤
│     ↓      │     ↓      │     ↓      │
│            │            │            │
│            │            │            │
├────────────┼────────────┼────────────┤
│            │     ↓      │     ↓      │
│            │            │            │
│            │            │            │
└────────────┴────────────┴────────────┘
```
## Performance Benefits

The incremental implementation provides significant performance benefits by caching and reusing intermediate results. The speedup pattern (13.68x → 7.30x) shows that we're achieving both:
- Maximum efficiency for unchanged inputs
- Reasonable performance for modified inputs by recomputing only necessary blocks

## Code Explanation

The main function `lcs` is decorated with `@ray.remote(incremental=True)`, indicating that it supports incremental computation. The function takes the input strings `X` and `Y`, block indices `(bleft, bup)`, block size `bsize`, and the left, up, and diagonal blocks as inputs. It returns the computed block at the given indices.

The main script initializes Ray with a storage path and sets up logging. It then performs three runs of the LCS calculation:
1. First run with the original input strings.
2. Second run with the same input strings to demonstrate caching.
3. Third run with a modified input string to show the incremental recomputation.

The results and performance metrics are printed for each run.

## Incremental Code in `remote_function.py`

The incremental functionality in Ray is implemented in the `remote_function.py` file. Here’s a breakdown of how it works:

- **Cache Key Generation**: A unique cache key is generated for each task based on its input data and dependencies. This is done using a combination of hashing the input data and the dependencies (left, up, and diagonal blocks).

- **Cache Retrieval and Storage**: Before executing a task, the system checks if a cached result exists for the generated cache key. If a cached result is found, it is returned immediately, avoiding redundant computation. If not, the task is executed, and the result is cached for future use.

- **Workflow Storage**: The caching mechanism uses a workflow storage system to save and retrieve task outputs. This ensures that results are persistent and can be reused across different runs.

- **Task Invocation**: The `@ray.remote` decorator is used to define remote functions. The `incremental` flag enables the caching mechanism, allowing tasks to benefit from incremental computation.

## Conclusion

The incremental implementation of the LCS algorithm using Ray demonstrates the power of caching and reusing intermediate results. By recomputing only the necessary blocks when the input changes, we achieve significant performance improvements while maintaining accuracy.


