# Outlier Detection and Removal Using IQR Method

**Source:** [Deep-ML Problem #355](https://www.deep-ml.com/problems/355)

**Difficulty:** Medium

**Category:** Data Preprocessing

## Problem

Detect and remove outliers using the Interquartile Range (IQR) method with a multiplier `k` (1.5 for outliers, 3.0 for extreme outliers).

## Solution

See `solution.py` for implementation using NumPy.

## Key Concepts

- Quartiles (Q1, Q3)
- Interquartile Range (IQR)
- Outlier thresholds: Q1 − k·IQR, Q3 + k·IQR
- Handling edge cases (all values within bounds)
