# Topic 4: Building Good Training Sets - Data Preprocessing

## Comprehensive Study Notes

### 4.0 The Primacy of Data Quality

> _"The quality of the data and the amount of useful information it contains are key factors that determine how well a machine learning algorithm can learn."_

In practice, data scientists spend **60–80% of their time** on preprocessing – not on modelling. This chapter establishes the essential techniques that separate robust ML systems from brittle ones.

#### The Preprocessing Pipeline (Conceptual)

```
Raw Data
    ↓
1. Handle Missing Values (detect, drop, or impute)
    ↓
2. Encode Categorical Features (ordinal mapping or one‑hot)
    ↓
3. Split into Train / Validation / Test Sets (stratified)
    ↓
4. Scale Numerical Features (standardisation or normalisation)
    ↓
5. Select Relevant Features (filter, wrapper, or embedded)
    ↓
Clean, Model‑Ready Data
```

---

### 4.1 Dealing with Missing Data

#### 4.1.1 Origins and Detection

Missing values arise from:

- Sensor failures
- Manual entry errors
- Survey respondents skipping questions
- Data corruption during transfer

In tabular data, missing values are usually represented as a special marker (e.g., `NaN`). The first step is to count missing entries per column to assess the extent of the problem.

#### 4.1.2 Three Fundamental Strategies

| Strategy           | Action                                                     | When to Apply                         |
| ------------------ | ---------------------------------------------------------- | ------------------------------------- |
| **Delete rows**    | Remove any sample containing at least one missing value    | Few missing values (<5% of rows)      |
| **Delete columns** | Remove an entire feature if it has too many missing values | >50% of values missing in that column |
| **Imputation**     | Replace missing entries with a statistical estimate        | Most flexible; retains all samples    |

#### Trade‑offs

| Strategy           | Advantages              | Disadvantages                                   |
| ------------------ | ----------------------- | ----------------------------------------------- |
| **Delete rows**    | Simple, fast            | Discards potentially valuable information       |
| **Delete columns** | Preserves complete rows | Loses an entire feature                         |
| **Imputation**     | Retains all data        | Introduces artificial values (adds uncertainty) |

#### 4.1.3 Imputation Methods – A Deeper Look

##### For Numerical Features

**Mean Imputation:**

$$
x_{\text{fill}} = \frac{1}{m} \sum_{i=1}^{m} x_i
$$

where only non‑missing values $x_i$ are used.

- **Pro:** Fast, works for normally distributed data
- **Con:** Sensitive to outliers

**Median Imputation:**

$$
x_{\text{fill}} = \text{median}(x_1, x_2, \ldots, x_m)
$$

- **Pro:** Robust to outliers
- **Con:** Ignores distribution shape

**Regression Imputation:** Predict missing values using other features (more advanced, requires careful implementation to avoid leakage).

##### For Categorical Features

**Mode Imputation:**

$$
x_{\text{fill}} = \text{most\_frequent}(x_1, x_2, \ldots, x_m)
$$

- **Pro:** Simple, works well when one category dominates
- **Con:** Can bias toward majority class

**Constant Imputation:** Replace with a new category like "unknown" to explicitly mark missingness.

#### Important Principle: Data Leakage

The imputation statistic (mean, median, mode) must be computed **only from the training set**, then applied to the test set. Using the whole dataset to compute statistics leaks information from the test set into training, leading to overly optimistic performance.

---

### 4.2 Handling Categorical Data

#### 4.2.1 Types of Categorical Variables

| Type        | Characteristic     | Example                       |
| ----------- | ------------------ | ----------------------------- |
| **Nominal** | No intrinsic order | Colour (red, blue, green)     |
| **Ordinal** | Meaningful order   | Size (small < medium < large) |

Most ML algorithms require numerical inputs. Categorical features must be encoded.

#### 4.2.2 Ordinal Encoding

For ordinal variables, preserve the order by mapping each category to an integer:

$$
\text{small} \rightarrow 1, \quad \text{medium} \rightarrow 2, \quad \text{large} \rightarrow 3
$$

This encoding respects the natural ordering and works well with linear models.

**Caution:** The mapping must reflect the true order; incorrect ordering misleads algorithms that assume numerical relationships.

#### 4.2.3 One‑Hot Encoding for Nominal Variables

One‑hot encoding creates binary (0/1) columns for each category:

| Color | Red | Blue | Green |
| ----- | --- | ---- | ----- |
| Red   | 1   | 0    | 0     |
| Blue  | 0   | 1    | 0     |
| Green | 0   | 0    | 1     |

For a feature with **k categories**, we create **k‑1** dummy variables to avoid multicollinearity (the "dummy variable trap").

**Why not label encoding for nominal?**

Assigning arbitrary integers (e.g., red=0, blue=1, green=2) implies a false order (red < blue < green). This can severely mislead linear models and distance‑based algorithms.

**Trade‑offs of one‑hot encoding:**

| Advantage                      | Disadvantage                                               |
| ------------------------------ | ---------------------------------------------------------- |
| No artificial ordering imposed | Increases dimensionality                                   |
| Works with any model           | Can cause curse of dimensionality when cardinality is high |

#### 4.2.4 Encoding the Target Variable

For the target (response) variable, label encoding (integer mapping) is safe because the target is not used in distance or gradient calculations. It is simply a way to convert class labels into numerical form for loss functions.

---

### 4.3 Partitioning Datasets

#### 4.3.1 The Three‑Set Paradigm

To build a reliable model, we need three distinct data splits:

| Set            | Purpose                                        | Typical Proportion |
| -------------- | ---------------------------------------------- | ------------------ |
| **Training**   | Learn model parameters (weights)               | 60–80%             |
| **Validation** | Tune hyperparameters and select models         | 10–20%             |
| **Test**       | Provide an unbiased final performance estimate | 10–20%             |

**Golden Rule:** Never look at the test set until the final model is chosen. All development (including feature engineering, scaling, and hyperparameter tuning) must be done using only training and validation data.

#### 4.3.2 Stratified Splitting

When the target classes are imbalanced, a simple random split can create training and test sets with different class proportions. **Stratification** ensures that each split preserves the original class distribution.

**Why it matters:** Without stratification, your test set might have a different class ratio, leading to misleading accuracy estimates.

#### 4.3.3 Data Leakage – The Hidden Danger

Data leakage occurs when information from outside the training set influences the training process.

**Common sources:**

| Source                     | Example                                                  |
| -------------------------- | -------------------------------------------------------- |
| **Scaling**                | Using global mean/std instead of training‑set statistics |
| **Imputation**             | Computing imputation values from the entire dataset      |
| **Feature selection**      | Selecting features based on the full dataset             |
| **Validation information** | Using validation results to guide preprocessing          |

**Prevention:** Always perform data‑dependent operations (scaling, imputation, feature selection) **inside** the training loop, using only training data to fit transformers.

---

### 4.4 Feature Scaling

#### 4.4.1 Why Scaling Is Necessary

Two classes of algorithms are particularly sensitive to feature scales:

1. **Distance‑based models** (KNN, SVM with RBF kernel, clustering):
   - Features with larger ranges dominate the distance calculation
   - Effectively ignores smaller‑scale features

2. **Gradient‑based models** (linear regression, logistic regression, neural networks):
   - Unscaled features create an elongated, "valley‑shaped" loss landscape
   - Gradient descent zig‑zags along the valley, leading to slow convergence

**The "Valley" Problem:** If $x_1 \in [0, 100000]$ and $x_2 \in [0, 100]$, the loss contours become highly elongated. Gradient descent zig‑zags along the valley, slowing convergence.

#### 4.4.2 Standardisation vs. Normalisation

| Method                        | Formula                             | Output Range      | Best Suited For                                                                   |
| ----------------------------- | ----------------------------------- | ----------------- | --------------------------------------------------------------------------------- |
| **Standardisation (Z‑score)** | $x' = \frac{x - \mu}{\sigma}$       | Mean = 0, Std = 1 | Most cases; works with outliers (moderately)                                      |
| **Min‑Max Scaling**           | $x' = \frac{x - \min}{\max - \min}$ | [0, 1]            | Algorithms that assume bounded inputs (e.g., neural nets) – sensitive to outliers |

**Which to choose?**

- **Standardisation** is the default choice for SVM, logistic regression, and linear models.
- **Min‑Max** is useful when you need a fixed range, but outliers can compress the rest of the data.

**Critical Rule:** The scaler must be fitted **on the training set only**; the same fitted scaler is then used to transform the test set.

#### 4.4.3 When to Skip Scaling

| Algorithm           | Scaling Required? |
| ------------------- | ----------------- |
| Logistic Regression | Yes               |
| SVM                 | Yes               |
| KNN                 | Yes (critical)    |
| Decision Trees      | No                |
| Random Forest       | No                |
| Naive Bayes         | No                |

---

### 4.5 Selecting Relevant Features

#### 4.5.1 Why Reduce Dimensionality?

| Benefit                               | Description                                                                           |
| ------------------------------------- | ------------------------------------------------------------------------------------- |
| **Prevents overfitting**              | Simpler models generalise better                                                      |
| **Improves interpretability**         | Fewer features are easier to explain                                                  |
| **Speeds up training and prediction** | Less computation                                                                      |
| **Mitigates curse of dimensionality** | With many features, data becomes sparse and models require exponentially more samples |

#### 4.5.2 Three Families of Feature Selection Methods

| Family       | Approach                                         | Examples                                 | Pros / Cons                                          |
| ------------ | ------------------------------------------------ | ---------------------------------------- | ---------------------------------------------------- |
| **Filter**   | Statistical measures independent of any model    | Correlation, χ² test, mutual information | Fast, but ignore feature interactions                |
| **Wrapper**  | Use a model to evaluate subsets                  | Forward/backward selection, RFE          | Consider interactions, but computationally expensive |
| **Embedded** | Feature selection integrated into model training | L1 regularisation, tree‑based importance | Efficient, capture interactions, but model‑specific  |

#### 4.5.3 Recursive Feature Elimination (RFE)

**How it works:**

1. Train a model on all features
2. Rank features by importance (coefficients or feature*importances*)
3. Remove the least important feature(s)
4. Repeat until the desired number of features remains

**Strength:** Considers feature combinations because the model sees all features together at each iteration.

**Weakness:** Can be slow for large feature sets.

#### 4.5.4 Embedded Methods: L1 Regularisation

L1 penalty adds the sum of absolute weights to the loss function:

$$
J(\mathbf{w}) = \text{loss}(\mathbf{w}) + \lambda \sum_{j=1}^{m} |w_j|
$$

This forces some weights to become exactly zero, effectively performing feature selection.

**Advantages:** Efficient, stable, works well in high dimensions.

**In scikit‑learn:** `LogisticRegression(penalty='l1', solver='saga')`

---

### 4.6 Random Forest Feature Importance

#### 4.6.1 The Underlying Mechanism

In a Random Forest, each tree is built by recursively splitting nodes. At each split, the algorithm chooses the feature and threshold that most reduce impurity (Gini or entropy).

**Calculation:**

1. For each tree, compute the total impurity reduction from splits using each feature
2. Average these reductions across all trees
3. Normalise so all importances sum to 1

$$
\text{Importance}(f) = \frac{\sum_{t=1}^{T} \text{ImpurityReduction}_{t}(f)}{\sum_{g=1}^{P} \sum_{t=1}^{T} \text{ImpurityReduction}_{t}(g)}
$$

**Interpretation:** A higher importance value means the feature contributes more to reducing uncertainty.

**Advantages:**

- Captures non‑linear relationships without any distributional assumptions
- Works with mixed data types (after encoding)
- Does not require scaling
- Provides a model‑agnostic view of feature relevance

**Caveats:**

| Caveat                           | Explanation                                                      |
| -------------------------------- | ---------------------------------------------------------------- |
| **Bias toward high‑cardinality** | Features with more distinct values tend to get higher importance |
| **Correlated features**          | Importance can be shared among correlated features               |
| **Unstable estimates**           | Use at least 100 trees for stable results                        |

#### 4.6.2 Using Importance for Feature Selection

After obtaining importances, you can:

1. **Rank features** and keep the top _k_
2. **Set a threshold** and discard features with importance below that threshold
3. **Use as input to RFE** for a more refined selection

---

### 4.7 The Preprocessing Pipeline – Conceptual Summary

| Step                       | Action                                         | Key Principle                                |
| -------------------------- | ---------------------------------------------- | -------------------------------------------- |
| **1. Detect missing data** | Understand the extent and patterns             | Count missing values per column              |
| **2. Decide on handling**  | Drop or impute                                 | Imputation retains data; dropping is simpler |
| **3. Encode categorical**  | Ordinal mapping or one‑hot                     | Ordinal for ordered, one‑hot for nominal     |
| **4. Split data**          | Train/validation/test                          | Stratify if classes are imbalanced           |
| **5. Scale features**      | Standardisation or normalisation               | Fit on training, transform all               |
| **6. Select features**     | Filter, wrapper, or embedded                   | Cross‑validate to ensure generalisation      |
| **7. Train and evaluate**  | Fit model, tune on validation, final test once | Never touch test set during development      |

**Golden Thread:** Every preprocessing step that depends on data statistics must be fitted on the training set and then applied to validation/test sets – never the reverse.

---

### 4.8 Key Takeaways

| Topic                    | Key Insight                                                                     |
| ------------------------ | ------------------------------------------------------------------------------- |
| **Missing data**         | Impute wisely; the choice of statistic (mean, median, mode) matters             |
| **Categorical encoding** | Respect the nature of the variable – ordinal vs. nominal                        |
| **Splitting**            | Stratify when classes are imbalanced; use three sets for proper evaluation      |
| **Scaling**              | Standardise unless you have a specific reason not to                            |
| **Feature selection**    | Use RFE for accuracy, L1 for efficiency, tree importance for non‑linear insight |
| **Data leakage**         | Always fit transformers on training data only                                   |

---

### 4.9 Practice Problems

#### 🟢 Beginner

1. Load the Titanic dataset. List columns with missing values and their percentage.
2. Implement mean and median imputation from scratch (no sklearn) on a small array.
3. Train KNN on the Iris dataset with and without scaling. Quantify the accuracy difference.

#### 🟡 Intermediate

4. Compare one‑hot vs. label encoding on a nominal feature with 5 categories using logistic regression.
5. Use RFE, L1, and Random Forest importance on the same dataset. Compare the selected features.

#### 🔴 Advanced

6. Build a complete sklearn `Pipeline` that imputes, scales, selects features, and trains a classifier. Use `GridSearchCV` to tune all hyperparameters.
7. Apply the full preprocessing pipeline to the Titanic dataset and submit predictions to Kaggle.

---

### 4.10 References

- **Primary Source:** Raschka, S., Liu, Y. H., & Mirjalili, V. (2022). _Machine Learning with PyTorch and Scikit‑Learn_. Packt Publishing. Chapter 4.
- **Scikit‑Learn Documentation:**
  - [Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html)
  - [Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
  - [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

- **Official Repository:** [Chapter 4 Notebooks](https://github.com/rasbt/machine-learning-book/tree/main/ch04)
