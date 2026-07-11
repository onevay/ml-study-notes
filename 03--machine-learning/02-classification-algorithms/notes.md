# Chapter 3: A Tour of Machine Learning Classifiers Using Scikit-Learn

## Comprehensive Study Notes (Handwritten Format)

---

## 3.0 Introduction: Why This Chapter Matters

After implementing perceptron and Adaline from scratch in Chapter 2, we now enter the **real world of machine learning**. In practice, you almost never implement algorithms from scratch — you use battle-tested libraries. **Scikit-learn** is the gold standard for classical ML in Python.

**Why scikit-learn?**

- **Consistent API**: Every model follows the same pattern: `.fit()`, `.predict()`, `.score()`
- **Optimized**: Written in C/C++ under the hood — much faster than pure Python
- **Production-ready**: Used by thousands of companies
- **Extensive documentation**: One of the best in the open-source world

This chapter covers **5 essential classifiers** that form the foundation of modern ML:

| Algorithm                     | Type              | When to Use                                    |
| ----------------------------- | ----------------- | ---------------------------------------------- |
| Logistic Regression           | Linear            | Baseline, interpretable, probabilistic outputs |
| SVM (Support Vector Machines) | Linear/Non-linear | Powerful boundaries, works in high dimensions  |
| Decision Trees                | Non-linear        | Highly interpretable, no scaling needed        |
| K-Nearest Neighbors           | Non-parametric    | Simple, no training, good for small datasets   |

---

## 3.1 The Scikit-Learn API: Understanding the Pattern

Before diving into individual algorithms, you MUST understand the **unified API** that scikit-learn uses. Every estimator (classifier or regressor) follows the same 3-step pattern:

```python
# Step 1: Instantiate the model (set hyperparameters)
model = AlgorithmName(param1=value1, param2=value2)

# Step 2: Train the model on data
model.fit(X_train, y_train)

# Step 3: Make predictions
predictions = model.predict(X_test)
```

**Why is this important?** Once you know how to use one scikit-learn model, you know how to use ALL of them. This consistency is what makes scikit-learn so powerful.

**Key methods you'll use constantly:**

| Method              | Purpose                                             | Example                               |
| ------------------- | --------------------------------------------------- | ------------------------------------- |
| `.fit(X, y)`        | Train the model                                     | `model.fit(X_train, y_train)`         |
| `.predict(X)`       | Make predictions                                    | `y_pred = model.predict(X_test)`      |
| `.score(X, y)`      | Return accuracy (classification) or R² (regression) | `acc = model.score(X_test, y_test)`   |
| `.predict_proba(X)` | Return class probabilities (if available)           | `probs = model.predict_proba(X_test)` |

---

## 3.2 Logistic Regression: Modeling Probabilities

### 3.2.1 Why Logistic Regression Matters

Remember the **perceptron's limitation** from Chapter 2? It only converges if classes are perfectly linearly separable. Logistic regression fixes this by modeling **probabilities** instead of hard class boundaries.

**The core idea**: Instead of asking "Is this class 1 or 0?", we ask "What's the probability this belongs to class 1?" This gives us:

- **Confidence** in predictions (not just yes/no)
- **Smooth, differentiable loss function** → works with gradient descent
- **Probabilistic interpretation** → useful for decision-making

### 3.2.2 The Sigmoid Function (The Heart of Logistic Regression)

The sigmoid (or logistic) function transforms any real number into a value between 0 and 1:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \mathbf{w}^T\mathbf{x} + b
$$

**Let's break this down**:

- **z**: weighted sum (same as perceptron/Adaline)
- **e^{-z}**: exponential decay — as z increases, e^{-z} → 0, so σ(z) → 1
- **σ(z)**: probability that the sample belongs to class 1

**Key properties to memorize**:

| Property                  | Value  | Why It Matters                                  |
| ------------------------- | ------ | ----------------------------------------------- |
| Output range              | (0, 1) | Can be interpreted as probability               |
| σ(0)                      | 0.5    | Decision boundary — both classes equally likely |
| σ(z) → 1 as z → ∞         | 1      | Very high confidence in class 1                 |
| σ(z) → 0 as z → -∞        | 0      | Very high confidence in class 0                 |
| Differentiable everywhere | Yes    | Can use gradient descent                        |

**Decision rule**: Predict class 1 if σ(z) ≥ 0.5, otherwise class 0. This is equivalent to z ≥ 0.

### 3.2.3 The Log-Loss Function (Cross-Entropy)

For perceptron, we used classification error. For Adaline, we used MSE. For logistic regression, we use **log-loss** (also called binary cross-entropy):

$$
J(\mathbf{w}) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log\left(\sigma(z^{(i)})\right) + (1-y^{(i)}) \log\left(1-\sigma(z^{(i)})\right) \right]
$$

**Why this specific function?**

Let's understand it intuitively by looking at two cases:

**Case 1: True label y = 1**

- Loss = $-\log(\sigma(z))$
- If prediction is correct (σ(z) → 1): loss = $-\log(1) = 0$ → no penalty
- If prediction is wrong (σ(z) → 0): loss = $-\log(0) \to \infty$ → huge penalty

**Case 2: True label y = 0**

- Loss = $-\log(1-\sigma(z))$
- If prediction is correct (σ(z) → 0): loss = $-\log(1) = 0$ → no penalty
- If prediction is wrong (σ(z) → 1): loss = $-\log(0) \to \infty$ → huge penalty

**Key insight**: Log-loss **heavily penalizes confident wrong predictions**. If you're 99% sure it's class 1 but it's actually class 0, you get a massive penalty. This encourages the model to be humble.

**Why not use MSE?** For logistic regression, MSE is **non-convex** — it has multiple local minima. Log-loss is **convex** — one global minimum. With convex loss, gradient descent will always find the optimal solution.

### 3.2.4 Comparison with Adaline (Why the Change?)

| Component                 | Adaline      | Logistic Regression  |
| ------------------------- | ------------ | -------------------- |
| **Linear output**         | z = wᵀx + b  | z = wᵀx + b          |
| **Activation (training)** | identity (z) | sigmoid (σ(z))       |
| **Loss function**         | MSE          | Log-loss             |
| **Output type**           | Continuous   | Probability (0 to 1) |
| **Prediction**            | sign(z)      | σ(z) ≥ 0.5           |

**Visualization**: Think of Adaline as fitting a line through data points (regression). Logistic regression fits a **probability curve** — the sigmoid squashes the line into an S-shape.

### 3.2.5 Regularization: Preventing Overfitting

**The problem**: If we have too many features or the model is too complex, logistic regression can **overfit** — it memorizes the training data but fails on new data.

**The solution**: Add a **penalty term** to the loss function that discourages large weights. This is called **regularization**.

**L2 Regularization** (most common):

$$
J(\mathbf{w}) = \text{log-loss} + \lambda \|\mathbf{w}\|_2^2 = \text{log-loss} + \lambda \sum_{j=1}^{m} w_j^2
$$

**Why this works:**

- Large weights → large penalty → model reduces weights
- Smaller weights → simpler model → better generalization
- The λ (lambda) controls the strength of regularization

**In scikit-learn, λ is represented by C**:

$$
C = \frac{1}{\lambda}
$$

| C value                  | Regularization | Effect                     |
| ------------------------ | -------------- | -------------------------- |
| **Large C** (e.g., 10)   | Weak           | Complex model, may overfit |
| **Small C** (e.g., 0.01) | Strong         | Simple model, may underfit |
| **C = 1**                | Default        | Balanced                   |

**Key insight**: Always tune C using cross-validation. The optimal value depends on your data.

**L1 Regularization** (alternative):

- Penalizes sum of absolute weights: $\lambda \sum |w_j|$
- Leads to **sparse** weights (some become exactly 0) → feature selection
- Good when you have many irrelevant features

---

## 3.3 Support Vector Machines (SVM): Maximum Margin Classification

### 3.3.1 The Core Idea

While logistic regression models probabilities, SVM takes a different approach: **find the decision boundary that maximizes the distance (margin) between classes**.

**Why maximize margin?** A larger margin generally leads to better generalization on unseen data. Think of it as creating a "safety buffer" between the classes.

**Terminology to memorize**:

- **Support vectors**: The data points closest to the decision boundary that define the margin
- **Margin**: The distance between the decision boundary and the support vectors
- **Hyperplane**: The decision boundary (in 2D, it's a line)

### 3.3.2 Hard-Margin SVM (Linearly Separable Data)

**Goal**: Find the hyperplane that:

1. Correctly separates ALL training samples
2. Maximizes the margin (distance to the closest points)

**Mathematical formulation**:

For a hyperplane defined by $\mathbf{w}^T\mathbf{x} + b = 0$:

- **Margin** = $\frac{2}{\|\mathbf{w}\|}$
- To maximize margin → **minimize** $\|\mathbf{w}\|$
- Subject to: $y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) \ge 1$ for all i

**Why the constraint?** This ensures that:

- All points are correctly classified
- Support vectors lie exactly at distance 1 from the boundary

**When does this work?** Only when data is **perfectly linearly separable**. If not, there's no solution.

### 3.3.3 Soft-Margin SVM (Handling Non-Separable Data)

In real life, data is rarely perfectly linearly separable. Soft-margin SVM addresses this by **allowing some violations** of the margin.

**Introducing slack variables** $\xi_i \ge 0$:

$$
\text{Minimize } \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i
$$

- **Subject to**: $y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) \ge 1 - \xi_i$
- $\xi_i$ measures how far point i violates the margin
- **C**: hyperparameter that controls the trade-off

| C Value                     | Effect                       | When to Use                          |
| --------------------------- | ---------------------------- | ------------------------------------ |
| **Very large (e.g., 100)**  | Hard margin, few violations  | Data is clean and linearly separable |
| **Very small (e.g., 0.01)** | Soft margin, many violations | Data has outliers or noise           |
| **Default (1.0)**           | Balanced                     | Most cases                           |

**Visualization**: Think of C as the "slack budget." Large C means you're strict — you'll spend budget to fit every point. Small C means you're lenient — you'll accept misclassifications for a simpler model.

### 3.3.4 The Kernel Trick: SVM's Secret Weapon

**The problem**: What if data is non-linearly separable even with soft margin? (e.g., XOR problem, concentric circles)

**The solution**: Transform data into a **higher-dimensional space** where it becomes linearly separable.

**Example**:

- Original: 2D data → non-linear boundary needed
- Transform: Use polynomial features → map to 3D space
- In 3D: Data becomes linearly separable with a plane
- Transform back → non-linear boundary in original 2D space

**The "trick"**: We don't actually compute the transformation. We use a **kernel function** that computes the dot product in the transformed space **without explicitly doing the transformation**.

**Common kernels**:

| Kernel             | Formula                                           | Use Case                                |
| ------------------ | ------------------------------------------------- | --------------------------------------- |
| **Linear**         | $\mathbf{x}_i^T\mathbf{x}_j$                      | High-dimensional data, text             |
| **Polynomial**     | $(\gamma \mathbf{x}_i^T\mathbf{x}_j + r)^d$       | Moderately complex boundaries           |
| **RBF (Gaussian)** | $\exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$ | Most complex boundaries, default choice |

**RBF kernel intuition**: It measures similarity between points. Similar points (close together) have high kernel values, pushing the decision boundary to separate them.

**Key parameters for RBF**:

- **γ (gamma)**: Controls the influence of a single training point
  - Large γ → narrow influence, more complex boundaries (may overfit)
  - Small γ → wide influence, simpler boundaries (may underfit)

---

## 3.4 Decision Trees: Interpretable and Intuitive

### 3.4.1 How Decision Trees Work

Decision trees mimic human decision-making: ask a series of questions about features, each question splitting the data until you reach a decision.

**Example**: "Will I enjoy this movie?"

- Is it a comedy? → If Yes: Do I like the lead actor? → If Yes: Watch it!
- If No: Is it an action movie? → ...

**Formally**, a decision tree:

1. **Starts at the root** with all training data
2. **Finds the best feature and threshold** to split the data (maximizing information gain)
3. **Splits the data** into child nodes
4. **Recursively repeats** until a stopping criterion is met

### 3.4.2 Splitting Criteria: How Trees Decide

The tree needs to know which split is "best." It uses **impurity measures** to evaluate splits.

**Goal**: Make each child node as **pure** as possible (contain mostly samples from one class).

**Two most common impurity measures**:

**1. Gini Impurity** (default in scikit-learn):

$$
Gini(t) = 1 - \sum_{k=1}^{K} p_k^2
$$

where $p_k$ is the proportion of samples in class k at node t.

**Interpretation**:

- If node is perfectly pure (all class 0): Gini = 1 - 1² = 0
- If node is perfectly mixed (50/50 for two classes): Gini = 1 - (0.5² + 0.5²) = 0.5
- Lower Gini = better split

**2. Entropy** (another option):

$$
Entropy(t) = -\sum_{k=1}^{K} p_k \log_2(p_k)
$$

**Interpretation**:

- Pure node: Entropy = 0
- Mixed node (50/50): Entropy = 1 (bits)
- Lower entropy = better split

**Comparison**:

- Gini is slightly faster to compute
- Entropy sometimes produces more balanced trees
- In practice, both give similar results

### 3.4.3 Advantages and Disadvantages

**Advantages**:

- **Highly interpretable**: You can visualize and explain every decision
- **No feature scaling needed**: Works with raw data
- **Handles mixed data types**: Both numerical and categorical features
- **Non-linear boundaries**: Can model complex relationships
- **Feature importance**: Built-in feature ranking

**Disadvantages**:

- **Prone to overfitting**: Without pruning, trees can memorize the training data
- **Unstable**: Small changes in data can create entirely different trees
- **Greedy algorithm**: May not find the globally optimal tree
- **Bias**: Prefers features with more levels (categorical with many categories)

### 3.4.4 Controlling Overfitting: Tree Pruning

**The problem**: A fully grown tree will perfectly classify training data but fail on test data.

**Solutions in scikit-learn**:

| Parameter           | What It Does                          | Default | Recommended               |
| ------------------- | ------------------------------------- | ------- | ------------------------- |
| `max_depth`         | Maximum depth of the tree             | None    | 3-10 for interpretability |
| `min_samples_split` | Minimum samples to split a node       | 2       | 5-20 for larger datasets  |
| `min_samples_leaf`  | Minimum samples in a leaf node        | 1       | 5-20                      |
| `max_features`      | Max features considered for splitting | None    | sqrt(n_features)          |

**Rule of thumb**: Always limit `max_depth` to prevent overfitting. Start with 3-5 and increase if needed.

---

## 3.5 K-Nearest Neighbors (KNN): The Lazy Learner

### 3.5.1 What Makes KNN Different

All previous algorithms (logistic regression, SVM, decision trees) are **eager learners** — they build a model during training and discard the training data. KNN is a **lazy learner**:

- **No training phase**: It simply stores all training data
- **No model building**: It makes predictions at query time by looking at neighbors

**Implication**:

- Training is instant (just store data)
- Prediction can be slow (compute distances to all training points)

### 3.5.2 The KNN Algorithm

**Training**: Just store $X_{train}$ and $y_{train}$

**Prediction**: For a new sample x:

1. **Compute distance** to every training sample
2. **Find k closest samples** (nearest neighbors)
3. **Take majority vote** of their labels (for classification)
4. **Return the class** with the most votes

**Distance metrics** (most common):

| Metric        | Formula                                         | When to Use                                           |
| ------------- | ----------------------------------------------- | ----------------------------------------------------- |
| **Euclidean** | $d(x,y) = \sqrt{\sum_i (x_i - y_i)^2}$          | Default, works well in most cases                     |
| **Manhattan** | $d(x,y) = \sum_i \mid x_i - y_i \mid$           | High-dimensional data                                 |
| **Minkowski** | $d(x,y) = (\sum_i \mid x_i - y_i \mid^p)^{1/p}$ | Generalization of Euclidean (p=2) and Manhattan (p=1) |
| **Cosine**    | $d(x,y) = 1 - \frac{x \cdot y}{\|x\|\|y\|}$     | Text data, high dimensions                            |

### 3.5.3 Choosing the Right k

The choice of $k$ is the most important hyperparameter:

| Small k (e.g., 1-3)          | Large k (e.g., 10-20)        |
| ---------------------------- | ---------------------------- |
| Low bias, high variance      | High bias, low variance      |
| Sensitive to noise           | Smoother decision boundaries |
| Decision boundary is complex | Decision boundary is simple  |
| May overfit                  | May underfit                 |

**Visualization**: Think of k as the size of the voting committee. Small committee = wild decisions, large committee = conservative decisions.

**Practical rule**:

- k should be **odd** (to avoid ties)
- k should be **$\sqrt{n}$** (roughly, where n = number of training samples)
- Use **cross-validation** to find the optimal k

**Typical values**: 3, 5, 7, 9, 11, 15

### 3.5.4 The Feature Scaling Requirement

**CRITICAL**: KNN is distance-based. Features with larger scales will dominate the distance calculation.

**Example**:

- Feature 1: Age (0-100)
- Feature 2: Salary (0-100,000)

Without scaling: Salary variation completely dominates Age variation.

**Solution**: Always **standardize** features before KNN:

$$
x'_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
$$

This ensures all features contribute equally to distance calculations.

**Complexity considerations**:

- **Training**: O(1) — just store data
- **Prediction**: O(n·d) — for each sample, compute distance to all n training points in d dimensions
- For large datasets, consider **KD-trees** or **Ball trees** (scikit-learn supports these)

---

## 3.6 Algorithm Comparison: When to Use What?

### 3.6.1 Comprehensive Comparison Table

| Aspect                   | Logistic Regression | SVM (Linear) | SVM (RBF) | Decision Tree  | KNN                    |
| ------------------------ | ------------------- | ------------ | --------- | -------------- | ---------------------- |
| **Interpretability**     | High                | Medium       | Low       | Very High      | Low                    |
| **Training Time**        | Fast                | Fast         | Medium    | Fast           | None                   |
| **Prediction Time**      | Fast                | Fast         | Medium    | Fast           | Slow                   |
| **Scaling Required**     | Yes                 | Yes          | Yes       | No             | Yes (critical)         |
| **Handles Non-Linear**   | No                  | No           | Yes       | Yes            | Yes                    |
| **Works with High Dim**  | Yes                 | Yes          | Yes       | Yes            | No (curse of dim)      |
| **Handles Outliers**     | Moderate            | Good         | Good      | Good           | Poor                   |
| **Feature Importance**   | Yes (coefficients)  | Yes (linear) | No        | Yes (built-in) | No                     |
| **Probabilistic Output** | Yes                 | No           | No        | No             | No (can use distances) |

### 3.6.2 Decision Rules for Choosing Algorithms

**Start with logistic regression** (baseline):

- Interpretable
- Fast
- Works surprisingly well
- Gives you a baseline to beat

**Use SVM if:**

- Data has complex non-linear patterns → RBF SVM
- High-dimensional feature space → Linear SVM
- Clear margin separation → Any SVM
- Black box is acceptable → RBF SVM

**Use Decision Trees if:**

- Interpretability is crucial (explain to stakeholders)
- No time for feature scaling
- Want to understand feature importance
- Need to handle mixed data types

**Use KNN if:**

- Small dataset (less than 1000 samples)
- Classes have clear clusters
- Training time must be instant
- You need a simple, intuitive model

**Avoid KNN if:**

- Dataset is large (slow predictions)
- High-dimensional data (distance breaks down)
- Need fast real-time predictions

---

## 3.7 Key Takeaways for Your Learning Journey

### 3.7.1 The Most Important Lessons

1. **Always start simple**: Logistic regression should be your first try. It's fast, interpretable, and gives you a baseline.

2. **Feature scaling is critical for distance-based models**: KNN and SVM rely on distance metrics. Always standardize features.

3. **Regularization prevents overfitting**: Use C to control model complexity for logistic regression and SVM. Small C = simple model, large C = complex model.

4. **Decision trees are your "explainability" tool**: When you need to explain your model to non-technical stakeholders, use decision trees.

5. **There's no free lunch**: No algorithm works best for all problems. Experiment, compare, and evaluate.

### 3.7.2 The 5-Step Workflow for Any ML Problem

1. **Load and explore data**: Understand features, distributions, and relationships
2. **Preprocess data**: Handle missing values, encode categorical, scale features
3. **Split data**: Train/test split (or cross-validation)
4. **Train multiple models**: Try 3-5 different algorithms
5. **Evaluate and tune**: Compare performance, tune hyperparameters, select best model

---

## 3.8 Practice Problems (For Your Notebook)

### Beginner Level

1. **Iris Classification**: Train all 5 classifiers on Iris dataset. Which performs best? Why?
2. **Feature Scaling Experiment**: Train KNN on scaled vs unscaled data. What difference do you see?
3. **Interpretation Exercise**: Print coefficients of logistic regression. Which features are most important?

### Intermediate Level

4. **Cross-Validation**: Perform 5-fold CV for each model. Report mean and standard deviation of accuracy.
5. **Hyperparameter Tuning**: Use GridSearchCV to find optimal C for logistic regression and SVM.
6. **Decision Tree Pruning**: Vary max_depth from 1 to 20. Plot training vs test accuracy. Where is the optimal depth?

### Advanced Level

7. **Pipeline Construction**: Build a Pipeline that scales features, selects top features, and trains a model. Use GridSearchCV to tune all steps.
8. **SVM Kernel Comparison**: Generate synthetic non-linear data (concentric circles). Compare linear, polynomial, and RBF kernels visually.
9. **Feature Importance**: Train a decision tree, extract feature importance, and compare with logistic regression coefficients.

### Kaggle Challenge

10. **Titanic**: Apply all classifiers to the Titanic dataset. Engineer features, handle missing values, and tune hyperparameters. Which model performs best? Submit predictions to Kaggle.

---

## 📚 References and Further Reading

**Primary Source**:

- Raschka, S., Liu, Y. H., & Mirjalili, V. (2022). _Machine Learning with PyTorch and Scikit‑Learn_. Packt Publishing.

**Official Documentation**:

- [Scikit-Learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Logistic Regression API Reference](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [SVM API Reference](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [Decision Tree API Reference](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [KNN API Reference](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

**Additional Resources**:

- [Scikit-Learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html)
- [Kaggle Courses](https://www.kaggle.com/learn)
- [Raschka's Official Repository](https://github.com/rasbt/machine-learning-book)notebooks and Jupyter implementations for comprehensive learning.\*
