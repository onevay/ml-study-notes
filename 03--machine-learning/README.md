# 🤖 03 · Machine Learning

> The main track of the repository. Topics follow a historically and logically sequential line: from the simplest artificial neuron to the ideas underlying modern architectures — arranged so each topic solves a problem raised by the one before it.

[⬅ Back to the repository index](../README.md)

---

## Philosophy of this section

Every topic here is broken down on three levels at once:

1. **Understand it by hand** — implement the algorithm from scratch in NumPy, without `sklearn`, so nothing feels like a black box.
2. **Understand it visually** — a presentation visualizing the key idea (e.g., the gradient descent trajectory on the loss surface).
3. **Understand it in words** — a full written summary that can be reread months later to reconstruct the whole line of reasoning.

## Topics

| #   | Topic                                                                                        | Status | Materials                                                                |
| --- | -------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------ |
| 1   | The perceptron and Adaline: the birth of gradient-based learning                             | ✅     | [`01-perceptron-adaline/`](01-perceptron-adaline/notes.md)               |
| 2   | The classification algorithms. Theory and implementation from scratch and using scikit-learn | ✅     | [`02-classification-algorithms/`](02-classification-algorithms/notes.md) |
| 3   | Data Preprocessing                                                                           | ✅     | [`03-data-preprocessing/`](03-data-preprocessing/notes.md) |

✅ — done · ⏳ — in progress / planned

Deeper, more specialized topics (transformer architectures, paper deep-dives, etc.) live in a separate section, [`06-advanced/`](../06-advanced/README.md), so they don't disrupt the sequential flow of the main track.

## Topic structure — using the finished Topic 1 as an example

```
01-perceptron-adaline/
├── notes.md                                  # full theoretical write-up
├── presentation.pdf                           # a presentation based on notes.md
└── notebooks/
    ├── 01-perceptron-from-scratch.ipynb       # perceptron in NumPy + an XOR example (where it fails to converge)
    └── 02-adaline-and-feature-scaling.ipynb    # Adaline, MSE, why feature scaling matters
```

Every `notes.md` ends with a **"References"** section linking back to the presentation and notebooks for that same topic — so the topic stays one coherent whole, whichever file you happen to open first.

## How to read this section

It's best to go through the topics in order — starting from topic 2, almost every one builds on concepts introduced earlier (e.g., logistic regression is essentially Adaline with a different activation and loss function). If a topic already feels familiar, feel free to jump straight to `notebooks/` and come back to `notes.md` as a reference.

## Sources

- S. Raschka — _Machine Learning with PyTorch and Scikit-Learn_
- Andrew Ng — _Machine Learning Specialization_ (Coursera)
- I. Goodfellow, Y. Bengio, A. Courville — _Deep Learning_
