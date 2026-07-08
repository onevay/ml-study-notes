# 📚 ml-study-notes

> A personal study repository: a path from mathematical foundations to advanced machine learning topics. Notes, presentations, code, and worked problems — all in one place, built so the material is useful not just to me, but to anyone walking a similar path.

![status](https://img.shields.io/badge/status-in%20progress-brightgreen)
![topics](https://img.shields.io/badge/topics-6-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 🧭 Why this repository exists

This isn't just a "file dump" — it's a structured learning system:

- 📝 **Notes** — theory in my own words, with formulas and examples.
- 🎞 **Presentation** — the same topic, condensed to its key ideas and visualizations.
- 💻 **Notebook** — the theory implemented by hand (often from scratch, without ready-made libraries).

This "notes → presentation → code" pipeline repeats for every topic — it's the core pattern of the whole repository.

---

## 🗂 Structure

| Section                                                 | Content                                                                                  |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| [`01-mathematics/`](01-mathematics/README.md)           | Linear algebra, calculus, probability and statistics — the mathematical foundation of ML |
| [`02-python/`](02-python/README.md)                     | Python from the ground up, data structures and algorithms                                |
| [`03-machine-learning/`](03-machine-learning/README.md) | The main track: from the perceptron to modern architectures                              |
| [`04-data-wrangling/`](04-data-wrangling/README.md)     | Working with data: cleaning, EDA, visualization, pandas/numpy                            |
| [`05-tasks/`](05-tasks/README.md)                       | Interesting math and programming problems with fully worked solutions                    |
| [`06-advanced/`](06-advanced/README.md)                 | Paper deep-dives and advanced topics for going further                                   |

```
ml-study-notes/
├── README.md
├── .gitignore
├── requirements.txt
│
├── 01-mathematics/
├── 02-python/
├── 03-machine-learning/
├── 04-data-wrangling/
├── 05-tasks/
├── 06-advanced/
└── assets/
    └── images/
```

Every **topic** inside a section follows the same template:

```
XX-topic-name/
├── notes.md            # hand-written theory notes
├── presentation.pdf     # a presentation based on notes.md
└── notebooks/
    └── 01-topic.ipynb   # practice: code, experiments, visualizations
```

---

## 🚦 Where to start reading

The material is arranged in increasing order of difficulty — suggested reading order:

1. **`01-mathematics`** — start here if linear algebra/calculus feel rusty.
2. **`02-python`** — the foundation for all the practical code in the repo.
3. **`03-machine-learning`** — the main track; topics go from simple to advanced.
4. **`04-data-wrangling`** — alongside ML, as needed for specific notebooks.
5. **`05-tasks`** — for reinforcement: practice problems mixed across topics.
6. **`06-advanced`** — once the basics feel solid, for going beyond the textbooks.

---

## ⚙️ Setup and reproducing the code

```bash
git clone https://github.com/<your-username>/ml-study-notes.git
cd ml-study-notes
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter lab
```

---

## 📌 Progress

- [x] Mathematics: linear algebra — fundamentals
- [x] ML: the perceptron and Adaline
- [ ] ML: logistic regression
- [ ] Data wrangling: pandas fundamentals
- [ ] Algorithms: sorting and complexity

_(This list is updated as work progresses — a kind of public learning log.)_

---

## 📖 Sources

Main books and courses this repository draws on:

- S. Raschka — _Machine Learning with PyTorch and Scikit-Learn_
- G. Strang — _Linear Algebra and Its Applications_
- Andrew Ng — _Machine Learning Specialization_ (Coursera)
- I. Goodfellow, Y. Bengio, A. Courville — _Deep Learning_

Topic-specific sources are also listed inside each topic's `notes.md`.

---

## 📝 License

Materials are distributed under the MIT license — feel free to use, adapt, and share, with attribution.
