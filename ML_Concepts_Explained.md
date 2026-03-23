# Supervised Learning Concepts Explained: Hallucination Classifier

Welcome to the beginner-friendly guide to your Machine Learning project! This document covers every major concept used across all three notebooks and explains how they connect to the broader world of **Supervised Learning**.

**Project Structure:**
```
notebooks/
  hallucination_classifier.ipynb      ← Notebook 1: Baseline classification (TF-IDF only)
  hallucination_classifier_v2.ipynb   ← Notebook 2: Classification + handcrafted features
  hallucination_regression.ipynb      ← Notebook 3: Regression (faithfulness scoring)
scripts/
  build_notebook.py                   ← Regenerates Notebook 1
  build_notebook_v2.py                ← Regenerates Notebook 2
  build_notebook_regression.py        ← Regenerates Notebook 3
```

***

## 1. What is Supervised Learning?

**Supervised Learning** is a type of machine learning where we teach a computer by giving it labeled examples. Imagine teaching a child what an apple is by showing them 100 pictures of apples and saying "This is an apple" each time.

In our project:
* **The Input (Features / $X$):** The text containing the Question and the Answer.
* **The Output (Label / $y$):** Either a binary tag (correct/hallucinated) or a continuous score (how faithful is this answer?).

Because we provided the computer with labeled examples, it learns the relationship between text patterns and the label. That is the "Supervision."

***

## 2. Classification vs. Regression

There are two main types of supervised learning:

| Type | Output | Example |
|---|---|---|
| **Classification** | A category | Hallucinated or Correct? |
| **Regression** | A continuous number | How faithful is this answer? (0.0 – 1.0) |

Our project uses **both**:
- **Notebooks 1 & 2** are Classification problems (binary label: 0 or 1)
- **Notebook 3** is a Regression problem (faithfulness score: 0.0 to 1.0)

***

## 3. Dealing with Text: TF-IDF

Machine Learning models only understand numbers, not words. **TF-IDF** converts text into numbers.

### Part 1: TF (Term Frequency)
How often does a word appear in *this specific answer*?

- Answer is 100 words long. "robot" appears 5 times → TF = 5/100 = **0.05**

*Problem:* Words like "the" and "is" appear everywhere but mean nothing.

### Part 2: IDF (Inverse Document Frequency)
How common is this word *across all 10,000 answers*?

- "the" appears in every answer → IDF score ≈ **0** (penalized heavily)
- "photosynthesis" appears rarely → IDF score = **high** (rewarded)

### The Final Formula: TF × IDF

> **"the"**: High TF × Zero IDF = **0** → ignored as irrelevant
> **"definitely"**: High TF × High IDF = **High Score** → model pays attention

Every answer becomes a row of 5,000 numbers — one per tracked word/phrase — representing its true informational content.

***

## 4. Handcrafted Linguistic Features (Notebook 2 & 3)

TF-IDF only captures *what words appear*. We also engineered 10 handcrafted features that capture *how the answer is written*:

| Feature | What It Measures | Why It Matters |
|---|---|---|
| `answer_length` | Word count of answer | Hallucinated answers tend to be longer/wordier |
| `hedge_density` | Rate of words like "might", "possibly", "around" | Vague language signals uncertainty |
| `number_density` | Rate of numeric tokens | Specific numbers = more factually grounded |
| `unique_word_ratio` | Unique words ÷ total words | Low ratio = repetitive/padded filler |
| `capital_word_ratio` | Capitalized words (not sentence-start) | Proxy for named entity density |
| `avg_word_length` | Average characters per word | Domain-specific vocab uses longer words |
| `length_ratio` | Answer length ÷ Question length | Is the answer proportionally detailed? |
| `sentence_count` | Number of sentences | Structural complexity |
| `word_overlap` | Shared vocabulary between question and answer | Does the answer address the actual question? |
| `qa_tfidf_similarity` | TF-IDF cosine similarity of question vs answer | Is the answer just echoing the question? |

**Result in Notebook 2:** Adding these 10 features to TF-IDF boosted F1 score by **+0.277** for Logistic Regression (from 0.669 → 0.946). This was the single biggest finding in the project.

***

## 5. Feature Combination: Sparse + Dense Matrix

TF-IDF produces a **sparse matrix** (5,000 columns, mostly zeros). The handcrafted features produce a **dense matrix** (10 columns, all filled). We combine them using `scipy.sparse.hstack` into one matrix of **5,010 features** per sample.

```
TF-IDF (5000 cols) | handcrafted (10 cols)  →  combined (5010 cols)
[0, 0, 0.12, 0, ...]  [12.0, 0.03, ...]     →  [0, 0, 0.12, ..., 12.0, 0.03, ...]
```

***

## 6. Train / Test Split

If you memorize exact answers to exam questions, you score 100% but learn nothing. ML models can do the same (called **Overfitting**). So we split data into two piles:

* **Training Set (80% = 16,000 samples):** What the model learns from.
* **Testing Set (20% = 4,000 samples):** The hidden "final exam" — never seen during training.

**Important:** In Notebooks 2 & 3 we split the *dataframe* first, then fit all transformers (TF-IDF, scaler, QA vectorizer) on training data only. This prevents **data leakage** — accidentally letting test-set information influence training.

***

## 7. The Classification Models (Notebooks 1 & 2)

### A. Logistic Regression
A simple, fast algorithm that draws a flat decision boundary through the data. Its biggest advantage is **interpretability** — you can read the coefficients to see exactly which features push it toward predicting "hallucinated."

### B. Random Forest
An "ensemble" method. It builds hundreds of small decision trees, each trained on random subsets of the data. Every tree votes, and the majority wins. More powerful than Logistic Regression but harder to interpret.

***

## 8. The Regression Models (Notebook 3)

Instead of predicting a category, these models predict a continuous number — the **faithfulness score**.

### A. Ridge Regression
Logistic Regression's counterpart for regression. Adds a penalty ($\alpha \times \sum w^2$) to keep coefficients small and prevent overfitting. Great for high-dimensional data like TF-IDF.

### B. Lasso Regression
Similar to Ridge, but its penalty ($\alpha \times \sum |w|$) automatically drives unimportant feature weights to exactly **zero**. This performs automatic feature selection — Lasso tells you which features matter and which don't.

### C. ElasticNet
A combination of Ridge and Lasso penalties. The `l1_ratio` parameter controls the mix. Gets the benefits of both: stability from Ridge, sparsity from Lasso.

### D. LinearSVR (Support Vector Regression)
Finds the flattest possible line/hyperplane that fits within a margin of error $\epsilon$ around the true values. Points outside the margin are the only ones that contribute to the loss. Works well in high-dimensional spaces.

### E. HistGradientBoosting Regressor
The most powerful model in our set. Builds an ensemble of decision trees **sequentially** — each new tree corrects the errors of the previous ones. Uses histogram binning for speed. Achieved the best R² (0.35) in our project.

***

## 9. Preventing Data Leakage with Pipelines

All regression models are wrapped in a `Pipeline`:
```
MaxAbsScaler → Model
```
`MaxAbsScaler` scales each feature to [-1, 1] by dividing by the maximum absolute value. It preserves the sparse structure of TF-IDF matrices. Putting it inside a Pipeline ensures the scaler is fit only on training folds during cross-validation — not on the whole dataset.

***

## 10. Hyperparameter Tuning: GridSearchCV

Every model has settings called **hyperparameters** — knobs you set *before* training (e.g., how strong is the regularization penalty?). We use **5-Fold Cross-Validation Grid Search** to find the best settings automatically.

**How 5-Fold CV works:**
1. Split the training set into 5 equal chunks.
2. Train on 4 chunks, test on the 1 remaining chunk.
3. Repeat 5 times (each chunk gets a turn as the test set).
4. Average the 5 scores → this is the true performance estimate for that setting.
5. Try every combination in the parameter grid and pick the best.

This prevents picking hyperparameters that just happen to work on one lucky split.

***

## 11. Evaluation Metrics

### Classification Metrics (Notebooks 1 & 2)
* **Accuracy:** What % of predictions were correct overall?
* **Precision:** Of all answers the model called hallucinated, how many actually were?
* **Recall:** Of all actual hallucinations, how many did the model catch?
* **F1 Score:** Harmonic mean of Precision and Recall — the best single-number summary.
* **ROC-AUC:** How well does the model separate the two classes across all probability thresholds? 1.0 = perfect, 0.5 = random guessing.
* **Confusion Matrix:** A 2×2 grid showing exactly which types of errors the model made.

### Regression Metrics (Notebook 3)
* **MSE (Mean Squared Error):** Average of squared errors. Penalizes large errors heavily. Lower = better.
* **MAE (Mean Absolute Error):** Average of absolute errors. More robust to outliers than MSE. Lower = better.
* **R² (R-squared):** What fraction of the variance in $y$ does the model explain? 1.0 = perfect, 0.0 = no better than predicting the mean, negative = worse than the mean.

***

## 12. Pairwise Ranking Accuracy (Notebook 3)

Beyond standard regression metrics, we also evaluated whether the model can **rank** answers correctly. For each question that has both a correct and hallucinated answer in the test set (402 pairs), we check: does the model assign a higher faithfulness score to the correct answer?

- **100%** = perfect ranker
- **50%** = random coin flip

Our models scored 13–16% — far below random. This leads to the most important finding in the project (see Section 14).

***

## 13. Classification Recovery (Notebook 3)

We can convert regression predictions back into binary labels by applying a threshold:
- Predicted faithfulness score < 0.5 → predicted "hallucinated" (label = 1)
- Predicted faithfulness score ≥ 0.5 → predicted "correct" (label = 0)

This lets us compare the regression approach directly against the classification approach on the same F1/AUC metrics.

***

## 14. The Key Finding: Why the Regression Failed

The most scientifically interesting result in this project is that the regression approach did not work as intended — and understanding *why* is more valuable than if it had simply worked.

**The problem:** Our regression target was TF-IDF cosine similarity between each answer and the knowledge context. We expected correct answers to score higher (more similar to the facts) and hallucinated answers to score lower.

**What actually happened:**

| Label | Mean Faithfulness Score |
|---|---|
| Correct (label=0) | **0.258** |
| Hallucinated (label=1) | **0.332** |

Hallucinated answers scored *higher* on faithfulness than correct ones.

**Why?** HaluEval's hallucinated answers are deliberately crafted to sound plausible — they borrow real vocabulary from the knowledge passage. Meanwhile, correct answers in the dataset are often extremely short and direct ("Arthur's Magazine", "Delhi") and share little TF-IDF vocabulary with the full knowledge paragraph.

**What this teaches us:** Bag-of-words similarity breaks down in adversarial settings where hallucinated text is crafted to look similar to factual text. This is a real and active problem in LLM safety research — it motivates the need for semantic embeddings (models that understand *meaning*, not just word overlap) for faithfulness scoring.

***

## 15. Ablation Study

An **ablation study** removes one component at a time to measure its exact contribution.

In Notebook 2, we trained models with and without the handcrafted features:

| Model | TF-IDF Only | + Handcrafted | Gain |
|---|---|---|---|
| Logistic Regression | F1: 0.669 | F1: 0.946 | **+0.277** |
| Random Forest | F1: 0.722 | F1: 0.947 | **+0.226** |

This proved that the handcrafted features — not TF-IDF — are the dominant signal in this dataset. The reason: in HaluEval, correct answers are short and direct while hallucinated answers are long, wordy, and echo the question. `answer_length` alone (coefficient: +5.17) captures most of this pattern.

***

## Summary of the Full Pipeline

**Notebooks 1 & 2 — Classification:**
1. Load HaluEval QA dataset (10,000 → 20,000 labeled samples)
2. Convert text to TF-IDF features
3. Engineer 10 handcrafted linguistic features (Notebook 2 only)
4. Combine into 5,010-feature matrix
5. Train Logistic Regression + Random Forest
6. Evaluate with Accuracy, F1, ROC-AUC, Confusion Matrix
7. Ablation: measure handcrafted feature contribution (Notebook 2 only)

**Notebook 3 — Regression:**
1. Load HaluEval QA dataset (keeping the `knowledge` column)
2. Compute faithfulness score = cosine similarity(answer, knowledge) → regression target $y$
3. Build same 5,010-feature matrix from question + answer only (no knowledge leakage)
4. Train Ridge, Lasso, ElasticNet, LinearSVR, HistGradientBoosting with GridSearchCV
5. Evaluate with MSE, MAE, R²
6. Pairwise ranking accuracy
7. Classification recovery (threshold regression score → binary label)
8. Discover that TF-IDF faithfulness scoring fails adversarially — motivates semantic embeddings
