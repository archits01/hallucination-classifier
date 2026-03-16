# Supervised Learning Concepts Explained: Hallucination Classifier

Welcome to the beginner-friendly guide to your Machine Learning project! This document will break down every major concept we used in the `hallucination_classifier.ipynb` notebook and explain how it connects to the broader world of **Supervised Learning**.

***

## 1. What is Supervised Learning?
**Supervised Learning** is a type of machine learning where we teach a computer by giving it examples. Imagine teaching a child what an apple is by showing them 100 pictures of apples and saying "This is an apple" each time. 

In our project:
* **The Input (Features / $X$):** The text containing the Question and the Answer.
* **The Output (Label / $y$):** A tag saying whether the answer is factually Correct (`0`) or Hallucinated (`1`).

Because we provided the computer with perfectly labeled examples (we know which ones are hallucinations and which ones aren't), the computer learns the relationship between the text and the label. This is the "Supervision."

***

## 2. Binary Classification
There are two main types of supervised learning: **Regression** (predicting a continuous number, like the price of a house) and **Classification** (predicting a category, like Dog vs. Cat).

Our project is a **Binary Classification** problem because there are exactly two possible categories:
* `Class 0`: Correct Answer
* `Class 1`: Hallucinated Answer

***

## 3. Dealing with Text: TF-IDF (Term Frequency - Inverse Document Frequency)
Machine Learning models are mathematical formulas. They don't understand English words like "banana" or "hallucination"; they only understand numbers. So, we have to convert our Question+Answer text into a mathematical representation.

**TF-IDF** is the specific mathematical formula we used to convert our text into numbers. Let's break it down into its two parts: **TF** and **IDF**.

### Part 1: TF (Term Frequency)
This part simply asks: **How often does a specific word appear in this specific answer?**

Imagine we have an answer that is 100 words long.
* If the word `"robot"` appears 5 times, its TF score is $5/100 = 0.05$.
* If the word `"the"` appears 10 times, its TF score is $10/100 = 0.10$.

*The Problem:* If we only use TF, the model will think that `"the"`, `"is"`, and `"and"` are the most important words in the sentence, simply because they appear the most often. But those words don't tell us anything useful about hallucinations!

### Part 2: IDF (Inverse Document Frequency)
This part solves the problem above. It asks: **How common is this word across *all 10,000 answers* in our dataset?**

IDF heavily penalizes words that appear everywhere, and boosts the score of rare, unique words.
* **High IDF Score:** Given to rare words (like `"photosynthesis"` or `"Einstein"`). If these words show up in a sentence, they carry a lot of meaning!
* **Low / Zero IDF Score:** Given to extremely common words (like `"the"` or `"is"`).

### The Final Formula (TF $\times$ IDF)
To get the final number for a single word in a single sentence, we multiply its TF score by its IDF score.

> **Example for the word "the":**
> High TF (appears a lot in this sentence) $\times$ Zero IDF (appears in every sentence in the dataset) = **0**
> *Result: "The" is mathematically ignored as irrelevant.*

> **Example for the word "definitely":**
> High TF (appears 3 times in this sentence) $\times$ High IDF (rarely appears across the whole dataset) = **High Score**
> *Result: The model pays close attention to this word.*

**End Result:** Every text example becomes a row of numbers, where each number represents the true "importance" of a specific word in that text. We limited our model to only track the top 5,000 most important words (`max_features=5000`).

***

## 4. Train / Test Split
If you study for a math test by memorizing the exact answers to the exact questions, you might score 100%. But if the teacher gives you a *new* question, you will fail because you just memorized, you didn't *learn*.

Machine Learning models can do the same thing (called **Overfitting**). To prevent this, we split our 10,000 examples into two piles:
* **Training Set (80%):** The 8,000 examples the model uses to learn and find patterns.
* **Testing Set (20%):** The 2,000 examples we hide from the model until the very end to act as the "Final Exam."

If the model does well on the Testing Set, it means it genuinely learned how to spot hallucinations in text it has never seen before!

***

## 5. The Models (Algorithms)
We trained two different "students" (algorithms) to see which one learns better.

### A. Logistic Regression
* **What is it?** A very simple, classic, and fast algorithm. It draws a single straight line (or flat plane) through the data to separate the `0`s from the `1`s.
* **Why use it?** It is highly **interpretable**. At the end of the notebook, we could actually look at the "Coefficients" of Logistic Regression to see *exactly which words* pushed the model toward predicting a hallucination.

### B. Random Forest
* **What is it?** An "Ensemble" method. Instead of one algorithm, a Random Forest creates hundreds of tiny "Decision Trees" (flowcharts of yes/no questions). Each tree makes a guess, and the forest takes a majority vote.
* **Why use it?** It is usually much more powerful than Logistic Regression and can find complex, non-linear patterns.

***

## 6. Evaluation Metrics: How do we grade the models?
Once the models take their "Final Exam" on the Test Set, we need to grade them. Accuracy isn't always enough!

* **Accuracy:** Overall, what percentage of predictions did the model get right? *(e.g., 85%)*
* **Precision:** Out of all the answers the model *claimed* were hallucinations, how many actually were? (High precision means we don't accidentally accuse a correct answer of being a hallucination).
* **Recall:** Out of *all the actual* hallucinations in reality, how many did the model manage to find? (High recall means no hallucinations slip through the cracks).
* **F1 Score:** A balanced metric that perfectly combines Precision and Recall into one number.
* **Confusion Matrix:** A $2 \times 2$ visual grid showing exactly where the model got confused (e.g., predicting `0` when the actual answer was `1`).

### What is ROC-AUC?
**ROC-AUC** stands for *Receiver Operating Characteristic - Area Under Curve*. 
Models don't just output `0` or `1`. They output a *probability* (e.g., "I am 87% sure this is a hallucination"). 
The ROC Curve charts how well the model separates the two classes across all different probability thresholds. The **AUC** is a score from 0 to 1, where 1.0 is a perfect model and 0.5 is a model that is just guessing completely randomly.

***

## Summary of the Pipeline
1. Take text examples (Inputs).
2. Assign Correct / Hallucinated labels (Outputs based on Supervision).
3. Convert Text to Numbers (TF-IDF).
4. Hide 20% of the data for testing (Train/Test Split).
5. Teach the algorithms (Logistic Regression & Random Forest).
6. Grade them on the hidden 20% (Metrics & Confusion Matrix).
