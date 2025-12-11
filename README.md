# 💰 Policy Optimization for Financial Decision-Making
### Bridging risk prediction and profitability with Deep Learning and Offline RL

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-MLP-red) ![d3rlpy](https://img.shields.io/badge/d3rlpy-Offline%20RL-green) ![Status](https://img.shields.io/badge/Status-Complete-success)

> Key idea: Most credit models predict default probability. Here I focus on expected profit. The offline RL agent learns to reject a specific high risk segment of loans that human underwriters historically approved, with a default rate of 47.9 percent.

---

## 📖 Overview

The project uses the public LendingClub loan book (about 1.37M accepted loans) and compares two approaches to loan approval:

1. **Supervised Deep Learning (risk model)**  
   A PyTorch MLP that estimates the probability that a loan will default.

2. **Offline Reinforcement Learning (value model)**  
   A Conservative Q Learning (CQL) agent trained on logged approvals to learn an approval policy that maximizes expected financial value.

The goal is not only to classify loans as good or bad, but to answer a more useful question:  
**Given the interest income and the risk of losing principal, should this loan be approved at all?**

---

## 🚀 Key Results

All numbers are computed on a held out test set of accepted loans.

| Metric                         | Historical policy (approve all) | RL agent (my policy) | Effect |
| :---                           | :---                            | :---                 | :--- |
| **Action**                     | Approve every loan              | Approve or deny      | Learns selective approvals |
| **Average value per loan**     | about -1.94k USD                | about -1.55k USD     | Loss reduced by roughly 400 USD per loan |
| **Default rate of approved**   | 21.5 percent                    | 20.0 percent         | Safer approved portfolio |
| **Loans denied by RL policy**  | 0                               | 14,415               | Concentrated high risk set |

The rejected set of 14,415 loans has a default rate of 47.9 percent.  
So the agent is not just copying historical behavior, it has found a region of the feature space where the expected loss from principal dominates any interest income.

---

## 🛠️ Project Structure

The repository follows the same order as the analysis.

### `01_eda_preprocessing.ipynb` 📊
**Goal:** Turn the raw CSV into a clean modeling dataset.

Main steps:
- Load a subset of LendingClub columns that are available at application time.
- Map loan statuses into a binary target `{0: fully paid, 1: defaulted}`.
- Convert text fields like `term` and `emp_length` into numeric features.
- Build features such as `credit_age_years`, `issue_year`, `issue_month`.
- Handle missing values and define categorical and numeric feature lists.
- Save the cleaned dataframe and a fitted preprocessing pipeline.

### `02_supervised_mlp.ipynb` 🧠
**Goal:** Train the baseline deep learning classifier.

- One hot encode categorical variables and standardize numeric ones.
- Train a PyTorch MLP on about 0.96M training rows.
- Tune the decision threshold for the positive class to maximize F1 on the validation set.

Key metrics on the test set:
- **AUC:** about 0.73  
- **F1 (at tuned threshold):** about 0.47  
- Accuracy is around 0.79, but the F1 and confusion matrix show that the model recovers a meaningful portion of defaults while keeping false alarms under control.

### `03_offline_rl.ipynb` 🤖
**Goal:** Learn an approval policy with offline RL.

Setup:
- **State:** 18 engineered financial features such as loan amount, interest rate, debt to income ratio, FICO ranges, revolver stats, credit age and issue month.
- **Action:** 0 = deny, 1 = approve.
- **Reward:**  
  - If approved and fully paid: loan_amnt * int_rate  
  - If approved and defaulted: negative loan_amnt  
  - If denied: 0  
  Rewards are scaled to thousands for stability.

Implementation:
- Standardize state features with `StandardScaler`.
- Build an `MDPDataset` from approved loans only, with terminal transitions.
- Train a Discrete CQL agent using `d3rlpy` for 100k gradient steps.
- Evaluate the learned policy offline by applying it to a held out test set and computing average realized reward.

Outcome:
- The learned policy denies a small but very risky group of loans.
- Portfolio value improves and the default rate among approved loans drops.

---

## 💻 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/lendingclub-policy-optimization.git
   cd lendingclub-policy-optimization
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate        # on Linux or macOS
   venv\Scriptsctivate           # on Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebooks in order**
   - `01_eda_preprocessing.ipynb` builds the cleaned dataset and preprocessing artifacts.
   - `02_supervised_mlp.ipynb` trains the supervised MLP and reports AUC, F1 and confusion matrix.
   - `03_offline_rl.ipynb` trains the CQL agent and compares the RL policy to the historical approvals.

---

## 📦 Main Dependencies

- Python 3.8 or later  
- torch  
- d3rlpy  
- pandas, numpy, scikit-learn  
- matplotlib, seaborn  

---

## 🔮 Possible Next Steps

- Add data from rejected loan applications to reduce selection bias.
- Model time explicitly by discounting cash flows rather than using simple interest totals.
- Combine the supervised risk score with the RL state to build a hybrid policy.

---

**Author:** Gautam Govind  
**Report:** `Policy_Optimization_RL_Report_Gautam_Govind.pdf`
