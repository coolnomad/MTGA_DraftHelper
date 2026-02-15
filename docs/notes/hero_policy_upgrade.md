
# Hero-Bot Policy Upgrade: Soft Value Distillation Strategy

## Overview

This document describes the **best strategy** for upgrading the hero-bot into a high‑quality drafting policy. The key idea is to **freeze the hero value function** as a critic \( V(s) \) and learn a **new policy** \( \pi_\theta(a \mid s) \) through *soft value distillation*.

This produces a bot that is:
- stochastic  
- smooth and "wiggly"  
- tunable  
- more human-like  
- significantly stronger than a greedy argmax policy  

---

## 1. Keep Hero as the Value Function

Hero’s learned XGBoost model approximates:

\[
V(s) \approx E[\text{deck\_bump} \mid s].
\]

This critic remains **frozen**. We do *not* modify it during policy training.

---

## 2. Compute Hero Values for Every Card Action

For each draft decision in `bc_dataset`:

1. Take the pack \( \{c_1, ..., c_k\} \).
2. For each card \( c \) in the pack:
   - simulate taking \( c \) → new state \( s_c \)
   - compute hero value:
     \[
     q_c = V(s_c)
     \]

This produces a dataset of hero Q-values for all candidate actions.

---

## 3. Convert Hero Values to Soft Targets

Transform hero Q-values into a probability distribution:

\[
p^*(a = c \mid s) = 
\frac{\exp(q_c / \tau)}{\sum_{c'} \exp(q_{c'} / \tau)}.
\]

Where:
- **\( \tau \)** = temperature  
  - small → nearly greedy  
  - medium → smooth, human-like behavior  
  - large → exploratory policy  

This distribution encodes the **shape** of hero’s preferences, not just the argmax.

---

## 4. Train a Policy Network \( \pi_\theta(a \mid s) \)

Train a model to match hero’s soft targets by minimizing cross‑entropy:

\[
\min_\theta 
\; \mathbb{E}[-\sum_{c} p^*(c \mid s) \log \pi_\theta(c \mid s)].
\]

The policy model \( \pi_\theta \) can be:

- a neural network  
- a boosted tree  
- any architecture producing a probability distribution  

This produces a **fully expressive drafting policy**, capable of subtle behavior.

---

## 5. Advantages of This Approach

### ✔ Stochastic Behavior  
Policy samples actions based on value gradients, not deterministic argmax.

### ✔ Smooth / "Wiggly" Choices  
Minor differences in hero Q-values become smooth differences in probabilities.

### ✔ Exploration Built Into the Temperature  
Lower \( \tau \): tighter to hero  
Higher \( \tau \): more diverse drafting  

### ✔ Strong Generalization  
Policy learns **structure**, not just greedy decisions.

### ✔ Can Blend Hero + Human  
You can blend hero Q-values with human logits:

\[
p^*(c|s) = 
\text{softmax}(\alpha q_c^{\text{hero}} 
+ (1-\alpha) z_c^{\text{human}}).
\]

This yields:
- hero-strength with  
- human-style early game preferences

### ✔ Ideal Starting Point for Offline RL  
This policy can be improved with:
- policy iteration  
- Q-augmented distillation  
- self-play  
- DPO-style preference optimization  

---

## 6. Summary

**Hero 2.0 Strategy:**  
1. Freeze hero’s value function.  
2. Compute Q-values for all card choices.  
3. Convert them into softmax targets.  
4. Train a flexible policy \( \pi_\theta(a|s) \) to imitate these target distributions.  

The result is a **powerful, tunable, and fully general drafting agent** that far exceeds what greedy hero can do.
