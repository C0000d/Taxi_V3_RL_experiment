# Reinforcement Learning Methods Experiment on Taxi_v3

A compact, reproducible notebook for mastering tabular Q‑learning and SARSA on the classic Gymnasium Taxi‑v3 environment.

## Description
Implemented and analysed tabular off‑policy Q‑learning and on‑policy SARSA for the Taxi‑v3 control task. The notebook doubles as an experiment harness: one cell flips between algorithms, action‑selection policies and hyper‑parameter grids, then plots learning curves and prints aggregate stats.

## Techniques involved
| Area                       | What I implemented                                                                                                                                              |
|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Core RL algorithms**     | Clean NumPy versions of Q‑learning and SARSA with separate update functions.                                                                                     |
| **Exploration strategies** | Plug‑in choice of **greedy**, **ε‑greedy** and **soft‑max** action selectors.                                                                          |
| **Vectorised computation** | Removed Python for‑loops inside the Q‑update; all state‑action value updates run as NumPy ops, cutting a 30 k‑episode sweep from ~26 s to ~13 s on CPU.          |
| **Hyper‑parameter search** | Programmatic sweeps over α, γ, ε and episode count; auto‑plots reward and steps per episode for every run.                                                       |
| **Evaluation harness**     | Tester that runs 100 evaluation episodes with the learned Q‑table, returning mean reward/steps and a state‑by‑state textual trace for qualitative inspection.    |
| **Visual diagnostics**     | Matplotlib curves for each sweep plus comparative tables of average reward and path length.                                                                      |

## Key Outcomes

- Parameter insight – Identified (α ≈ 0.55, γ ≈ 0.99, ε ≈ 0.01, 10 k episodes) as the best trade‑off, yielding +15 % average reward and ‑40 % steps‑to‑goal vs. default settings.

- Performance boost – Pure‑Python reference loop replaced by vectorised NumPy, halving training time for all sweeps.

- Robust experimentation workflow – Single‑notebook pipeline lets new sweeps or policies run with two lines of code, making the setup reusable for future grid‑world experiments.

## Issues Tackled

- Runtime bottleneck – Original nested loops became a major drag during sweeps; fixed by batching Q‑updates and action‑probability calculations.

- Hyper‑param brittleness – Added auto‑sweep utilities and plotting to expose unstable regions, preventing over‑fit to arbitrary defaults.

- Debugging agent behaviour – Built a step‑by‑step episode visualiser to surface mis‑behaving state transitions, accelerating convergence troubleshooting.

This project showcases practical reinforcement‑learning fundamentals (tabular methods, exploration policies, evaluation) wrapped in a concise, performant and extensible codebase.
