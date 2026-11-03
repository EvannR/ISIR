# Empathy and the Tragedy of the Commons in Multi-Agent Reinforcement Learning

This work is carried out at ISIR (Institut des Systèmes Intelligents et de Robotique), under the supervision of Mehdi Khamassi and Ismael T. Freire.

Independent deep reinforcement learning agents share a non-renewable resource. Each agent emits an affective signal derived from its recent consumption. The experiments vary how precisely agents perceive the signals of the others, and how much they are rewarded for the others' state.

## Contents

The code, results and analysis notebooks are in `03_GT/`:

- `main_gt_ais.py`: runs the experiments
- `env_gt_ais.py`: environment
- `agent_policies_gt_ais.py`: agents and reward
- `analysis/`: one notebook per experiment

## Quick start

```bash
pip install -r requirements.txt
cd 03_GT
python main_gt_ais.py --smoke                  # quick check: 2 episodes per condition
python main_gt_ais.py --exp exp1_resource_obs  # reference 2x2 of Axis 1
```
