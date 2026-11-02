# Empathic Multi-Agent Resource Sharing

This repository contains 2D grid-based and graph-based reinforcement learning environments
for studying empathic vs non-empathic agents in collective resource sharing tasks.

- **2D environment**: agents move on a grid and collect regenerating resources (`2D/`).
- **Graph environment**: agents occupy nodes in a social graph and share a central pool (`agents_graph/`).

## Quick start

```bash
pip install -r requirements.txt
cd 2D && python main_2d.py          # run 2D grid experiments
cd agents_graph && python main_graph.py  # run graph-based experiments
```