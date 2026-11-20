# AI651 Assignment 3 · Graphs and Traffic Forecasting

**Namra Nadeem · Student ID 25280097**

[Open the assignment notebook](25280097_AI651_PA3.ipynb)

The notebook covers neighbor aggregation and a ring-road simulation (Q1–Q2), preparation of traffic readings for 716 San Diego sensors (Q3), and forecasting with temporal-only, fixed, adaptive, and hybrid graph models (Q4–Q6).

## Reproduce the experiments in Kaggle

1. Import `25280097_AI651_PA3.ipynb` into a Kaggle notebook. The code uses Python, NumPy, pandas, PyTables for HDF5, Matplotlib, and PyTorch.
2. Attach the [LargeST dataset](https://www.kaggle.com/datasets/liuxu77/largest) via **Add Input**. Q1 and Q2 can run independently of LargeST.
3. Check `data_dir` in Q3. It defaults to `/kaggle/input/datasets/liuxu77/largest` and should contain `ca_his_raw_2020.h5`, `ca_his_raw_2021.h5`, `ca_meta.csv`, and `ca_rn_adj.npy`. Update the path if the mounted folder differs.
4. Run the notebook from top to bottom. Use a CPU session to match the hardware type of the recorded runs. The code automatically uses CUDA when available.

## Experiment settings

Traffic readings are aggregated into 15-minute intervals. The chronological splits are **2020 for training**, **January–March 2021 for validation**, and **April–December 2021 for testing**. Normalization uses observed training values. An observation mask excludes missing targets from the loss and metrics.

| Setting | Value |
|---|---|
| Input and forecast | 8 input readings (2 hours), 4 target readings (1 hour) |
| Hidden dimension | 16 |
| Adaptive embedding dimension | 16 |
| Hybrid fixed-graph weight | 0.5 |
| Training | 5 epochs, batch size 32 |
| Optimizer and loss | Adam, learning rate 0.001, masked MAE |
| Random seed | 42 |

## Forecast results

MAE and RMSE are in vehicles per 15-minute interval. Timings are for CPU training and exclude evaluation.

| Model | Validation MAE | Validation RMSE | Test MAE | Test RMSE | Training (s) |
|---|---:|---:|---:|---:|---:|
| Temporal-only | 59.984 | 96.943 | **67.742** | **106.987** | 75.5 |
| Fixed STGNN | 121.073 | 185.338 | 135.135 | 202.073 | 136.8 |
| Adaptive STGNN | 65.582 | 109.319 | 76.707 | 120.438 | 172.2 |
| Hybrid STGNN | 71.392 | 113.562 | 80.759 | 122.438 | 181.2 |

The temporal-only baseline had the lowest test error. The adaptive STGNN performed best among the graph models.

## Outputs

Saved plots and metrics are embedded in the notebook. A fresh run writes the prepared data, four model checkpoints, learned adjacency arrays, and forecast figures to `/kaggle/working/pa3_outputs/`. The Q2 ring-road heatmap is saved separately as `q2_ring_road_heatmaps.png`.
