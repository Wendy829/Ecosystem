import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mean_sd(df, seed_cols):
    x = df["step"].to_numpy()
    Y = df[seed_cols].to_numpy(dtype=float)
    mean = Y.mean(axis=1)
    sd = Y.std(axis=1, ddof=1)  # 样本SD
    return x, mean, sd

# 1) 读两份数据（改成你的路径/文件名）
mlp = pd.read_excel(r"C:\Users\lyk\Desktop\mlp_100.xlsx")      # 或 pd.read_csv(...)
trans = pd.read_excel(r"C:\Users\lyk\Desktop\tf_100.xlsx")  # 或 pd.read_csv(...)

# 2) 改成你真实的列名（每份数据各自的3个seed列名）
mlp_seed_cols = ["MLP_891_bank_profit", "MLP_105_bank_profit", "MLP_117_bank_profit"]
trans_seed_cols = ["tf_891_bank_profit", "tf_105_bank_profit", "tf_117_bank_profit"]

# 3) 算 mean/sd
x1, m1, s1 = mean_sd(mlp, mlp_seed_cols)
x2, m2, s2 = mean_sd(trans, trans_seed_cols)

# 4) （可选但推荐）对齐step：用 merge 取交集，防止两边 step 不完全一致
mlp_ms = pd.DataFrame({"step": x1, "mean_mlp": m1, "sd_mlp": s1})
trans_ms = pd.DataFrame({"step": x2, "mean_trans": m2, "sd_trans": s2})
df = mlp_ms.merge(trans_ms, on="step", how="inner").sort_values("step")

# 5) 画在同一张图
plt.figure(figsize=(8,4))

eps = 1e-3  # 或者 1e-3 / 1.0，选一个比你数据最小正值更小的数

plt.figure(figsize=(8,4))

# MLP
# lower1 = np.maximum(df["mean_mlp"] - df["sd_mlp"], eps)
# upper1 = np.maximum(df["mean_mlp"] + df["sd_mlp"], eps)
plt.plot(df["step"], df["mean_mlp"], color="C0", lw=2, label="PPO-MLP mean")
plt.fill_between(df["step"], df["mean_mlp"] - df["sd_mlp"], df["mean_mlp"] + df["sd_mlp"], color="C0", alpha=0.15)

# Transformer
# lower2 = np.maximum(df["mean_trans"] - df["sd_trans"], eps)
# upper2 = np.maximum(df["mean_trans"] + df["sd_trans"], eps)
plt.plot(df["step"], df["mean_trans"], color="C1", lw=2, label="PPO-Transformer mean")
plt.fill_between(df["step"], df["mean_trans"] - df["sd_trans"], df["mean_trans"] + df["sd_trans"], color="C1", alpha=0.15)

plt.yscale("log")
plt.xlim(left=0)
# 不要 plt.ylim(bottom=0)；如需下界，用正数：
# plt.ylim(bottom=eps)

plt.xlabel("env steps")
plt.ylabel("bank profit")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()