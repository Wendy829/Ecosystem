import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 方式1：直接指定常见中文字体（按顺序尝试，系统有哪个用哪个）
mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "PingFang SC"]
mpl.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题

def mean_sd_common(df, seed_cols, step_col="step"):
    # 处理缺失值：只保留 seed_cols 都不为 NaN 的行
    d = df[[step_col] + seed_cols].dropna(subset=seed_cols).copy()
    x = d[step_col].to_numpy()
    Y = d[seed_cols].to_numpy(float)
    mean = Y.mean(axis=1)
    sd = Y.std(axis=1, ddof=1)
    return x, mean, sd

# 1) 读两份数据（改成你的路径/文件名）
mlp = pd.read_excel(r"C:\Users\lyk\Desktop\mlp_100.xlsx")      # 或 pd.read_csv(...)
trans = pd.read_excel(r"C:\Users\lyk\Desktop\tf_100.xlsx")  # 或 pd.read_csv(...)

# 2) 改成你真实的列名（每份数据各自的3个seed列名）
mlp_seed_cols_consumption = ["MLP_891_consumption", "MLP_105_consumption", "MLP_117_consumption"]
trans_seed_cols_consumption = ["tf_891_consumption", "tf_105_consumption", "tf_117_consumption"]
mlp_seed_cols_production =["MLP_891_production", "MLP_105_production", "MLP_117_production"]
trans_seed_cols_production = ["tf_891_production", "tf_105_production", "tf_117_production"]

# 3) 算 mean/sd
x1, m1, s1 = mean_sd_common(mlp, mlp_seed_cols_consumption)
x2, m2, s2 = mean_sd_common(trans, trans_seed_cols_consumption)
x3, m3, s3 = mean_sd_common(mlp, mlp_seed_cols_production)
x4, m4, s4 = mean_sd_common(trans, trans_seed_cols_production)

# 4) （可选但推荐）对齐step：用 merge 取交集，防止两边 step 不完全一致
mlp_consumption = pd.DataFrame({"step": x1, "mean_mlp_consumption": m1, "sd_mlp_consumption": s1})
trans_consumption = pd.DataFrame({"step": x2, "mean_trans_consumption": m2, "sd_trans_consumption": s2})
mlp_production=pd.DataFrame({"step": x3, "mean_mlp_production": m3, "sd_mlp_production": s3})
trans_production=pd.DataFrame({"step": x4, "mean_trans_production": m4, "sd_trans_production": s4})

from functools import reduce

tables = [mlp_consumption, trans_consumption, mlp_production, trans_production]
df = reduce(lambda l, r: l.merge(r, on="step", how="inner"), tables).sort_values("step")
# 5) 画在同一张图
plt.figure(figsize=(8,4))

eps = 1e-3  # 或者 1e-3 / 1.0，选一个比你数据最小正值更小的数

plt.figure(figsize=(8,4))

# MLP
# lower1 = np.maximum(df["mean_mlp"] - df["sd_mlp"], eps)
# upper1 = np.maximum(df["mean_mlp"] + df["sd_mlp"], eps)
plt.plot(df["step"], df["mean_mlp_consumption"], color="#70C965", lw=1.25, label="MLP_consumption1")
plt.plot(df["step"], df["mean_mlp_production"], color="#9F6C65", lw=1.25, label="MLP_production1",linestyle='--')
# plt.fill_between(df["step"], df["mean_mlp"] - df["sd_mlp"], df["mean_mlp"] + df["sd_mlp"], color="C0", alpha=0.15)

# Transformer
# lower2 = np.maximum(df["mean_trans"] - df["sd_trans"], eps)
# upper2 = np.maximum(df["mean_trans"] + df["sd_trans"], eps)
plt.plot(df["step"], df["mean_trans_consumption"], color="#DD392C", lw=1.25, label="tf_consumption1")
plt.plot(df["step"], df["mean_trans_production"], color="#5E93B5", lw=1.25, label="tf_production1",linestyle='--')
# plt.fill_between(df["step"], df["mean_trans"] - df["sd_trans"], df["mean_trans"] + df["sd_trans"], color="C1", alpha=0.15)

plt.yscale("log")
plt.xlim(left=0)
# 不要 plt.ylim(bottom=0)；如需下界，用正数：
# plt.ylim(bottom=eps)

plt.xlabel("episode")
plt.ylabel("企业每百回合平均收益")
plt.title("MLP vs Transformer (mean across seeds)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("enterprise_income_100.svg", format="svg", bbox_inches="tight")
plt.show()