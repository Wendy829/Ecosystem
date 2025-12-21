import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mean_sd_common(df, seed_cols, step_col="step"):
    # 处理缺失值：只保留 seed_cols 都不为 NaN 的行
    d = df[[step_col] + seed_cols].dropna(subset=seed_cols).copy()
    x = d[step_col].to_numpy()
    Y = d[seed_cols].to_numpy(float)
    mean = Y.mean(axis=1)
    sd = Y.std(axis=1, ddof=1)
    return x, mean, sd
# 1) 读两份数据
mlp = pd.read_excel(r"C:\Users\lyk\Desktop\mlp_100.xlsx")      # 或 pd.read_csv(...)
trans = pd.read_excel(r"C:\Users\lyk\Desktop\tf_100.xlsx")  # 或 pd.read_csv(...)

# 2) 每份数据各自的3个seed列名
mlp_seed_cols = ["MLP_891", "MLP_105", "MLP_117"]
trans_seed_cols = ["tf_891", "tf_105", "tf_117"]

# 3) 算 mean/sd
x1, m1, s1 = mean_sd_common(mlp, mlp_seed_cols)
x2, m2, s2 = mean_sd_common(trans, trans_seed_cols)

# 4) 齐step：用 merge 取交集，防止两边 step 不完全一致
#用x1,m1,s1构造新的表格，新表格理由step、mean_mlp、sd_mlp三列
mlp_ms = pd.DataFrame({"step": x1, "mean_mlp": m1, "sd_mlp": s1})
trans_ms = pd.DataFrame({"step": x2, "mean_trans": m2, "sd_trans": s2})
#合并两个表格，按step列对齐，how="inner"：内连接（取交集）
df = mlp_ms.merge(trans_ms, on="step", how="inner").sort_values("step")

# 5) 画在同一张图
plt.figure(figsize=(8,4))

plt.plot(df["step"], df["mean_mlp"], color="C0", linewidth=2, label="MLP_PPO mean")
plt.fill_between(df["step"],
                 df["mean_mlp"] - df["sd_mlp"],
                 df["mean_mlp"] + df["sd_mlp"],
                 color="C0", alpha=0.2)

plt.plot(df["step"], df["mean_trans"], color="C1", linewidth=2, label="Transformer_PPO mean")
plt.fill_between(df["step"],
                 df["mean_trans"] - df["sd_trans"],
                 df["mean_trans"] + df["sd_trans"],
                 color="C1", alpha=0.2)

plt.xlabel("episode")
plt.ylabel("survival days")
plt.title("MLP vs Transformer (mean ± SD across seeds)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.savefig("survival_days_100.svg", format="svg", bbox_inches="tight")
plt.show()