import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============ 1) 读Excel ============
path = r"C:\Users\lyk\Desktop\data\merged_limday100.xls"  # 改成你的文件
df = pd.read_excel(path)

STEP_COL = "step"

# ============ 2) 配色/样式 ============
STYLE = {
    "MLP": dict(color="C1"),   # 雾蓝灰
    "tf":  dict(color="C0"),   # 豆沙粉
}
# MARK_EVERY = 30

# ============ 3) 指标列表，加一个 use_log 控制是否取log轴 ============
METRICS = [
    ("每一百回合平均存活天数", "Avg survival days (per 100 ep)", False),  # 第1张：不log
    ("每百回合/综合收益/生产企业1", "Production enterprise 1 (comprehensive income)", True),
    ("每百回合/综合收益/消费企业1", "Consumption enterprise 1 (comprehensive income)", True),
    ("每百回合/累计利润/银行", "Bank (cumulative profit)", True),
]
SYMLINTHRESH = 1.0

#自动找到某算法某指标对应的列，不是step列，列名以algo_prefix开头，且包含metric_key
def find_cols(df, algo_prefix: str, metric_key: str):
    cols = [
        c for c in df.columns
        if c != STEP_COL
        and c.startswith(algo_prefix + "_")
        and (metric_key in c)
    ]
    return cols

#在子图上画某个指标
def plot_metric(ax, metric_key, title, use_log=False):
    x = df[STEP_COL]

    for algo in ["MLP", "tf"]:
        cols = find_cols(df, algo, metric_key)
        if len(cols) == 0:
            ax.text(0.02, 0.95, f"{algo}: no columns found", transform=ax.transAxes, va="top")
            continue

        if len(cols) != 3:
            print(f"[WARN] {algo} metric='{metric_key}' matched {len(cols)} cols:\n  " + "\n  ".join(cols))

        y = df[cols].apply(pd.to_numeric, errors="coerce")

        mean = y.mean(axis=1)
        std = y.std(axis=1)

        # log轴：要求 y > 0；把 <=0 的点置为 NaN（会断线）
        if use_log:
            bad = (mean <= 0) | ((mean - std) <= 0) | ((mean + std) <= 0)
            if bad.any():
                print(f"[WARN] {algo} metric='{metric_key}' has non-positive values; they will be skipped on log scale.")
            mean = mean.mask(mean <= 0, np.nan)
            lower = (mean - std).mask((mean - std) <= 0, np.nan)
            upper = (mean + std).mask((mean + std) <= 0, np.nan)
        else:
            lower = mean - std
            upper = mean + std

        st = STYLE[algo]
        ax.plot(
            x, mean,
            label=f"{algo} (mean±std, n={len(cols)})",
            color=st["color"], lw=1.6,
            # marker=st["marker"], markersize=4,
            # markevery=MARK_EVERY,
            # markerfacecolor="white", markeredgecolor=st["color"], markeredgewidth=1.0
        )
        ax.fill_between(x, lower, upper, color=st["color"], alpha=0.18, linewidth=0)

    ax.set_title(title)
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.25)
    ax.legend()

    if use_log:
        ax.set_yscale("log")

# ============ 4) 画 2x2 ============
# fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
# axes = axes.ravel()
#
# for ax, (metric_key, title, use_log) in zip(axes, METRICS):
#     plot_metric(ax, metric_key, title, use_log=use_log)
#
# plt.tight_layout()
# plt.savefig("100.svg", format="svg", bbox_inches="tight")
# plt.show()
import matplotlib.ticker as mticker

# ============ 4) 画 2x2 ============
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
axes = axes.ravel()

for ax, (metric_key, title, use_log) in zip(axes, METRICS):
    plot_metric(ax, metric_key, title, use_log=use_log)

# ---- 强制所有子图都显示 x 轴刻度（sharex 时默认上排不显示）----
for ax in axes:
    ax.tick_params(axis="x", which="both", labelbottom=True)

# ---- 让 x 从 0 贴着左边框开始（去掉默认padding）----
xmin = 0
xmax = df[STEP_COL].max()+10
for ax in axes:
    ax.set_xlim(xmin, xmax)
    ax.margins(x=0)  # 去掉x方向留白

# ---- 统一设置横坐标刻度：比如每50一个（你也可以改成100）----
xtick_step = 50
xticks = list(range(0, int(xmax) + 1, xtick_step))
for ax in axes:
    ax.set_xticks(xticks)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(25))  # 可选：小刻度每25

plt.tight_layout()
plt.savefig("100.svg", format="svg")

plt.show()