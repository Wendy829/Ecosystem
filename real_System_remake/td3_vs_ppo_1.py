import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mticker
# 读你的表（csv/excel都行，下面以excel为例）
df = pd.read_excel(r"C:\Users\lyk\Desktop\test.xlsx")  # 改路径

WINDOW = 100
df["episode_center"] = df["step"] * WINDOW + WINDOW / 2  # step=0 -> 50 (对应0-99窗口中心)

fig, ax = plt.subplots(figsize=(8.5, 4.8))

ax.plot(df["episode_center"], df["PPO_enterprise"], lw=1.8, color="C1", label="PPO_enterprise")
ax.plot(df["episode_center"], df["TD3_enterprise"], lw=1.8, color="C0", label="TD3_enterprise")

ax.set_xlabel("episode (center of 100-episode window)")
ax.set_ylabel("avg comprehensive income (per 100 episodes)")
ax.grid(alpha=0.25)
ax.legend()

# TD3 step到45 => 最后一个窗口中心大约 45*100+50 = 4550
td3_end = int(df.loc[df["TD3"].notna(), "episode_center"].max())

axins = inset_axes(ax, width="45%", height="45%", loc="lower right", borderpad=2.5)
axins.plot(df["episode_center"], df["PPO_enterprise"], lw=1.4, color="C1")
axins.plot(df["episode_center"], df["TD3_enterprise"], lw=1.4, color="C0")

# 1) inset 的 x 只看 TD3 区间
axins.set_xlim(0, td3_end)

# 2) inset 的 y 单独按“0~td3_end”这段数据自动设范围（关键）
m = df["episode_center"].between(0, td3_end)
ymin = np.nanmin([df.loc[m, "PPO_enterprise"].min(), df.loc[m, "TD3_enterprise"].min()])
ymax = np.nanmax([df.loc[m, "PPO_enterprise"].max(), df.loc[m, "TD3_enterprise"].max()])
pad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
axins.set_ylim(ymin - pad, ymax + pad)

# 3) 关闭科学计数法（避免出现 1e7 + 0,1,2,3 这种）
axins.ticklabel_format(style="plain", axis="y", useOffset=False)
axins.yaxis.set_major_locator(mticker.MaxNLocator(4))

axins.grid(alpha=0.2)

ax.set_yscale("log")
# plt.tight_layout()
plt.savefig("td3_vs_ppo.svg", format="svg", bbox_inches="tight")
plt.show()
#
#
# x = df["episode_center"].to_numpy()
# x = np.maximum(x, 1)  # log轴避免0
#
# fig, ax = plt.subplots(figsize=(8.5, 4.8))
# ax.plot(x, df["PPO_enterprise"], lw=1.8, color="C1", label="PPO_enterprise")
# ax.plot(x, df["TD3_enterprise"], lw=1.8, color="C0", label="TD3_enterprise")
#
# ax.set_xscale("log")
# ax.set_yscale("log")
#
# ax.set_xlabel("episode (log scale)")
# ax.set_ylabel("avg comprehensive income (per 100 episodes)")
# plt.title("PPO vs TD3")
#
# ax.grid(alpha=0.25)
# ax.legend()
# plt.tight_layout()
# plt.savefig("td3_vs_ppo.svg", format="svg", bbox_inches="tight")
#
# plt.show()