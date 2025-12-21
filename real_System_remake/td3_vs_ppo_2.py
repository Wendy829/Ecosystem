import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 画“同预算对比 + 全程对比”两子图 ============
# ============ 1) 读数据 ============
# 你的表应包含三列：step, TD3, PPO
path = r"C:\Users\lyk\Desktop\test.xlsx"  # 改成你的文件（也可以是 .xls）
df = pd.read_excel(path)

# 确保列是数值（防止Excel里混入文本）
df["step"] = pd.to_numeric(df["step"], errors="coerce")
df["TD3"] = pd.to_numeric(df["TD3_enterprise"], errors="coerce")
df["PPO"] = pd.to_numeric(df["PPO_enterprise"], errors="coerce")

# ============ 2) step -> episode（窗口中心）===========
# step=0 对应回合[0,99]，用窗口中心 50 更自然
WINDOW = 100
df["episode_center"] = df["step"] * WINDOW + WINDOW / 2  # 50,150,250,...

# ============ 3) 计算各算法结束点 ============
td3_end = float(df.loc[df["TD3"].notna(), "episode_center"].max())
ppo_end = float(df.loc[df["PPO"].notna(), "episode_center"].max())

if np.isnan(td3_end) or np.isnan(ppo_end):
    raise ValueError("TD3 或 PPO 列全是空值，请检查数据文件与列名。")

# ============ 4) 画“同预算对比 + 全程对比”两子图 ============
COLOR = {"TD3": "C0", "PPO": "C1"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)

# ---- (a) Same budget：只看 0 ~ TD3_end ----
m_early = df["episode_center"].between(0, td3_end)
ax1.plot(df.loc[m_early, "episode_center"], df.loc[m_early, "TD3"], lw=2.0, color=COLOR["TD3"], label="TD3_enterprise")
ax1.plot(df.loc[m_early, "episode_center"], df.loc[m_early, "PPO"], lw=2.0, color=COLOR["PPO"], label="PPO_enterprise")
ax1.set_title("(a) Same training budget\n(0 ~ TD3 end)")
ax1.set_xlim(0, td3_end+100)
# ax1.set_ylim(0,105)
ax1.set_xlabel("episode (center of 100-episode window)")
ax1.set_ylabel("avg comprehensive income(per 100 episodes)")
ax1.set_yscale("log")

ax1.grid(alpha=0.25)
ax1.legend()

# ---- (b) Full：PPO 全程 + TD3 截止，并标注 TD3 ended ----
ax2.plot(df["episode_center"], df["PPO"], lw=2.0, color=COLOR["PPO"], label="PPO_enterprise")
ax2.plot(df["episode_center"], df["TD3"], lw=2.0, color=COLOR["TD3"], label="TD3_enterprise")

# 竖线：TD3 训练结束
ax2.axvline(td3_end, color="black", ls="--", lw=1.2)

# 灰色区域：TD3 无数据区（防误读）
ax2.axvspan(td3_end, ppo_end, color="gray", alpha=0.12, lw=0)

# 文本标注（放在图上方一点点）
y_top = np.nanmax([df["TD3"].max(), df["PPO"].max()])
ax2.text(td3_end, y_top, "  TD3 ended", va="top", ha="left", fontsize=10)

ax2.set_title("(b) Full training\n(PPO longer)")
ax2.set_xlim(0, ppo_end)
ax2.set_xlabel("episode (center of 100-episode window)")
ax2.grid(alpha=0.25)
ax2.legend()
ax2.set_yscale("log")
plt.tight_layout()

# ============ 5) 保存/显示 ============
plt.savefig("survival_td3_ppo_compare.svg", format="svg", bbox_inches="tight")
plt.show()