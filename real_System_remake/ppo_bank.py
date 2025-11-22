from collections import deque

import numpy as np

from Agent import Config_PPO
from Agent.PPO import PPO
import warnings

warnings.filterwarnings('ignore')

io_path = 'io/'
ex_path = io_path + 'bank_nnu/'
logs_path = ex_path + 'logs'
session_path = ex_path + 'session'
model_filename = ex_path + 'model'

clustered_devices = None


class bank_nnu:
    def __init__(self, config: Config_PPO):
        self.current_seq_state = None
        self.scope = config.scope
        self.bank = PPO(config=config)  # 生成num个mod
        # === 新增：历史状态队列 ===
        self.seq_len = config.seq_len if hasattr(config, 'seq_len') else 1
        self.state_window = deque(maxlen=self.seq_len)

    def choose_action(self, state):
        # --- 【新增】函数，替换旧的 run_enterprise ---
        # action = self.bank.choose_action(state)

        # 1. 更新状态窗口
        # 【修改】加入冷启动逻辑
        # 如果是新回合第一步(window为空)，用当前状态填满整个窗口
        # 这样 Transformer 就会认为"过去一直是这个状态"，而不是"过去全是0"
        if len(self.state_window) == 0:
            for _ in range(self.seq_len):
                self.state_window.append(state)
        else:
            self.state_window.append(state)

        # 2. 制作 Transformer 需要的 "State"
        # shape: (10, 35)
        seq_state = np.array(self.state_window)

        # 3. 传给 Actor 选择动作
        # 注意：这里要把 seq_state 存下来！不仅仅是 raw_state
        self.current_seq_state = seq_state

        return self.bank.choose_action(seq_state)

    def choose_action_deterministic(self, state):
        action = self.bank.choose_action_deterministic(state)

        return action

    def store_transition(self, state,mu,sigma,action, logprob, reward, is_terminal, next_value, nonterminal):  # CHANGED
        self.bank.store_transition(self.current_seq_state,mu,sigma,action, logprob, reward, is_terminal, next_value, nonterminal)

    def get_value(self, state):  # NEW
        return self.bank.get_value(state)

    def learn(self, last_value):
        # --- 【新增】函数 ---
        self.bank.learn(last_value, agent_type=self.scope)

    def clear_memory(self):
        # --- 【新增】函数 ---
        self.bank.clear_memory()

    def log(self):
        # var = self.bank.get_var()
        critic_loss, actor_loss = self.bank.get_loss()
        avg_entropy, avg_clip_frac = self.bank.get_test_indicator()
        return critic_loss, actor_loss, avg_entropy, avg_clip_frac

    def reset_window(self):
        self.state_window.clear()