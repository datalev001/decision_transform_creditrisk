import os
import logging
import torch
import random
import numpy as np
import pandas as pd

# Configuration and read data
CONFIG = {
    "SEED": 123,
    "RESULT_DIR": "artifacts_credit_rl_full",
    "DATA_CSV": "credit_panel.csv",
    "DT_WINDOW": 20,
    "DT_EPOCHS": 60,
    "DT_BATCH": 64,
    "DT_PATIENCE": 6,
    "GRU_EPOCHS": 40,
    "GRU_PATIENCE": 5,
    "GRU_LOOKBACK": 6,
    "CQL_STEPS": 15_000,
    "CQL_BATCH": 512,
    "CQL_ALPHA": 0.5,
    "CQL_LOG_INT": 500,
    "CANDIDATE_ACTIONS": 32,
    "GAMMA": 0.98,
    "AWAC_STEPS": 5_000,
    "AWAC_LAMBDA": 0.07,
    "AWAC_LOG_INT": 250,
    "ACTION_LOW": -0.15,
    "ACTION_HIGH": 0.15
}

os.makedirs(CONFIG["RESULT_DIR"], exist_ok=True)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
random.seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])
torch.manual_seed(CONFIG["SEED"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")

# Load panel data
panel_sim = pd.read_csv(CONFIG["DATA_CSV"])
logging.info(f"Loaded panel data: {panel_sim.shape}")


# ==========================================
# 2. Action-Aware Trajectory Reconstruction
# ==========================================

def behavior_policy_simple(prev_pd, pd_trend, util, segment):
    """
    Heuristic behavior policy that decides Δlimit% given current risk/utilization:
        - Shrink limits if PD rising & utilization high
        - Modestly increase if PD falling & utilization moderate
        - Otherwise small neutral drift
    """
    if pd_trend > 0.01 and util > 0.7:
        return np.random.uniform(-0.12, -0.02)
    elif pd_trend < -0.005 and util < 0.6:
        return np.random.uniform(0.015, 0.08)
    else:
        base = np.random.uniform(-0.02, 0.03)
        # Segment slight bias
        if segment == "Prime":
            base += 0.005
        elif segment == "Subprime":
            base -= 0.005
        return float(base)


def evolve_state(state: dict, action: float):
    """
    Deterministic + stochastic evolution:
    - credit_limit updated by action
    - utilization mean reversion + effect of action
    - PD depends on utilization, payment ratio, macro, action side-effects
    - Payment ratio anti-correlated to utilization
    - LGD, EAD heuristics
    """
    new_state = dict(state)  # shallow copy
    # Update credit limit
    new_state["credit_limit"] = float(np.clip(state["credit_limit"] * (1 + action), 500, 100_000))

    # Utilization dynamics
    util = state["utilization"]
    util_mu = util + 0.08 * action - 0.05 * (util - 0.5)  # push toward mid + action influence
    new_util = np.clip(util_mu + np.random.normal(0, 0.05), 0.01, 0.98)
    new_state["utilization"] = float(new_util)

    # Payment ratio (higher if lower utilization)
    pay_mu = 0.40 + 0.3 * (0.5 - new_util)
    new_state["payment_ratio"] = float(np.clip(np.random.normal(pay_mu, 0.08), 0.05, 0.95))

    # Point-in-Time PD (proxy)
    seg = state["segment"]
    seg_base = 0.02 if seg == "Prime" else (0.05 if seg == "NearPrime" else 0.11)
    pd_point = seg_base \
               + 0.18 * (new_util - 0.5) \
               + 0.10 * (new_state["payment_ratio"] < 0.2) \
               + 0.35 * (state.get("dpd", 0) >= 30) \
               + 0.4 * (state["macro_unemp"] - 0.04) \
               + 0.05 * max(action, 0) \
               + np.random.normal(0, 0.01)
    pd_point = float(np.clip(pd_point, 0.001, 0.55))
    new_state["pd_point"] = pd_point

    # DPD (probabilistic)
    base_dpd_prob = 0.01 + 0.04 * (seg == "Subprime") + 0.03 * (new_util > 0.85)
    dpd = 0
    if np.random.rand() < base_dpd_prob:
        dpd = np.random.choice([0, 15, 30, 60, 90], p=[0.55, 0.2, 0.15, 0.07, 0.03])
    new_state["dpd"] = int(dpd)

    # LGD & EAD
    lgd = np.clip(0.35 + 0.2 * (seg != "Prime") + 0.15 * (new_util > 0.7)
                  + np.random.normal(0, 0.03), 0.2, 0.9)
    new_state["lgd_est"] = float(lgd)
    new_state["ead"] = float(new_state["credit_limit"] * new_util)

    return new_state


def compute_reward(state: dict, action: float):
    """
    Monthly risk-adjusted profit proxy.
        interest_income = util * limit * 0.18 / 12
        funding_cost    = util * limit * 0.05 / 12
        expected_loss   = pd * lgd * ead / 12
        capital_charge  = 0.10 * pd * ead * 0.08
        reward = (interest_income - funding_cost) - expected_loss - capital_charge
    Scale by 1/1000 for numeric stability.
    """
    util = state["utilization"]
    limit_val = state["credit_limit"]
    pd_point = state["pd_point"]
    lgd = state["lgd_est"]
    ead = state["ead"]

    interest_income = util * limit_val * (0.18 / 12)
    funding_cost = util * limit_val * (0.05 / 12)
    expected_loss = pd_point * lgd * ead * (1 / 12)
    capital_charge = 0.10 * pd_point * ead * 0.08

    reward = (interest_income - funding_cost) - expected_loss - capital_charge
    return float(reward / 1000.0)


def build_action_influenced_trajectories(panel: pd.DataFrame) -> pd.DataFrame:
    """
    For each customer:
        - Initialize first month's dynamic fields
        - Roll forward applying behavior policy + causal evolution
        - Store per-month realized reward & action
    Returns a fully simulated panel (one row per (customer, month)).
    """
    simulated_rows = []
    for cid, grp in panel.groupby("customer_id"):
        grp = grp.sort_values("month").reset_index(drop=True)
        # Initialize first month with neutral PD / LGD (will be overwritten by evolve)
        # Seed state from row 0
        base_row = grp.loc[0].to_dict()
        # Add placeholders
        state = {
            "customer_id": cid,
            "month": 0,
            "segment": base_row["segment"],
            "base_fico": base_row["base_fico"],
            "annual_income": base_row["annual_income"],
            "region": base_row["region"],
            "product_type": base_row["product_type"],
            "credit_limit": base_row["credit_limit"],
            "utilization": base_row["utilization"],
            "payment_ratio": base_row["payment_ratio"],
            "macro_unemp": base_row["macro_unemp"],
            "macro_rate": base_row["macro_rate"],
            "dpd": 0,
            "pd_point": 0.05,       # neutral initial seed
            "lgd_est": 0.45,
            "ead": base_row["credit_limit"] * base_row["utilization"]
        }
        prev_pd = state["pd_point"]
        # First month action = 0
        action = 0.0
        reward = compute_reward(state, action)
        state_record = dict(state)
        state_record["action"] = action
        state_record["reward"] = reward
        simulated_rows.append(state_record)

        for t in range(1, len(grp)):
            # PD trend estimate
            pd_trend = state["pd_point"] - prev_pd
            prev_pd = state["pd_point"]
            # Choose behavior action
            action = behavior_policy_simple(state["pd_point"], pd_trend, state["utilization"], state["segment"])
            action = float(np.clip(action, CONFIG["ACTION_LOW"], CONFIG["ACTION_HIGH"]))
            # Advance macro
            macro_unemp = 0.04 + 0.01 * math.sin(t / 12 * 2 * math.pi) + np.random.normal(0, 0.002)
            macro_rate = 0.02 + 0.005 * math.sin((t + 6) / 18 * 2 * math.pi) + np.random.normal(0, 0.0015)
            state["macro_unemp"] = macro_unemp
            state["macro_rate"] = macro_rate
            # Evolve state
            next_state = evolve_state(state, action)
            # Reward from *next* state snapshot (choice: post-update metrics)
            reward = compute_reward(next_state, action)
            next_state["action"] = action
            next_state["reward"] = reward
            next_state["customer_id"] = cid
            next_state["month"] = t
            simulated_rows.append(next_state)
            state = next_state

    sim_df = pd.DataFrame(simulated_rows)
    if len(sim_df) > CONFIG["TARGET_ROWS"]:
        sim_df = sim_df.iloc[:CONFIG["TARGET_ROWS"]].copy()
    return sim_df


panel_sim = build_action_influenced_trajectories(panel_raw)
logging.info(f"Simulated action-influenced panel rows: {len(panel_sim)}")

# =====================================
# 3. Feature Engineering & Forecasting
# =====================================

# Train/Val/Test split by customer to avoid leakage
all_customers = panel_sim["customer_id"].unique()
train_cust, test_cust = train_test_split(all_customers, test_size=0.2, random_state=CONFIG["SEED"])
val_cust, test_cust = train_test_split(test_cust, test_size=0.5, random_state=CONFIG["SEED"])

panel_sim["set"] = "train"
panel_sim.loc[panel_sim.customer_id.isin(val_cust), "set"] = "val"
panel_sim.loc[panel_sim.customer_id.isin(test_cust), "set"] = "test"

# Forecast target features
FORECAST_TARGETS = ["pd_point", "utilization"]

class ForecastDataset(Dataset):
    def __init__(self, df_sub: pd.DataFrame, feature: str, lookback: int):
        self.lookback = lookback
        xs, ys = [], []
        for _, g in df_sub.groupby("customer_id"):
            arr = g.sort_values("month")[feature].values.astype(np.float32)
            if len(arr) <= lookback:
                continue
            for t in range(lookback, len(arr)):
                xs.append(arr[t - lookback:t])
                ys.append(arr[t])
        self.x = torch.tensor(xs)  # (N,L)
        self.y = torch.tensor(ys).unsqueeze(-1)  # (N,1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class GRUForecaster(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.lin = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B,L)
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        h = out[:, -1]
        return self.lin(h)


def train_gru_forecaster(feature: str):
    lookback = CONFIG["GRU_LOOKBACK"]
    train_df = panel_sim[panel_sim.set == "train"]
    val_df = panel_sim[panel_sim.set == "val"]

    train_ds = ForecastDataset(train_df, feature, lookback)
    val_ds = ForecastDataset(val_df, feature, lookback)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Insufficient data for forecasting {feature}")

    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = GRUForecaster().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best = float("inf")
    wait = 0

    for epoch in range(CONFIG["GRU_EPOCHS"]):
        model.train()
        train_losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = F.mse_loss(pred, yb)
                val_losses.append(loss.item())
            val_loss = np.mean(val_losses)

        logging.info(f"[GRU {feature}] epoch={epoch} train={train_loss:.5f} val={val_loss:.5f}")

        if val_loss < best - 1e-5:
            best = val_loss
            wait = 0
            torch.save(model.state_dict(), os.path.join(CONFIG["RESULT_DIR"], f"gru_{feature}.pt"))
        else:
            wait += 1
            if wait >= CONFIG["GRU_PATIENCE"]:
                logging.info(f"[GRU {feature}] early stop.")
                break

    model.load_state_dict(torch.load(os.path.join(CONFIG["RESULT_DIR"], f"gru_{feature}.pt")))
    return model


forecast_models = {}
for ft in FORECAST_TARGETS:
    forecast_models[ft] = train_gru_forecaster(ft)


def add_forecast_columns(df: pd.DataFrame, feature: str, model: nn.Module):
    lookback = CONFIG["GRU_LOOKBACK"]
    preds_all = []
    model.eval()
    for _, g in df.groupby("customer_id"):
        g = g.sort_values("month").reset_index(drop=True)
        arr = g[feature].values.astype(np.float32)
        preds = [np.nan] * len(arr)
        for t in range(lookback, len(arr)):
            seq = torch.tensor(arr[t - lookback:t]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = model(seq).cpu().item()
            preds[t] = pred
        preds_all.extend(preds)
    df[f"fwd_{feature}"] = preds_all
    # Fill missing
    df[f"fwd_{feature}"] = df.groupby("customer_id")[f"fwd_{feature}"].bfill().ffill()
    return df

for ft in FORECAST_TARGETS:
    panel_sim = add_forecast_columns(panel_sim, ft, forecast_models[ft])

STATE_FEATURES = [
    "base_fico", "annual_income", "credit_limit", "utilization", "payment_ratio",
    "pd_point", "lgd_est", "ead", "dpd", "macro_unemp", "macro_rate",
    "fwd_pd_point", "fwd_utilization"
]

# Standardize continuous features
scaler = StandardScaler()
panel_sim[STATE_FEATURES] = scaler.fit_transform(panel_sim[STATE_FEATURES])

# =====================================
# 4. Build Offline Transition Buffer
# =====================================

@dataclass
class Transition:
    customer_id: int
    s: np.ndarray
    a: float
    r: float
    s_next: np.ndarray
    done: bool
    month: int


def build_transitions(df: pd.DataFrame) -> List[Transition]:
    transitions = []
    for cid, g in df.groupby("customer_id"):
        g = g.sort_values("month").reset_index(drop=True)
        for t in range(len(g) - 1):
            s_vec = g.loc[t, STATE_FEATURES].values.astype(np.float32)
            s_next_vec = g.loc[t + 1, STATE_FEATURES].values.astype(np.float32)
            a = float(g.loc[t, "action"])
            r = float(g.loc[t, "reward"])
            done = (t + 1 == len(g) - 1)
            transitions.append(Transition(
                customer_id=cid,
                s=s_vec,
                a=a,
                r=r,
                s_next=s_next_vec,
                done=done,
                month=int(g.loc[t, "month"])
            ))
    return transitions


train_df = panel_sim[panel_sim.set == "train"]
val_df = panel_sim[panel_sim.set == "val"]
test_df = panel_sim[panel_sim.set == "test"]

train_buf = build_transitions(train_df)
val_buf = build_transitions(val_df)
test_buf = build_transitions(test_df)

logging.info(f"Transitions: train={len(train_buf)} val={len(val_buf)} test={len(test_buf)}")

# ====================================
# 5. Decision Transformer Pretraining
# ====================================

def group_transitions_by_customer(buffer: List[Transition]) -> Dict[int, List[Transition]]:
    groups = {}
    for tr in buffer:
        groups.setdefault(tr.customer_id, []).append(tr)
    # sort by month for safety
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda x: x.month)
    return groups


def build_dt_sequences(buffer: List[Transition], window: int):
    groups = group_transitions_by_customer(buffer)
    S_list, A_list, RTG_list = [], [], []
    for cid, seq in groups.items():
        if len(seq) <= window:
            continue
        rewards = np.array([tr.r for tr in seq], dtype=np.float32)
        # sliding windows
        for i in range(len(seq) - window):
            sub = seq[i:i + window]
            s_arr = np.stack([t.s for t in sub])
            a_arr = np.stack([[t.a] for t in sub])
            r_arr = np.array([t.r for t in sub], dtype=np.float32)
            rtg = np.cumsum(r_arr[::-1])[::-1]
            rtg = (rtg - rtg.mean()) / (rtg.std() + 1e-6)
            S_list.append(s_arr)
            A_list.append(a_arr)
            RTG_list.append(rtg.reshape(-1, 1))
    S = np.array(S_list)
    A = np.array(A_list)
    RTG = np.array(RTG_list)
    return S, A, RTG


dt_states, dt_actions, dt_rtg = build_dt_sequences(train_buf, CONFIG["DT_WINDOW"])
logging.info(f"DT sequences built: {dt_states.shape}")

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim=1, d_model=128, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.state_embed = nn.Linear(state_dim, d_model)
        self.action_embed = nn.Linear(action_dim, d_model)
        self.rtg_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Embedding(CONFIG["DT_WINDOW"], d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, action_dim)
        )

    def forward(self, S, A, RTG):
        B, T, _ = S.shape
        x = self.state_embed(S) + self.action_embed(A) + self.rtg_embed(RTG) + \
            self.pos_embed(torch.arange(T, device=S.device)).unsqueeze(0)
        h = self.encoder(x)
        out = self.head(h)
        return out  # per timestep action prediction


class DTDataset(Dataset):
    def __init__(self, S, A, RTG):
        self.S = torch.tensor(S, dtype=torch.float32)
        self.A = torch.tensor(A, dtype=torch.float32)
        self.R = torch.tensor(RTG, dtype=torch.float32)

    def __len__(self):
        return len(self.S)

    def __getitem__(self, idx):
        return self.S[idx], self.A[idx], self.R[idx]


dt_dataset = DTDataset(dt_states, dt_actions, dt_rtg)
dt_loader = DataLoader(dt_dataset, batch_size=CONFIG["DT_BATCH"], shuffle=True)

dt_model = DecisionTransformer(state_dim=dt_states.shape[-1]).to(DEVICE)
dt_opt = torch.optim.Adam(dt_model.parameters(), lr=1e-4)

best_dt = float("inf")
wait = 0
dt_losses = []

for epoch in range(CONFIG["DT_EPOCHS"]):
    dt_model.train()
    ep_losses = []
    for S_b, A_b, R_b in dt_loader:
        S_b = S_b.to(DEVICE)
        A_b = A_b.to(DEVICE)
        R_b = R_b.to(DEVICE)
        pred = dt_model(S_b, A_b, R_b)
        loss = F.mse_loss(pred, A_b)
        dt_opt.zero_grad()
        loss.backward()
        dt_opt.step()
        ep_losses.append(loss.item())
    ep_loss = np.mean(ep_losses)
    dt_losses.append(ep_loss)
    logging.info(f"[DT] epoch={epoch} loss={ep_loss:.6f}")
    if ep_loss < best_dt - 1e-5:
        best_dt = ep_loss
        wait = 0
        torch.save(dt_model.state_dict(), os.path.join(CONFIG["RESULT_DIR"], "dt_model.pt"))
    else:
        wait += 1
        if wait >= CONFIG["DT_PATIENCE"]:
            logging.info("[DT] early stop.")
            break

plt.figure(figsize=(6, 4))
plt.plot(dt_losses)
plt.title("Decision Transformer Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.savefig(os.path.join(CONFIG["RESULT_DIR"], "dt_loss_curve.png"))
plt.show() 
plt.close()

dt_model.load_state_dict(torch.load(os.path.join(CONFIG["RESULT_DIR"], "dt_model.pt")))
logging.info(f"[DT] best_loss={best_dt:.6f}")

# =======================================
# 6. Conservative Q-Learning (Twin Q)
# =======================================
# =======================================
# 6. Conservative Q‑Learning (Twin Q)
# =======================================

import copy, logging, os, matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from typing import List

# -------------------------------------------------------------------
# Helper dataclass used earlier in the script
# -------------------------------------------------------------------
@dataclass
class Transition:
    s: np.ndarray        # state   (state_dim,)
    a: float             # action  (scalar)
    r: float             # reward
    s_next: np.ndarray   # next state
    done: bool           # terminal flag
# -------------------------------------------------------------------

state_dim = len(STATE_FEATURES)

class TwinQ(nn.Module):
    """Q1 / Q2 twin critics – MLP style."""
    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        def _branch():
            return nn.Sequential(
                nn.Linear(state_dim + 1, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),       nn.ReLU(),
                nn.Linear(hidden, 1)
            )
        self.q1 = _branch()
        self.q2 = _branch()

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        """
        s : (B, state_dim)
        a : (B, 1)
        returns two tensors each (B, 1)
        """
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)


twin_q      = TwinQ(state_dim).to(DEVICE)
twin_q_tgt  = copy.deepcopy(twin_q).eval()        # Polyak target
cql_opt     = torch.optim.Adam(twin_q.parameters(), lr=3e-4)

# -------------------------------------------------------------------
# Mini‑batch sampler
# -------------------------------------------------------------------
def sample_batch(buffer: List[Transition], batch_size: int):
    idx   = np.random.randint(0, len(buffer), size=batch_size)
    batch = [buffer[i] for i in idx]

    s  = torch.tensor(np.stack([b.s       for b in batch]), dtype=torch.float32, device=DEVICE)
    a  = torch.tensor(np.stack([[b.a]     for b in batch]), dtype=torch.float32, device=DEVICE)
    r  = torch.tensor(np.stack([[b.r]     for b in batch]), dtype=torch.float32, device=DEVICE)
    s2 = torch.tensor(np.stack([b.s_next for b in batch]), dtype=torch.float32, device=DEVICE)
    d  = torch.tensor(np.stack([[b.done]  for b in batch]), dtype=torch.float32, device=DEVICE)

    return s, a, r, s2, d


# -------------------------------------------------------------------
# ε‑greedy arg‑max helper for CQL target
# -------------------------------------------------------------------
def policy_argmax_q(s_batch: torch.Tensor) -> torch.Tensor:
    """
    Greedy action w.r.t. *target* twin Q.
    Returns tensor (B,1)  – NOT (B,1,1)  (keeps shape compatible).
    """
    B = s_batch.size(0)

    cand = torch.rand(
        B, CONFIG["CANDIDATE_ACTIONS"], 1, device=DEVICE
    ) * (CONFIG["ACTION_HIGH"] - CONFIG["ACTION_LOW"]) + CONFIG["ACTION_LOW"]

    # Evaluate both critics on all candidates
    s_rep = s_batch.unsqueeze(1).expand(-1, CONFIG["CANDIDATE_ACTIONS"], -1)
    q1c, q2c = twin_q_tgt(s_rep.reshape(-1, state_dim),
                          cand.reshape(-1, 1))
    qc = torch.min(q1c, q2c).view(B, CONFIG["CANDIDATE_ACTIONS"])

    best_idx = qc.argmax(dim=1)                         # (B,)
    best_a   = cand[torch.arange(B), best_idx]          # (B,1)  <<<<<< no extra dim
    return best_a                                       


# -------------------------------------------------------------------
# Main CQL training loop
# -------------------------------------------------------------------
cql_total, cql_bell, cql_reg_list = [], [], []

for step in range(1, CONFIG["CQL_STEPS"] + 1):

    s, a, r, s2, d = sample_batch(train_buf, CONFIG["CQL_BATCH"])

    # ----------  target Q value ----------
    with torch.no_grad():
        a2       = policy_argmax_q(s2)                        # (B,1)
        q1_t, q2_t = twin_q_tgt(s2, a2)
        q_t      = torch.min(q1_t, q2_t)
        target   = r + CONFIG["GAMMA"] * (1 - d) * q_t

    # ----------  current Q estimates ----------
    q1, q2 = twin_q(s, a)
    bellman_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

    # ----------  CQL regulariser ----------
    rand_a = torch.rand_like(a.repeat(1, CONFIG["CANDIDATE_ACTIONS"])) \
             * (CONFIG["ACTION_HIGH"] - CONFIG["ACTION_LOW"]) + CONFIG["ACTION_LOW"]
    rand_a  = rand_a.view(-1, 1)
    s_rep   = s.repeat_interleave(CONFIG["CANDIDATE_ACTIONS"], dim=0)

    rq1, rq2 = twin_q(s_rep, rand_a)
    rq       = torch.min(rq1, rq2).view(-1, CONFIG["CANDIDATE_ACTIONS"])

    logsumexp_q = torch.logsumexp(rq, dim=1).mean()          # 𝔼_a Q under uniform
    data_q      = torch.min(q1, q2).mean()                   # 𝔼_a~buffer Q
    cql_reg     = (logsumexp_q - data_q) * CONFIG["CQL_ALPHA"]

    # ----------  total loss & optimisation ----------
    loss = bellman_loss + cql_reg

    cql_opt.zero_grad()
    loss.backward()
    cql_opt.step()

    # ----------  target update ----------
    with torch.no_grad():
        polyak = 0.995
        for p, pt in zip(twin_q.parameters(), twin_q_tgt.parameters()):
            pt.data.mul_(polyak).add_((1 - polyak) * p.data)

    # ----------  logging ----------
    if step % CONFIG["CQL_LOG_INT"] == 0:
        logging.info(
            f"[CQL] step={step}  loss={loss.item():.4f}  "
            f"bell={bellman_loss.item():.4f}  reg={cql_reg.item():.4f}"
        )
        cql_total.append(loss.item())
        cql_bell.append(bellman_loss.item())
        cql_reg_list.append(cql_reg.item())


# -------------------------------------------------------------------
# Quick diagnostic plot
# -------------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.plot(cql_total, label="Total")
plt.plot(cql_bell, label="Bellman")
plt.plot(cql_reg_list, label="Reg")
plt.title("CQL Loss Components")
plt.xlabel(f"×{CONFIG['CQL_LOG_INT']} steps")
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(CONFIG["RESULT_DIR"], "cql_losses.png"))
plt.show()
plt.close()


def cql_policy(s_vec: np.ndarray):
    s_t = torch.tensor(s_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        a = policy_argmax_q(s_t)
    return float(a.cpu().item())

# =======================================
# 7. AWAC Fine-Tuning
# =======================================

class Actor(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh()
        )

    def forward(self, s):
        # Scale to action bounds
        return self.net(s) * (CONFIG["ACTION_HIGH"] - CONFIG["ACTION_LOW"]) / 2.0 + \
               (CONFIG["ACTION_HIGH"] + CONFIG["ACTION_LOW"]) / 2.0


actor = Actor(state_dim).to(DEVICE)

# Warm start from dataset (behavior cloning to speed stabilization)
bc_opt = torch.optim.Adam(actor.parameters(), lr=5e-4)
for _ in range(300):
    s_b, a_b, *_ = sample_batch(train_buf, 256)
    pred_a = actor(s_b)
    loss_bc = F.mse_loss(pred_a, a_b)
    bc_opt.zero_grad()
    loss_bc.backward()
    bc_opt.step()
logging.info("[AWAC] Warm-start BC done.")

awac_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
awac_losses = []
awac_wmeans = []

def compute_advantage(s_batch, a_batch):
    with torch.no_grad():
        q1_b, q2_b = twin_q(s_batch, a_batch)
        a_pi = actor(s_batch)
        q1_pi, q2_pi = twin_q(s_batch, a_pi)
        q_b = torch.min(q1_b, q2_b)
        q_pi = torch.min(q1_pi, q2_pi)
        adv = q_b - q_pi
        # Normalize
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-6)
    return adv_norm

for step in range(1, CONFIG["AWAC_STEPS"] + 1):
    s_b, a_b, r_b, s2_b, d_b = sample_batch(train_buf, 512)
    adv = compute_advantage(s_b, a_b)
    weights = torch.clamp(torch.exp(adv / CONFIG["AWAC_LAMBDA"]), max=40.0)
    pred_a = actor(s_b)
    loss_awac = (weights * (pred_a - a_b).pow(2)).mean()
    awac_opt.zero_grad()
    loss_awac.backward()
    awac_opt.step()

    if step % CONFIG["AWAC_LOG_INT"] == 0:
        awac_losses.append(loss_awac.item())
        awac_wmeans.append(weights.mean().item())
        logging.info(f"[AWAC] step={step} loss={loss_awac.item():.5f} w_mean={weights.mean().item():.3f}")

plt.figure(figsize=(7, 4))
plt.plot(awac_losses, label="AWAC Loss")
plt.title("AWAC Fine-tune Loss")
plt.xlabel(f"×{CONFIG['AWAC_LOG_INT']} steps")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(CONFIG["RESULT_DIR"], "awac_loss.png"))
plt.show()
plt.close()

def awac_policy(s_vec: np.ndarray):
    with torch.no_grad():
        s_t = torch.tensor(s_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return float(actor(s_t).cpu().item())

# ==============================
# 8. Evaluation
# ==============================

def evaluate_policy(buffer: List[Transition], policy_fn, label: str):
    """
    Evaluate discounted return, risk statistics.
    Construct episodes per customer.
    """
    groups = group_transitions_by_customer(buffer)
    ep_returns = []
    all_rewards = []
    all_actions = []
    pd_values = []

    for cid, seq in groups.items():
        G = 0.0
        disc = 1.0
        for tr in seq:
            # Use recorded state, replace action with policy_fn if needed for diagnostic
            # Here we use *recorded reward* (counterfactual simulation omitted).
            a = policy_fn(tr.s)
            all_actions.append(a)
            all_rewards.append(tr.r)
            # pick PD feature index (after standardization the absolute value is not PD but still relative risk)
            pd_idx = STATE_FEATURES.index("pd_point")
            pd_values.append(tr.s[pd_idx])
            G += disc * tr.r
            disc *= CONFIG["GAMMA"]
        ep_returns.append(G)

    mean_ret = float(np.mean(ep_returns))
    std_ret = float(np.std(ep_returns) + 1e-9)
    sharpe = mean_ret / std_ret if std_ret > 0 else 0.0
    rewards_arr = np.array(ep_returns)
    var95 = float(np.percentile(rewards_arr, 5))
    cvar95 = float(rewards_arr[rewards_arr <= np.percentile(rewards_arr, 5)].mean())
    result = dict(
        policy=label,
        episodes=len(ep_returns),
        mean_return=mean_ret,
        std_return=std_ret,
        sharpe=sharpe,
        var95=var95,
        cvar95=cvar95,
        mean_pd_feature=float(np.mean(pd_values)),
        action_mean=float(np.mean(all_actions)),
        action_std=float(np.std(all_actions))
    )
    logging.info(f"[EVAL {label}] "
                 f"mean={mean_ret:.4f} sharpe={sharpe:.3f} VaR95={var95:.4f} CVaR95={cvar95:.4f}")
    return result


# Behavior baseline: use *recorded* action (identity function)
def behavior_policy_eval(s_vec: np.ndarray):
    # We do not re-compute; just placeholder identity (action not used to recompute reward)
    return 0.0  # or could decode approximate from dataset distribution if needed


eval_results = []
eval_results.append(evaluate_policy(test_buf, behavior_policy_eval, "Behavior"))
eval_results.append(evaluate_policy(test_buf, cql_policy, "CQL"))
eval_results.append(evaluate_policy(test_buf, awac_policy, "CQL+AWAC"))

# Action distributions (diagnostic)
def plot_action_distribution(policy_fn, name: str):
    acts = [policy_fn(tr.s) for tr in test_buf[:5000]]
    plt.figure(figsize=(5, 3))
    plt.hist(acts, bins=40, alpha=0.75)
    plt.title(f"Action Dist: {name}")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(CONFIG["RESULT_DIR"], f"actions_{name}.png"))
    plt.close()

plot_action_distribution(cql_policy, "cql_greedy")
plot_action_distribution(awac_policy, "awac")

with open(os.path.join(CONFIG["RESULT_DIR"], "eval_results.json"), "w") as f:
    json.dump(eval_results, f, indent=2)

# PD bucket analysis
def pd_bucket_report(buffer: List[Transition], policy_fn, label: str):
    pd_idx = STATE_FEATURES.index("pd_point")
    buckets = [( -5, -1.0), (-1.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 5.0)]
    # (Note: pd_point standardized => bucket is on scaled values, interpret relative only)
    stats = []
    for lo, hi in buckets:
        filtered = [tr for tr in buffer if lo <= tr.s[pd_idx] < hi]
        if not filtered:
            continue
        acts = [policy_fn(tr.s) for tr in filtered]
        rews = [tr.r for tr in filtered]
        stats.append(dict(
            bucket=f"[{lo},{hi})",
            n=len(filtered),
            mean_action=float(np.mean(acts)),
            mean_reward=float(np.mean(rews))
        ))
    with open(os.path.join(CONFIG["RESULT_DIR"], f"pd_bucket_{label}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats

pd_bucket_report(test_buf, cql_policy, "cql")
pd_bucket_report(test_buf, awac_policy, "awac")

# ==============================
# 9. Save Artifacts
# ==============================

torch.save({
    "dt_state_dict": dt_model.state_dict(),
    "twin_q_state_dict": twin_q.state_dict(),
    "actor_state_dict": actor.state_dict(),
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "state_features": STATE_FEATURES,
    "config": CONFIG,
    "eval_results": eval_results
}, os.path.join(CONFIG["RESULT_DIR"], "model_bundle.pt"))

pd.DataFrame(eval_results).to_csv(os.path.join(CONFIG["RESULT_DIR"], "evaluation_summary.csv"), index=False)

logging.info("Pipeline complete. Artifacts saved.")
logging.info("Evaluation Summary:")
for r in eval_results:
    logging.info(str(r))

print("\n=== Done ===")

