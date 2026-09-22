import os
import copy
import math
import time
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


@dataclass
class Config:
    seed: int = 42

    data_path: str = "diabetes_binary_health_indicators_BRFSS2015.csv"
    label_col: str = "Diabetes_binary"

    sample_size: int = 30000
    test_size: float = 0.2

    num_clients: int = 5
    noniid_alpha: float = 1
    communication_rounds: int = 50
    local_epochs: int = 2

    knn_k: int = 10
    learned_k: int = 10

    hidden_dim: int = 64
    dropout: float = 0.3


    lr: float = 1e-3
    weight_decay: float = 1e-4

    eta_adaptive_graph: float = 0.8
    lambda_graph: float = 0.001
    lambda_prox: float = 0.001
    lambda_adv: float = 0.0005

    eval_threshold: float = 0.35
    max_class_weight: float = 8.0

    results_dir: str = "results_diabetes"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_tensor(x, dtype=torch.float32, device=None):
    if device is None:
        device = cfg.device
    return torch.tensor(x, dtype=dtype, device=device)


def label_entropy(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def prediction_uncertainty(logits: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=1)
    ent = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
    return float(ent.mean().detach().cpu().item())


def get_class_weights(y: torch.Tensor, num_classes: int = 2, max_weight: float = 8.0):
    counts = torch.bincount(y, minlength=num_classes).float()
    counts = counts + 1.0
    total = counts.sum()
    weights = total / (num_classes * counts)
    weights = torch.clamp(weights, max=max_weight)
    return weights.to(y.device)


def find_label_column(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred

    candidates = [
        "Diabetes_binary",
        "Diabetes_012",
        "diabetes",
        "target",
        "label",
        "Label",
        "Outcome",
        "class",
        "Class"
    ]

    for c in candidates:
        if c in df.columns:
            print(f"[Info] label_col '{preferred}' not found. Use detected label column: {c}")
            return c

    raise ValueError(
        f"Cannot find label column '{preferred}'. Available columns are: {list(df.columns)}"
    )


def load_diabetes_dataset(cfg: Config):
    if os.path.exists(cfg.data_path):
        print(f"[Info] Found local dataset: {cfg.data_path}")

        df = pd.read_csv(cfg.data_path)
        label_col = find_label_column(df, cfg.label_col)

        df = df.dropna().reset_index(drop=True)

        y = df[label_col].values

        if label_col == "Diabetes_012":
            y = (y > 0).astype(int)
        else:
            y = y.astype(int)

        X_df = df.drop(columns=[label_col])

    else:
        print("[Info] Local CSV not found.")
        print("[Info] Downloading CDC Diabetes Health Indicators Dataset from UCI...")

        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError:
            raise ImportError(
                "Package ucimlrepo is not installed.\n"
                "Please run this in terminal:\n"
                "pip install ucimlrepo"
            )

        diabetes = fetch_ucirepo(id=891)

        X_df = diabetes.data.features
        y_df = diabetes.data.targets

        print("[Info] UCI dataset downloaded successfully.")
        print("[Info] Feature columns:", list(X_df.columns))
        print("[Info] Target columns:", list(y_df.columns))

        if isinstance(y_df, pd.DataFrame):
            target_col = y_df.columns[0]
            y = y_df[target_col].values
        else:
            y = np.asarray(y_df).reshape(-1)

        print("[Info] Original label distribution:", dict(zip(*np.unique(y, return_counts=True))))

        if len(np.unique(y)) > 2:
            y = (y > 0).astype(int)
            print("[Info] Converted Diabetes_012 into binary label.")
        else:
            y = y.astype(int)

        ensure_dir("data")
        save_df = X_df.copy()
        save_df["Diabetes_binary"] = y
        saved_path = os.path.join("data", "cdc_diabetes_from_uci.csv")
        save_df.to_csv(saved_path, index=False)
        print(f"[Info] Saved downloaded dataset to: {saved_path}")

    X_df = pd.get_dummies(X_df, drop_first=True)
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    X = X_df.values.astype(np.float32)

    if cfg.sample_size is not None and cfg.sample_size > 0 and cfg.sample_size < len(y):
        idx_all = np.arange(len(y))
        idx_sample, _ = train_test_split(
            idx_all,
            train_size=cfg.sample_size,
            random_state=cfg.seed,
            stratify=y
        )

        X = X[idx_sample]
        y = y[idx_sample]

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=y
    )

    print("\n========== Dataset ==========")
    print(f"Total used samples: {len(y)}")
    print(f"Train samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Feature dimension: {X_train.shape[1]}")
    print("Binary label distribution:", dict(zip(*np.unique(y, return_counts=True))))
    print("Device:", cfg.device)

    return X_train, X_test, y_train, y_test




def dirichlet_noniid_split(X, y, num_clients, alpha=0.3, seed=42, min_size=100):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    attempt = 0

    while True:
        attempt += 1
        client_indices = [[] for _ in range(num_clients)]

        for c in classes:
            class_idx = np.where(y == c)[0]
            rng.shuffle(class_idx)

            proportions = rng.dirichlet(alpha * np.ones(num_clients))
            split_points = (np.cumsum(proportions)[:-1] * len(class_idx)).astype(int)
            splits = np.split(class_idx, split_points)

            for i in range(num_clients):
                client_indices[i].extend(splits[i].tolist())

        sizes = [len(idx) for idx in client_indices]

        if min(sizes) >= min_size:
            break

        if attempt > 100:
            print("[Warning] Could not satisfy min_size. Use current split.")
            break

    clients = []

    for i, idx in enumerate(client_indices):
        idx = np.array(idx, dtype=int)
        rng.shuffle(idx)

        clients.append({
            "client_id": i,
            "X": X[idx],
            "y": y[idx],
            "indices": idx
        })

    return clients


def scipy_sparse_to_torch_sparse(mat: sp.spmatrix, device: str):
    mat = mat.tocoo().astype(np.float32)

    indices = torch.from_numpy(
        np.vstack([mat.row, mat.col]).astype(np.int64)
    )

    values = torch.from_numpy(mat.data.astype(np.float32))
    shape = torch.Size(mat.shape)

    return torch.sparse_coo_tensor(
        indices,
        values,
        shape,
        device=device
    ).coalesce()


def build_knn_graph_sparse(X: np.ndarray, k: int, device: str):
    n = X.shape[0]
    real_k = min(k + 1, n)

    nbrs = NearestNeighbors(
        n_neighbors=real_k,
        metric="euclidean",
        algorithm="auto"
    )

    nbrs.fit(X)
    _, indices = nbrs.kneighbors(X)

    rows = []
    cols = []

    for i in range(n):
        for j in indices[i]:
            if i != j:
                rows.append(i)
                cols.append(j)

                rows.append(j)
                cols.append(i)

    data = np.ones(len(rows), dtype=np.float32)

    A_raw = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(n, n)
    ).tocsr()

    A_raw.data[:] = 1.0
    A_raw.eliminate_zeros()

    num_edges = A_raw.nnz / 2.0
    avg_degree = A_raw.sum(axis=1).A1.mean()
    density = num_edges / (n * (n - 1) / 2.0 + 1e-12)

    A = A_raw + sp.eye(n, dtype=np.float32, format="csr")
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg + 1e-12, -0.5)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)

    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    A_norm = A_norm.tocoo()

    A_norm_torch = scipy_sparse_to_torch_sparse(A_norm, device=device)

    raw_coo = A_raw.tocoo()

    edge_index = torch.tensor(
        np.vstack([raw_coo.row, raw_coo.col]),
        dtype=torch.long,
        device=device
    )

    stats = {
        "nodes": int(n),
        "edges": float(num_edges),
        "avg_degree": float(avg_degree),
        "density": float(density)
    }

    return A_norm_torch, edge_index, stats


def prepare_clients(raw_clients, cfg: Config):
    clients = []

    for c in raw_clients:
        X_np = c["X"]
        y_np = c["y"]

        A_norm, edge_index, stats = build_knn_graph_sparse(
            X_np,
            cfg.knn_k,
            cfg.device
        )

        clients.append({
            "client_id": c["client_id"],
            "X": to_tensor(X_np, torch.float32, cfg.device),
            "y": torch.tensor(y_np, dtype=torch.long, device=cfg.device),
            "A_norm": A_norm,
            "edge_index": edge_index,
            "num_samples": int(len(y_np)),
            "label_entropy": label_entropy(y_np),
            "stats": stats
        })

    return clients


def save_client_statistics(clients, cfg: Config):
    rows = []

    for c in clients:
        y_np = c["y"].detach().cpu().numpy()
        class_values, counts = np.unique(y_np, return_counts=True)
        count_dict = {int(k): int(v) for k, v in zip(class_values, counts)}

        rows.append({
            "client": c["client_id"],
            "samples": c["num_samples"],
            "class_0": count_dict.get(0, 0),
            "class_1": count_dict.get(1, 0),
            "label_entropy": c["label_entropy"],
            "nodes": c["stats"]["nodes"],
            "edges": c["stats"]["edges"],
            "avg_degree": c["stats"]["avg_degree"],
            "density": c["stats"]["density"]
        })

    df = pd.DataFrame(rows)

    path = os.path.join(cfg.results_dir, "client_statistics.csv")
    df.to_csv(path, index=False)

    print("\n========== Client Statistics ==========")
    print(df)
    print(f"Saved: {path}")

    return df


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradientReversalFunction.apply(x, lambd)


class SparseFedGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_clients: int,
        dropout: float = 0.3,
        adaptive_graph: bool = False,
        adversarial: bool = False,
        eta: float = 0.8,
        learned_k: int = 10
    ):
        super().__init__()

        self.adaptive_graph = adaptive_graph
        self.adversarial = adversarial
        self.eta = eta
        self.learned_k = learned_k
        self.dropout = dropout

        self.gcn1 = nn.Linear(input_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        if adversarial:
            self.domain_discriminator = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_clients)
            )

    def learned_topk_message(self, H: torch.Tensor):
        n = H.shape[0]
        k = min(self.learned_k + 1, n)

        H_norm = F.normalize(H, p=2, dim=1)
        sim = H_norm @ H_norm.t() / math.sqrt(H.shape[1])

        vals, idx = torch.topk(sim, k=k, dim=1)
        attn = F.softmax(vals, dim=1)

        neigh = H[idx]
        msg = torch.sum(neigh * attn.unsqueeze(-1), dim=1)

        return msg

    def forward(self, X, A_norm, grl_lambda: float = 1.0):
        H = torch.sparse.mm(A_norm, X)
        H = self.gcn1(H)
        H = F.relu(H)
        H = F.dropout(H, p=self.dropout, training=self.training)

        H_knn = torch.sparse.mm(A_norm, H)

        if self.adaptive_graph:
            H_learn = self.learned_topk_message(H)
            H_msg = self.eta * H_knn + (1.0 - self.eta) * H_learn
        else:
            H_msg = H_knn

        Z = self.gcn2(H_msg)
        Z = F.relu(Z)
        Z = F.dropout(Z, p=self.dropout, training=self.training)

        logits = self.classifier(Z)

        domain_logits = None

        if self.adversarial:
            Z_rev = grad_reverse(Z, grl_lambda)
            domain_logits = self.domain_discriminator(Z_rev)

        return logits, Z, domain_logits



def graph_smoothness_loss(Z: torch.Tensor, edge_index: torch.Tensor):
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=Z.device)

    src = edge_index[0]
    dst = edge_index[1]

    diff = Z[src] - Z[dst]
    loss = torch.mean(torch.sum(diff * diff, dim=1))

    return loss


def proximal_loss(local_model: nn.Module, global_state: dict):
    loss = torch.tensor(
        0.0,
        device=next(local_model.parameters()).device
    )

    for name, param in local_model.named_parameters():
        if name in global_state:
            loss = loss + torch.sum((param - global_state[name]) ** 2)

    return loss


def evaluate_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor, A_norm: torch.Tensor, cfg: Config):
    model.eval()

    with torch.no_grad():
        logits, _, _ = model(X, A_norm)
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = (probs >= cfg.eval_threshold).astype(int)
        y_true = y.detach().cpu().numpy()

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)

    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = np.nan

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc)
    }



def aggregate_states(local_states, weights):
    new_state = copy.deepcopy(local_states[0])

    for key in new_state.keys():
        new_state[key] = torch.zeros_like(new_state[key])

        for state, w in zip(local_states, weights):
            new_state[key] += state[key] * float(w)

    return new_state


def normalize_summary(E):
    E = np.array(E, dtype=np.float32)
    mean = E.mean(axis=0, keepdims=True)
    std = E.std(axis=0, keepdims=True) + 1e-8
    return (E - mean) / std


def self_attention_client_weights(client_summaries):

    E = normalize_summary(client_summaries)
    E_t = torch.tensor(E, dtype=torch.float32)

    d = E_t.shape[1]

    scores = E_t @ E_t.t() / math.sqrt(d)
    attention_matrix = F.softmax(scores, dim=1)
    context = attention_matrix @ E_t

    # Higher data size, higher loss improvement, useful graph structure;
    # lower local loss and lower uncertainty.
    score_vec = torch.tensor(
        [0.20, -0.30, 0.35, 0.15, 0.10, 0.20, -0.20],
        dtype=torch.float32
    )

    raw_scores = context @ score_vec
    alpha = F.softmax(raw_scores, dim=0)

    return alpha.detach().cpu().numpy(), attention_matrix.detach().cpu().numpy()


def train_local_model(
    global_model,
    client,
    cfg: Config,
    use_prox=False,
    use_graph_loss=False,
    use_adv=False
):
    local_model = copy.deepcopy(global_model)
    local_model.train()

    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    X = client["X"]
    y = client["y"]
    A_norm = client["A_norm"]
    edge_index = client["edge_index"]
    client_id = client["client_id"]

    global_state = copy.deepcopy(global_model.state_dict())
    class_weights = get_class_weights(y, num_classes=2, max_weight=cfg.max_class_weight)

    with torch.no_grad():
        logits_before, _, _ = local_model(X, A_norm)
        loss_before = F.cross_entropy(logits_before, y, weight=class_weights).item()

    for _ in range(cfg.local_epochs):
        optimizer.zero_grad()

        logits, Z, domain_logits = local_model(X, A_norm, grl_lambda=1.0)

        loss_cls = F.cross_entropy(logits, y, weight=class_weights)
        total_loss = loss_cls

        if use_graph_loss:
            loss_g = graph_smoothness_loss(Z, edge_index)
            total_loss = total_loss + cfg.lambda_graph * loss_g

        if use_prox:
            loss_p = proximal_loss(local_model, global_state)
            total_loss = total_loss + cfg.lambda_prox * loss_p

        if use_adv and domain_logits is not None:
            domain_labels = torch.full(
                size=(len(y),),
                fill_value=client_id,
                dtype=torch.long,
                device=cfg.device
            )

            loss_adv = F.cross_entropy(domain_logits, domain_labels)
            total_loss = total_loss + cfg.lambda_adv * loss_adv

        total_loss.backward()
        optimizer.step()

    local_model.eval()

    with torch.no_grad():
        logits_after, _, _ = local_model(X, A_norm)
        loss_after = F.cross_entropy(logits_after, y, weight=class_weights).item()
        uncertainty = prediction_uncertainty(logits_after)

    delta_loss = loss_before - loss_after

    summary = [
        client["num_samples"],
        loss_after,
        delta_loss,
        client["stats"]["density"],
        client["stats"]["avg_degree"],
        client["label_entropy"],
        uncertainty
    ]

    return {
        "state": copy.deepcopy(local_model.state_dict()),
        "loss_before": float(loss_before),
        "loss_after": float(loss_after),
        "delta_loss": float(delta_loss),
        "summary": summary
    }



def run_federated_experiment(
    name,
    clients,
    X_test,
    y_test,
    cfg: Config,
    use_attention=False,
    use_prox=False,
    adaptive_graph=False,
    use_graph_loss=False,
    adversarial=False
):
    print(f"\n========== Running {name} ==========")

    start_time = time.time()
    input_dim = clients[0]["X"].shape[1]

    global_model = SparseFedGNN(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        num_classes=2,
        num_clients=cfg.num_clients,
        dropout=cfg.dropout,
        adaptive_graph=adaptive_graph,
        adversarial=adversarial,
        eta=cfg.eta_adaptive_graph,
        learned_k=cfg.learned_k
    ).to(cfg.device)

    A_test, _, _ = build_knn_graph_sparse(X_test, cfg.knn_k, cfg.device)

    X_test_t = to_tensor(X_test, torch.float32, cfg.device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=cfg.device)

    metrics_rows = []
    attention_rows = []

    for r in range(1, cfg.communication_rounds + 1):
        local_states = []
        client_summaries = []
        client_losses = []

        for client in clients:
            result = train_local_model(
                global_model=global_model,
                client=client,
                cfg=cfg,
                use_prox=use_prox,
                use_graph_loss=use_graph_loss,
                use_adv=adversarial
            )

            local_states.append(result["state"])
            client_summaries.append(result["summary"])
            client_losses.append(result["loss_after"])

        if use_attention:
            weights, _ = self_attention_client_weights(client_summaries)
        else:
            sizes = np.array(
                [c["num_samples"] for c in clients],
                dtype=np.float32
            )
            weights = sizes / sizes.sum()

        new_state = aggregate_states(local_states, weights)
        global_model.load_state_dict(new_state)

        metrics = evaluate_model(global_model, X_test_t, y_test_t, A_test, cfg)

        row = {
            "round": r,
            "method": name,
            "mean_client_loss": float(np.mean(client_losses)),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "auc": metrics["auc"]
        }

        for i, w in enumerate(weights):
            row[f"weight_client_{i}"] = float(w)

        metrics_rows.append(row)

        attn_row = {
            "round": r,
            "method": name
        }

        for i, w in enumerate(weights):
            attn_row[f"client_{i}"] = float(w)

        attention_rows.append(attn_row)

        if r == 1 or r % 5 == 0 or r == cfg.communication_rounds:
            print(
                f"[{name}] Round {r:03d} | "
                f"Loss={row['mean_client_loss']:.4f} | "
                f"Acc={row['accuracy']:.4f} | "
                f"Precision={row['precision']:.4f} | "
                f"Recall={row['recall']:.4f} | "
                f"F1={row['f1']:.4f} | "
                f"AUC={row['auc']:.4f} | "
                f"Weights={np.round(weights, 3)}"
            )

    metrics_df = pd.DataFrame(metrics_rows)
    attention_df = pd.DataFrame(attention_rows)

    metrics_path = os.path.join(cfg.results_dir, f"{name}_metrics.csv")
    attention_path = os.path.join(cfg.results_dir, f"{name}_attention_weights.csv")

    metrics_df.to_csv(metrics_path, index=False)
    attention_df.to_csv(attention_path, index=False)

    final_metrics = metrics_rows[-1].copy()
    final_metrics["method"] = name
    final_metrics["time_seconds"] = time.time() - start_time

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved weights: {attention_path}")

    return final_metrics



def plot_metric_curve(metric, cfg: Config):
    fig_dir = os.path.join(cfg.results_dir, "figures")
    ensure_dir(fig_dir)

    files = [
        "FedAvg-GNN_metrics.csv",
        "FedProx-GNN_metrics.csv",
        "SA-FedGNN_metrics.csv",
        "SA-Ada-FedGNN_metrics.csv",
        "SA-AdaAdv-FedGNN_metrics.csv"
    ]

    plt.figure()
    plotted = False

    for file in files:
        path = os.path.join(cfg.results_dir, file)

        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        label = file.replace("_metrics.csv", "")

        plt.plot(df["round"], df[metric], label=label)
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("Communication Round")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} Curve")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(fig_dir, f"{metric}_curve.png")
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"Saved figure: {out}")


def plot_attention_heatmap(method_name, cfg: Config):
    path = os.path.join(cfg.results_dir, f"{method_name}_attention_weights.csv")

    if not os.path.exists(path):
        return

    fig_dir = os.path.join(cfg.results_dir, "figures")
    ensure_dir(fig_dir)

    df = pd.read_csv(path)
    client_cols = [c for c in df.columns if c.startswith("client_")]

    if len(client_cols) == 0:
        return

    data = df[client_cols].values.T

    plt.figure()
    plt.imshow(data, aspect="auto")
    plt.colorbar(label="Aggregation Weight")
    plt.xlabel("Communication Round")
    plt.ylabel("Client")
    plt.yticks(range(len(client_cols)), client_cols)
    plt.title(f"{method_name} Aggregation Weights")
    plt.tight_layout()

    out = os.path.join(fig_dir, f"{method_name}_weights_heatmap.png")
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"Saved figure: {out}")


def plot_final_bar(metric, cfg: Config):
    path = os.path.join(cfg.results_dir, "final_comparison.csv")

    if not os.path.exists(path):
        return

    fig_dir = os.path.join(cfg.results_dir, "figures")
    ensure_dir(fig_dir)

    df = pd.read_csv(path)

    if metric not in df.columns:
        return

    plt.figure()
    plt.bar(df["method"], df[metric])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(metric.upper())
    plt.title(f"Final {metric.upper()} Comparison")
    plt.tight_layout()

    out = os.path.join(fig_dir, f"final_{metric}_comparison.png")
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"Saved figure: {out}")


def plot_client_distribution(cfg: Config):
    path = os.path.join(cfg.results_dir, "client_statistics.csv")

    if not os.path.exists(path):
        return

    fig_dir = os.path.join(cfg.results_dir, "figures")
    ensure_dir(fig_dir)

    df = pd.read_csv(path)

    plt.figure()
    plt.bar(df["client"], df["class_0"], label="Class 0")
    plt.bar(df["client"], df["class_1"], bottom=df["class_0"], label="Class 1")
    plt.xlabel("Client")
    plt.ylabel("Number of Samples")
    plt.title("Non-IID Client Label Distribution")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(fig_dir, "client_label_distribution.png")
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"Saved figure: {out}")


def make_plots(cfg: Config):
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        plot_metric_curve(metric, cfg)

    for method_name in ["SA-FedGNN", "SA-Ada-FedGNN", "SA-AdaAdv-FedGNN"]:
        plot_attention_heatmap(method_name, cfg)

    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        plot_final_bar(metric, cfg)

    plot_client_distribution(cfg)


def main():
    set_seed(cfg.seed)
    ensure_dir(cfg.results_dir)

    print("\n========== Configuration ==========")
    for k, v in cfg.__dict__.items():
        print(f"{k}: {v}")

    X_train, X_test, y_train, y_test = load_diabetes_dataset(cfg)

    raw_clients = dirichlet_noniid_split(
        X_train,
        y_train,
        num_clients=cfg.num_clients,
        alpha=cfg.noniid_alpha,
        seed=cfg.seed,
        min_size=max(50, cfg.sample_size // cfg.num_clients // 20)
    )

    clients = prepare_clients(raw_clients, cfg)
    save_client_statistics(clients, cfg)

    final_results = []

    final_results.append(
        run_federated_experiment(
            name="FedAvg-GNN",
            clients=clients,
            X_test=X_test,
            y_test=y_test,
            cfg=cfg,
            use_attention=False,
            use_prox=False,
            adaptive_graph=False,
            use_graph_loss=False,
            adversarial=False
        )
    )


    final_results.append(
        run_federated_experiment(
            name="FedProx-GNN",
            clients=clients,
            X_test=X_test,
            y_test=y_test,
            cfg=cfg,
            use_attention=False,
            use_prox=True,
            adaptive_graph=False,
            use_graph_loss=False,
            adversarial=False
        )
    )


    final_results.append(
        run_federated_experiment(
            name="SA-FedGNN",
            clients=clients,
            X_test=X_test,
            y_test=y_test,
            cfg=cfg,
            use_attention=True,
            use_prox=False,
            adaptive_graph=False,
            use_graph_loss=False,
            adversarial=False
        )
    )


    final_results.append(
        run_federated_experiment(
            name="SA-Ada-FedGNN",
            clients=clients,
            X_test=X_test,
            y_test=y_test,
            cfg=cfg,
            use_attention=True,
            use_prox=True,
            adaptive_graph=True,
            use_graph_loss=True,
            adversarial=False
        )
    )


    final_results.append(
        run_federated_experiment(
            name="SA-AdaAdv-FedGNN",
            clients=clients,
            X_test=X_test,
            y_test=y_test,
            cfg=cfg,
            use_attention=True,
            use_prox=True,
            adaptive_graph=True,
            use_graph_loss=True,
            adversarial=True
        )
    )

    final_df = pd.DataFrame(final_results)

    final_path = os.path.join(cfg.results_dir, "final_comparison.csv")
    final_df.to_csv(final_path, index=False)

    print("\n========== Final Comparison ==========")
    print(final_df)
    print(f"\nSaved final comparison: {final_path}")

    make_plots(cfg)

    print("\nDone.")
    print(f"All results are saved in: {cfg.results_dir}")


if __name__ == "__main__":
    main()