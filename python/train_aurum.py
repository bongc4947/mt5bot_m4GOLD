"""
train_aurum.py — unified CLI for the AURUM v2 AI stack (production grade).

Subcommands map 1:1 to the phased rollout in docs/DESIGN_AURUM.md:

    python python/train_aurum.py baseline    # Phase 1 — purged-CV controls
    python python/train_aurum.py pretrain    # Phase 2 — L0 SSL encoders
    python python/train_aurum.py finetune    # Phase 2-3 — backbone + heads
    python python/train_aurum.py meta        # Phase 4 — meta-label gate
    python python/train_aurum.py conformal   # Phase 4 — conformal calibration
    python python/train_aurum.py export      # ONNX bundle + deploy gate
    python python/train_aurum.py all         # full pipeline, gated

Production hardening (vs the prototype):
  * Leak-free 3-way chronological split — train / calib / test, with
    label-horizon gaps. The deploy decision is made on a `test` window
    that NOTHING was fit or selected on.
  * The deploy gate measures AURUM's TRUE deployed performance — profit
    factor AFTER the conformal singleton filter and the meta-label gate,
    i.e. the exact pipeline the EA runs — not the raw direction head.
  * Apples-to-apples: the XGBoost baseline is re-scored on AURUM's own
    `test` window, so the gate compares like with like.
  * Deterministic seeding, single dataset build per run, exec head
    trained on real MFE/MAE/timing targets, rich spec + report JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from aurum.aurum_config import (
    SEED, VAL_SPLIT, FINETUNE_EPOCHS, FINETUNE_LR, WEIGHT_DECAY, PATIENCE,
    FREEZE_ENCODER_EPOCHS, LOSS_W_DIRECTION, LOSS_W_QUANTILE, LOSS_W_EXEC,
    LOSS_W_REGIME, QUANTILES, CV_N_SPLITS, CV_EMBARGO_PCT, LABEL_HORIZON_BARS,
    CONFORMAL_CAL_FRAC, GATE_MIN_PF, GATE_MIN_EXCESS_VS_BASELINE,
    SLICE_DIR, META_ACT_THRESHOLD, BASELINE_XGB_ESTIMATORS,
)

log = logging.getLogger(__name__)
_ARTIFACT_DIR = Path(__file__).parent.parent / "onnx_out"
_GATE_MIN_TRADES = 100      # min filtered trades for a trustworthy deploy PF
_PF_CAP = 10.0              # cap pathological inf PF when losses ~ 0


# ===========================================================================
# Determinism
# ===========================================================================
def _seed_all(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ===========================================================================
# Splits — leak-free 3-way chronological partition
# ===========================================================================
def _three_way_split(n: int) -> tuple[slice, slice, slice]:
    """
    [ train ][gap][ calib ][gap][ test ]   — chronological, no leakage.

    test  = last VAL_SPLIT of the series (deploy decision is made here).
    calib = CONFORMAL_CAL_FRAC before test (conformal calibration +
            fine-tune early-stop selection).
    train = everything before that.
    `gap` = LABEL_HORIZON_BARS, so a label's forward window cannot span
    a segment boundary.
    """
    gap = LABEL_HORIZON_BARS
    n_test = int(n * VAL_SPLIT)
    n_cal = int(n * CONFORMAL_CAL_FRAC)
    test = slice(n - n_test, n)
    cal = slice(n - n_test - gap - n_cal, n - n_test - gap)
    train = slice(0, max(1, cal.start - gap))
    return train, cal, test


# ===========================================================================
# Metrics
# ===========================================================================
def _profit_factor(pnl: np.ndarray, min_trades: int = 1) -> float:
    """Profit factor with production guards — capped, min-trade floor."""
    if len(pnl) < min_trades:
        return 0.0
    gains = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    if losses <= 1e-12:
        return _PF_CAP if gains > 0 else 0.0
    return min(_PF_CAP, gains / losses)


def _direction_pf(pred_dir: np.ndarray, y_ret: np.ndarray,
                  min_trades: int = 1) -> float:
    """Profit factor of acting on a 3-class direction prediction."""
    sign = np.where(pred_dir == 2, 1.0, np.where(pred_dir == 0, -1.0, 0.0))
    pnl = (sign * y_ret)[sign != 0]
    return _profit_factor(pnl, min_trades)


def _accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float((pred == true).mean()) if len(pred) else 0.0


# ===========================================================================
# Deployed-pipeline evaluation — AURUM's TRUE number
# ===========================================================================
def _predict_primary(net, X: np.ndarray, device: str, batch: int) -> np.ndarray:
    """Run the export wrapper -> [N, OUTPUT_DIM] with dir/regime softmaxed."""
    import torch
    from aurum.model import AurumExportWrapper
    wrapper = AurumExportWrapper(net).eval()
    Xt = torch.from_numpy(X).float()
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xt), batch):
            outs.append(wrapper(Xt[i:i + batch].to(device)).cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 13), np.float32)


def evaluate_deployed(primary: np.ndarray, y_dir: np.ndarray,
                      y_ret: np.ndarray, *, q_hat: float,
                      meta_gate=None, meta_feats: np.ndarray | None = None
                      ) -> dict:
    """
    AURUM's TRUE deployed performance — profit factor over exactly the
    trades the EA would take: direction head -> conformal singleton
    filter -> meta-label gate. This is what the deploy gate must judge.
    """
    from aurum.conformal import singleton_mask

    probs = primary[:, SLICE_DIR[0]:SLICE_DIR[1]]
    pred = probs.argmax(axis=1)

    # L5 — conformal: keep only confident (singleton-set) bars.
    singleton = singleton_mask(probs, q_hat)
    # L4 — meta gate: keep only bars the gate says "act".
    if meta_gate is not None and meta_feats is not None:
        pact = meta_gate.predict_act_prob(meta_feats)
        act = pact >= META_ACT_THRESHOLD
    else:
        act = np.ones(len(pred), dtype=bool)
    tradeable = singleton & act & (pred != 1)      # 1 == flat -> never traded

    sign = np.where(pred == 2, 1.0, np.where(pred == 0, -1.0, 0.0))
    pnl = sign * y_ret
    pnl_kept = pnl[tradeable]

    n_kept = int(tradeable.sum())
    return {
        "filtered_pf": _profit_factor(pnl_kept, _GATE_MIN_TRADES),
        "raw_pf": _direction_pf(pred, y_ret),
        "n_trades": n_kept,
        "trade_rate": float(tradeable.mean()) if len(pred) else 0.0,
        "win_rate": float((pnl_kept > 0).mean()) if n_kept else 0.0,
        "direction_acc": _accuracy(pred, y_dir),
        "singleton_frac": float(singleton.mean()) if len(pred) else 0.0,
    }


# ===========================================================================
# Phase 1 — baselines under purged CV (informational control)
# ===========================================================================
def cmd_baseline(args, ds=None) -> int:
    from aurum.datamodule import build_dataset, summary_features
    from aurum.aurum_config import BASELINE_MAX_SAMPLES
    from cv.purged_kfold import PurgedKFold
    from baselines.xgb_direction import create_xgb_baseline

    _seed_all()
    ds = ds or build_dataset(labelled=True, max_bars=args.max_bars)
    X, y, y_ret = ds["X_flat"], ds["y_dir"], ds["y_ret"]
    X_xgb = summary_features(ds["X"])
    if len(y) > BASELINE_MAX_SAMPLES:
        log.info("[baseline] capping %d -> %d most-recent samples",
                 len(y), BASELINE_MAX_SAMPLES)
        X, X_xgb = X[-BASELINE_MAX_SAMPLES:], X_xgb[-BASELINE_MAX_SAMPLES:]
        y, y_ret = y[-BASELINE_MAX_SAMPLES:], y_ret[-BASELINE_MAX_SAMPLES:]
    pk = PurgedKFold(n_splits=CV_N_SPLITS, horizon=LABEL_HORIZON_BARS,
                     embargo_pct=CV_EMBARGO_PCT)
    log.info("[baseline] %d samples  xgb_features=%d  dlinear_features=%d",
             len(y), X_xgb.shape[1], X.shape[1])

    report = {"dlinear": [], "xgboost": []}
    for fold, (tr, te) in enumerate(pk.split(len(y))):
        xgb = create_xgb_baseline(use_gpu=args.use_gpu,
                                  n_estimators=BASELINE_XGB_ESTIMATORS)
        xgb.fit(X_xgb[tr], y[tr])
        xpf = _direction_pf(xgb.predict(X_xgb[te]), y_ret[te])
        report["xgboost"].append(xpf)
        dpf = _train_dlinear_fold(X[tr], y[tr], X[te], y_ret[te], args)
        report["dlinear"].append(dpf)
        log.info("[baseline] fold %d/%d  xgb_PF=%.3f  dlinear_PF=%.3f",
                 fold + 1, pk.get_n_splits(), xpf, dpf)

    summary = {k: {"mean_pf": float(np.mean(v)),
                   "folds": [round(x, 3) for x in v]}
               for k, v in report.items()}
    summary["best_baseline_pf"] = max(summary["dlinear"]["mean_pf"],
                                      summary["xgboost"]["mean_pf"])
    out = _ARTIFACT_DIR / "AURUM_baseline_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    log.info("[baseline] purged-CV control — DLinear meanPF=%.3f  "
             "XGBoost meanPF=%.3f", summary["dlinear"]["mean_pf"],
             summary["xgboost"]["mean_pf"])
    return 0


def _train_dlinear_fold(Xtr, ytr, Xte, yret_te, args) -> float:
    import torch
    from baselines.dlinear import create_dlinear
    dev = _device()
    _seed_all()
    model = create_dlinear().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    counts = np.bincount(ytr, minlength=3).astype(np.float64)
    cw = counts.sum() / np.maximum(counts, 1.0)
    cw = cw / cw.mean()
    lossf = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32, device=dev))
    Xt, yt = torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).long()
    bs = 4096
    for _ in range(args.dlinear_epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            loss = lossf(model(Xt[idx].to(dev)), yt[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte).float().to(dev)).argmax(1).cpu().numpy()
    return _direction_pf(pred, yret_te)


def _xgb_test_window_pf(ds: dict, train_idx, test_idx, use_gpu: bool) -> float:
    """
    Apples-to-apples baseline: XGBoost trained on AURUM's own train+calib
    rows, scored on AURUM's exact test window — directly comparable to
    AURUM's deployed PF, same data, same split.
    """
    from aurum.datamodule import summary_features
    from baselines.xgb_direction import create_xgb_baseline
    X = summary_features(ds["X"])
    xgb = create_xgb_baseline(use_gpu=use_gpu,
                              n_estimators=BASELINE_XGB_ESTIMATORS)
    xgb.fit(X[train_idx], ds["y_dir"][train_idx])
    pred = xgb.predict(X[test_idx])
    return _direction_pf(pred, ds["y_ret"][test_idx], _GATE_MIN_TRADES)


# ===========================================================================
# Phase 2 — SSL pretraining
# ===========================================================================
def cmd_pretrain(args, ds=None) -> int:
    from aurum.datamodule import build_dataset
    from aurum.pretrain import pretrain_all
    from aurum.aurum_config import SSL_EPOCHS
    _seed_all()
    ds = ds or build_dataset(labelled=False, max_bars=args.max_bars)
    epochs = args.epochs or SSL_EPOCHS
    paths = pretrain_all(ds, device=_device(), epochs=epochs)
    log.info("[pretrain] encoders: %s", {k: v.name for k, v in paths.items()})
    return 0


# ===========================================================================
# Phase 2-3 — fine-tune backbone + multi-task heads
# ===========================================================================
def cmd_finetune(args, ds=None) -> int:
    import torch
    from aurum.datamodule import build_dataset
    from aurum.model import create_aurum_net
    from aurum.heads import PinballLoss

    dev = _device()
    _seed_all()
    ds = ds or build_dataset(labelled=True, max_bars=args.max_bars)
    X = ds["X_flat"]
    n = len(X)
    tr, cal, te = _three_way_split(n)
    log.info("[finetune] split  train=%d  calib=%d  test=%d  (n=%d)",
             tr.stop - tr.start, cal.stop - cal.start, te.stop - te.start, n)

    net = create_aurum_net().to(dev)
    net.load_norm(ds["norm"])
    ssl_paths = {tf: _ARTIFACT_DIR / f"aurum_ssl_encoder_{tf}.pt"
                 for tf in net.tf_order}
    if all(p.exists() for p in ssl_paths.values()):
        try:
            net.load_encoders(ssl_paths)
            log.info("[finetune] loaded SSL-pretrained encoders")
        except Exception as e:  # noqa: BLE001 — dim mismatch etc.
            log.warning("[finetune] SSL encoders incompatible (%s) — "
                        "training encoders from scratch", e)
    else:
        log.warning("[finetune] no SSL encoders found — training from scratch. "
                    "Run `train_aurum.py pretrain` first for best results.")

    opt = torch.optim.AdamW(net.parameters(), lr=FINETUNE_LR,
                            weight_decay=WEIGHT_DECAY)
    counts = np.bincount(ds["y_dir"][tr], minlength=3).astype(np.float64)
    cls_w = counts.sum() / np.maximum(counts, 1.0)
    cls_w = cls_w / cls_w.mean()
    log.info("[finetune] direction class counts=%s  weights=%s",
             counts.astype(int).tolist(), np.round(cls_w, 3).tolist())
    ce_dir = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(cls_w, dtype=torch.float32, device=dev))
    ce = torch.nn.CrossEntropyLoss()
    pinball = PinballLoss(QUANTILES).to(dev)
    mse = torch.nn.MSELoss()

    def _batches(sl, bs, shuffle):
        idx = np.arange(sl.start, sl.stop)
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, len(idx), bs):
            yield idx[i:i + bs]

    Xt = torch.from_numpy(X).float()
    yd = torch.from_numpy(ds["y_dir"]).long()
    yr = torch.from_numpy(ds["y_ret"]).float()
    yg = torch.from_numpy(ds["y_regime"]).long()
    ye = torch.from_numpy(ds["y_exec"]).float()
    bs = args.batch_size

    best_pf, best_state, stale = -1e9, None, 0
    n_epochs = args.epochs or FINETUNE_EPOCHS
    for ep in range(n_epochs):
        net.set_encoder_grad(ep >= FREEZE_ENCODER_EPOCHS)
        net.train()
        for idx in _batches(tr, bs, shuffle=True):
            xb = Xt[idx].to(dev)
            out = net.forward_dict(xb)
            loss = (LOSS_W_DIRECTION * ce_dir(out["direction"], yd[idx].to(dev))
                    + LOSS_W_QUANTILE * pinball(out["quantile"], yr[idx].to(dev))
                    + LOSS_W_REGIME * ce(out["regime"], yg[idx].to(dev))
                    # exec head: real MFE/MAE/timing targets, not a proxy
                    + LOSS_W_EXEC * mse(out["exec"], ye[idx].to(dev)))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        # model selection on the calib slice (test stays untouched)
        net.eval()
        preds = []
        with torch.no_grad():
            for idx in _batches(cal, bs, shuffle=False):
                d = net.forward_dict(Xt[idx].to(dev))["direction"]
                preds.append(d.argmax(1).cpu().numpy())
        vpred = np.concatenate(preds)
        vpf = _direction_pf(vpred, ds["y_ret"][cal])
        vacc = _accuracy(vpred, ds["y_dir"][cal])
        log.info("[finetune] epoch %d/%d  calib_PF=%.3f  calib_acc=%.3f",
                 ep + 1, n_epochs, vpf, vacc)
        if vpf > best_pf:
            best_pf, stale = vpf, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                log.info("[finetune] early stop at epoch %d", ep + 1)
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    ckpt = _ARTIFACT_DIR / "aurum_net.pt"
    torch.save({"state_dict": net.state_dict(), "norm": ds["norm"],
                "n_samples": n, "calib_pf": float(best_pf)}, ckpt)
    log.info("[finetune] best calib_PF=%.3f -> %s", best_pf, ckpt.name)
    return 0


# ===========================================================================
# Phase 4 — meta-label gate
# ===========================================================================
def _meta_features_for(ds: dict, primary: np.ndarray):
    from aurum.meta_label import build_meta_features
    realized_vol = ds["X"]["M5"][:, :, 0].std(axis=1).astype(np.float32)
    atr_norm = ds["X"]["M5"][:, -1, 7]
    return build_meta_features(primary, realized_vol, atr_norm)


def cmd_meta(args, ds=None) -> int:
    import torch
    from aurum.datamodule import build_dataset
    from aurum.model import create_aurum_net
    from aurum.meta_label import build_meta_target, create_meta_gate

    ckpt = _ARTIFACT_DIR / "aurum_net.pt"
    if not ckpt.exists():
        log.error("[meta] %s missing — run `finetune` first", ckpt.name)
        return 1
    _seed_all()
    dev = _device()
    net = create_aurum_net().to(dev)
    net.load_state_dict(torch.load(ckpt, map_location=dev)["state_dict"])
    net.eval()

    ds = ds or build_dataset(labelled=True, max_bars=args.max_bars)
    n = len(ds["X_flat"])
    tr, cal, te = _three_way_split(n)
    primary = _predict_primary(net, ds["X_flat"], dev, args.batch_size)
    feats = _meta_features_for(ds, primary)
    primary_dir = primary[:, SLICE_DIR[0]:SLICE_DIR[1]].argmax(1)
    target = build_meta_target(primary_dir, ds["y_dir"], ds["y_ret"])

    # Train the gate on train only — calib/test stay clean for conformal
    # and the final deploy decision.
    fit_idx = np.arange(tr.start, tr.stop)
    gate = create_meta_gate(use_gpu=args.use_gpu)
    gate.fit(feats[fit_idx], target[fit_idx])
    # report precision on the test window
    test_idx = np.arange(te.start, te.stop)
    pact = gate.predict_act_prob(feats[test_idx])
    acted = pact >= META_ACT_THRESHOLD
    prec = float(target[test_idx][acted].mean()) if acted.any() else 0.0
    log.info("[meta] test precision @act=%.3f  act_rate=%.3f",
             prec, float(acted.mean()))
    gate.save(_ARTIFACT_DIR / "aurum_meta.json")
    gate.export_onnx(_ARTIFACT_DIR / "M4GOLD_AURUM_META_GOLD.onnx")
    return 0


# ===========================================================================
# Phase 4 — conformal calibration
# ===========================================================================
def cmd_conformal(args, ds=None) -> int:
    import torch
    from aurum.datamodule import build_dataset
    from aurum.model import create_aurum_net
    from aurum.conformal import calibrate_threshold, evaluate_coverage

    ckpt = _ARTIFACT_DIR / "aurum_net.pt"
    if not ckpt.exists():
        log.error("[conformal] %s missing — run `finetune` first", ckpt.name)
        return 1
    _seed_all()
    dev = _device()
    net = create_aurum_net().to(dev)
    net.load_state_dict(torch.load(ckpt, map_location=dev)["state_dict"])
    net.eval()

    ds = ds or build_dataset(labelled=True, max_bars=args.max_bars)
    n = len(ds["X_flat"])
    tr, cal, te = _three_way_split(n)
    # Calibrate on the calib slice — disjoint from train and from test.
    primary = _predict_primary(net, ds["X_flat"][cal], dev, args.batch_size)
    probs = primary[:, SLICE_DIR[0]:SLICE_DIR[1]]
    labels = ds["y_dir"][cal]
    q_hat = calibrate_threshold(probs, labels)
    cov = evaluate_coverage(probs, labels, q_hat)
    log.info("[conformal] q_hat=%.4f  coverage=%.3f  singleton_frac=%.3f",
             q_hat, cov["coverage"], cov["singleton_frac"])
    (_ARTIFACT_DIR / "AURUM_conformal.json").write_text(
        json.dumps({"q_hat": q_hat, **cov}, indent=2))
    return 0


# ===========================================================================
# Export — deploy gate on AURUM's TRUE deployed performance
# ===========================================================================
def cmd_export(args, ds=None) -> int:
    import torch
    from aurum.datamodule import build_dataset
    from aurum.model import create_aurum_net
    from aurum.export import export_main_net, write_spec
    from aurum.meta_label import create_meta_gate

    ckpt = _ARTIFACT_DIR / "aurum_net.pt"
    if not ckpt.exists():
        log.error("[export] %s missing — run `finetune` first", ckpt.name)
        return 1
    _seed_all()
    dev = _device()
    blob = torch.load(ckpt, map_location="cpu")
    net = create_aurum_net()
    net.load_state_dict(blob["state_dict"])
    net.to(dev).eval()

    conf_path = _ARTIFACT_DIR / "AURUM_conformal.json"
    q_hat = (json.loads(conf_path.read_text())["q_hat"]
             if conf_path.exists() else 0.5)
    meta_path = _ARTIFACT_DIR / "aurum_meta.json"
    meta_gate = None
    if meta_path.exists():
        try:
            meta_gate = create_meta_gate().load(meta_path)
        except Exception as e:  # noqa: BLE001
            log.warning("[export] could not load meta gate (%s)", e)

    ds = ds or build_dataset(labelled=True, max_bars=args.max_bars)
    n = len(ds["X_flat"])
    tr, cal, te = _three_way_split(n)
    test_idx = np.arange(te.start, te.stop)
    fit_idx = np.arange(tr.start, cal.stop)        # train + calib

    # --- AURUM's TRUE deployed performance, on the untouched test window --
    primary_te = _predict_primary(net, ds["X_flat"][te], dev, args.batch_size)
    meta_feats_te = _meta_features_for(
        {"X": {"M5": ds["X"]["M5"][te]}}, primary_te)
    dep = evaluate_deployed(primary_te, ds["y_dir"][te], ds["y_ret"][te],
                            q_hat=q_hat, meta_gate=meta_gate,
                            meta_feats=meta_feats_te)
    # --- apples-to-apples baseline on the same test window ----------------
    base_test_pf = _xgb_test_window_pf(ds, fit_idx, test_idx, args.use_gpu)

    aurum_pf = dep["filtered_pf"]
    onnx_path, onnx_ok = export_main_net(net.cpu(), _ARTIFACT_DIR)
    enough_trades = dep["n_trades"] >= _GATE_MIN_TRADES
    deploy = bool(onnx_ok and enough_trades
                  and aurum_pf >= GATE_MIN_PF
                  and aurum_pf >= base_test_pf + GATE_MIN_EXCESS_VS_BASELINE)

    report = {
        "deploy": deploy,
        "deploy_metric": "filtered_pf (conformal + meta gate) on test window",
        "aurum": {
            "filtered_pf": round(aurum_pf, 4),
            "raw_direction_pf": round(dep["raw_pf"], 4),
            "n_trades": dep["n_trades"],
            "trade_rate": round(dep["trade_rate"], 4),
            "win_rate": round(dep["win_rate"], 4),
            "direction_acc": round(dep["direction_acc"], 4),
            "singleton_frac": round(dep["singleton_frac"], 4),
            "calib_pf": round(float(blob.get("calib_pf", 0.0)), 4),
        },
        "baseline_same_test_window_pf": round(base_test_pf, 4),
        "gate": {
            "min_pf": GATE_MIN_PF,
            "min_excess_vs_baseline": GATE_MIN_EXCESS_VS_BASELINE,
            "min_trades": _GATE_MIN_TRADES,
            "onnx_parity_ok": onnx_ok,
        },
        "test_window_samples": int(len(test_idx)),
        "conformal_q_hat": q_hat,
    }
    base_cv = _ARTIFACT_DIR / "AURUM_baseline_report.json"
    if base_cv.exists():
        report["baseline_purged_cv"] = json.loads(base_cv.read_text())
    (_ARTIFACT_DIR / "AURUM_report.json").write_text(
        json.dumps(report, indent=2))

    write_spec(_ARTIFACT_DIR, conformal_q=q_hat, norm=blob["norm"],
               cv_report=report, deploy=deploy)

    log.info("=" * 70)
    log.info("[export] AURUM deployed filtered_PF = %.3f   "
             "(raw dir PF %.3f, %d trades, WR %.1f%%)",
             aurum_pf, dep["raw_pf"], dep["n_trades"],
             100 * dep["win_rate"])
    log.info("[export] baseline on same test window = %.3f", base_test_pf)
    log.info("[export] gate: PF>=%.2f and PF>=baseline+%.2f and trades>=%d "
             "-> DEPLOY=%s", GATE_MIN_PF, GATE_MIN_EXCESS_VS_BASELINE,
             _GATE_MIN_TRADES, deploy)
    if not deploy and enough_trades and onnx_ok:
        gap = (base_test_pf + GATE_MIN_EXCESS_VS_BASELINE) - aurum_pf
        log.info("[export] short of the gate by %.3f PF — see "
                 "AURUM_report.json", max(0.0, gap))
    log.info("=" * 70)
    return 0


# ===========================================================================
# all — full pipeline, single dataset build
# ===========================================================================
def cmd_all(args) -> int:
    from aurum.datamodule import build_dataset
    _seed_all()
    log.info("[all] building datasets once (labelled + unlabelled) ...")
    ds_lab = build_dataset(labelled=True, max_bars=args.max_bars)
    ds_unlab = build_dataset(labelled=False, max_bars=args.max_bars)

    phases = [
        ("baseline",  cmd_baseline,  ds_lab),
        ("pretrain",  cmd_pretrain,  ds_unlab),
        ("finetune",  cmd_finetune,  ds_lab),
        ("meta",      cmd_meta,      ds_lab),
        ("conformal", cmd_conformal, ds_lab),
        ("export",    cmd_export,    ds_lab),
    ]
    for name, fn, ds in phases:
        rc = fn(args, ds=ds)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if rc != 0:
            log.error("[all] phase '%s' returned %d — stopping", name, rc)
            return rc
    return 0


# ===========================================================================
# Helpers / CLI
# ===========================================================================
def _device() -> str:
    try:
        from hardware_detector import get as _hw
        return _hw().device
    except Exception:
        return "cpu"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_aurum.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def _common(sp):
        sp.add_argument("--max-bars", type=int, default=None,
                        help="cap M5 bars (use the most recent N)")
        sp.add_argument("--batch-size", type=int, default=256)
        sp.add_argument("--epochs", type=int, default=0,
                        help="override default epoch count (0 = use config)")
        sp.add_argument("--use-gpu", action="store_true",
                        help="GPU for the XGBoost baseline / meta gate")
        sp.add_argument("--dlinear-epochs", type=int, default=15)

    for name, fn in [("baseline", cmd_baseline), ("pretrain", cmd_pretrain),
                     ("finetune", cmd_finetune), ("meta", cmd_meta),
                     ("conformal", cmd_conformal), ("export", cmd_export),
                     ("all", cmd_all)]:
        sp = sub.add_parser(name, help=(fn.__doc__ or name))
        _common(sp)
        sp.set_defaults(func=fn)
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stdout, force=True)
    args = build_parser().parse_args(argv)
    t0 = time.time()
    # cmd_all takes (args); the per-phase commands take (args, ds=None).
    rc = args.func(args)
    log.info("[%s] done in %.0fs (rc=%d)", args.command, time.time() - t0, rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
