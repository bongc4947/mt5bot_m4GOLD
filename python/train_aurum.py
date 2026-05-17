"""
train_aurum.py — unified CLI for the AURUM v2 AI stack.

Subcommands map 1:1 to the phased rollout in docs/DESIGN_AURUM.md:

    python python/train_aurum.py baseline    # Phase 1 — purged-CV controls
    python python/train_aurum.py pretrain    # Phase 2 — L0 SSL encoders
    python python/train_aurum.py finetune    # Phase 2-3 — backbone + heads
    python python/train_aurum.py meta        # Phase 4 — meta-label gate
    python python/train_aurum.py conformal   # Phase 4 — conformal calibration
    python python/train_aurum.py export      # ONNX bundle + spec
    python python/train_aurum.py all         # full pipeline, gated

Hardware (CUDA / CPU) is auto-detected via hardware_detector.py. Heavy
SSL pretraining is best run in the cloud (cloud/runner.sh); fine-tuning
and everything downstream fits a modest local GPU.
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
    SLICE_DIR, SLICE_QUANT,
)

log = logging.getLogger(__name__)
_ARTIFACT_DIR = Path(__file__).parent.parent / "onnx_out"


# ===========================================================================
# Metrics
# ===========================================================================
def _direction_pf(pred_dir: np.ndarray, y_ret: np.ndarray) -> float:
    """Profit factor of acting on a 3-class direction prediction."""
    sign = np.where(pred_dir == 2, 1.0, np.where(pred_dir == 0, -1.0, 0.0))
    pnl = sign * y_ret
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses <= 1e-12:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def _accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float((pred == true).mean())


# ===========================================================================
# Phase 1 — baselines under purged CV
# ===========================================================================
def cmd_baseline(args) -> int:
    from aurum.datamodule import build_dataset
    from cv.purged_kfold import PurgedKFold
    from baselines.xgb_direction import create_xgb_baseline

    ds = build_dataset(labelled=True, max_bars=args.max_bars)
    X, y, y_ret = ds["X_flat"], ds["y_dir"], ds["y_ret"]
    pk = PurgedKFold(n_splits=CV_N_SPLITS, horizon=LABEL_HORIZON_BARS,
                     embargo_pct=CV_EMBARGO_PCT)

    report = {"dlinear": [], "xgboost": []}
    for fold, (tr, te) in enumerate(pk.split(len(y))):
        # XGBoost baseline
        xgb = create_xgb_baseline(use_gpu=args.use_gpu)
        xgb.fit(X[tr], y[tr])
        xpred = xgb.predict(X[te])
        xpf = _direction_pf(xpred, y_ret[te])
        report["xgboost"].append(xpf)
        # DLinear baseline
        dpf = _train_dlinear_fold(X[tr], y[tr], X[te], y_ret[te], args)
        report["dlinear"].append(dpf)
        log.info("[baseline] fold %d/%d  xgb_PF=%.3f  dlinear_PF=%.3f",
                 fold + 1, pk.get_n_splits(), xpf, dpf)

    summary = {k: {"mean_pf": float(np.mean(v)),
                   "folds": [round(x, 3) for x in v]}
               for k, v in report.items()}
    best = max(summary["dlinear"]["mean_pf"], summary["xgboost"]["mean_pf"])
    summary["best_baseline_pf"] = best
    out = _ARTIFACT_DIR / "AURUM_baseline_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    log.info("[baseline] DLinear meanPF=%.3f  XGBoost meanPF=%.3f  -> %s",
             summary["dlinear"]["mean_pf"], summary["xgboost"]["mean_pf"],
             out.name)
    return 0


def _train_dlinear_fold(Xtr, ytr, Xte, yret_te, args) -> float:
    """Train DLinear on one CV fold, return test-fold profit factor."""
    import torch
    from baselines.dlinear import create_dlinear
    dev = _device()
    torch.manual_seed(SEED)
    model = create_dlinear().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    # Same inverse-frequency balancing AURUM gets — fair comparison.
    counts = np.bincount(ytr, minlength=3).astype(np.float64)
    cw = counts.sum() / np.maximum(counts, 1.0)
    cw = cw / cw.mean()
    lossf = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32, device=dev))
    Xt = torch.from_numpy(Xtr).float()
    yt = torch.from_numpy(ytr).long()
    bs = 4096
    for ep in range(args.dlinear_epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            xb, yb = Xt[idx].to(dev), yt[idx].to(dev)
            loss = lossf(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte).float().to(dev)).argmax(1).cpu().numpy()
    return _direction_pf(pred, yret_te)


# ===========================================================================
# Phase 2 — SSL pretraining
# ===========================================================================
def cmd_pretrain(args) -> int:
    from aurum.datamodule import build_dataset
    from aurum.pretrain import pretrain_all
    from aurum.aurum_config import SSL_EPOCHS
    ds = build_dataset(labelled=False, max_bars=args.max_bars)
    epochs = args.epochs or SSL_EPOCHS
    paths = pretrain_all(ds, device=_device(), epochs=epochs)
    log.info("[pretrain] encoders: %s", {k: v.name for k, v in paths.items()})
    return 0


# ===========================================================================
# Phase 2-3 — fine-tune backbone + multi-task heads
# ===========================================================================
def cmd_finetune(args) -> int:
    import torch
    from aurum.datamodule import build_dataset
    from aurum.model import create_aurum_net
    from aurum.heads import PinballLoss

    dev = _device()
    torch.manual_seed(SEED)
    ds = build_dataset(labelled=True, max_bars=args.max_bars)
    X = ds["X_flat"]
    n = len(X)
    n_val = int(n * VAL_SPLIT)
    gap = LABEL_HORIZON_BARS
    tr = slice(0, n - n_val - gap)
    va = slice(n - n_val, n)

    net = create_aurum_net().to(dev)
    net.load_norm(ds["norm"])
    # Load SSL-pretrained encoders if present.
    ssl_paths = {tf: _ARTIFACT_DIR / f"aurum_ssl_encoder_{tf}.pt"
                 for tf in net.tf_order}
    if all(p.exists() for p in ssl_paths.values()):
        net.load_encoders(ssl_paths)
        log.info("[finetune] loaded SSL-pretrained encoders")
    else:
        log.warning("[finetune] no SSL encoders found — training from scratch. "
                    "Run `train_aurum.py pretrain` first for best results.")

    opt = torch.optim.AdamW(net.parameters(), lr=FINETUNE_LR,
                            weight_decay=WEIGHT_DECAY)
    # Triple-barrier labels are skewed (SL at 1 ATR is hit before TP at
    # 2 ATR more often) — inverse-frequency class weights stop the
    # direction head collapsing onto the majority class.
    counts = np.bincount(ds["y_dir"][tr], minlength=3).astype(np.float64)
    cls_w = (counts.sum() / np.maximum(counts, 1.0))
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
    bs = args.batch_size

    best_pf, best_state, stale = -1e9, None, 0
    for ep in range(args.epochs or FINETUNE_EPOCHS):
        net.set_encoder_grad(ep >= FREEZE_ENCODER_EPOCHS)
        net.train()
        for idx in _batches(tr, bs, shuffle=True):
            xb = Xt[idx].to(dev)
            out = net.forward_dict(xb)
            loss = (LOSS_W_DIRECTION * ce_dir(out["direction"], yd[idx].to(dev))
                    + LOSS_W_QUANTILE * pinball(out["quantile"], yr[idx].to(dev))
                    + LOSS_W_REGIME * ce(out["regime"], yg[idx].to(dev))
                    + LOSS_W_EXEC * mse(out["exec"].mean(1),
                                        yr[idx].to(dev).abs()))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        # validation profit factor
        net.eval()
        preds = []
        with torch.no_grad():
            for idx in _batches(va, bs, shuffle=False):
                d = net.forward_dict(Xt[idx].to(dev))["direction"]
                preds.append(d.argmax(1).cpu().numpy())
        vpred = np.concatenate(preds)
        vpf = _direction_pf(vpred, ds["y_ret"][va])
        vacc = _accuracy(vpred, ds["y_dir"][va])
        log.info("[finetune] epoch %d  val_PF=%.3f  val_acc=%.3f",
                 ep + 1, vpf, vacc)
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
    import torch as _t
    _t.save({"state_dict": net.state_dict(), "norm": ds["norm"],
             "val_pf": best_pf}, ckpt)
    log.info("[finetune] best val_PF=%.3f -> %s", best_pf, ckpt.name)
    return 0


# ===========================================================================
# Phase 4 — meta-label gate
# ===========================================================================
def cmd_meta(args) -> int:
    import torch
    from aurum.datamodule import build_dataset
    from aurum.model import create_aurum_net, AurumExportWrapper
    from aurum.meta_label import (build_meta_features, build_meta_target,
                                  create_meta_gate)

    ckpt = _ARTIFACT_DIR / "aurum_net.pt"
    if not ckpt.exists():
        log.error("[meta] %s missing — run `finetune` first", ckpt.name)
        return 1
    dev = _device()
    net = create_aurum_net().to(dev)
    blob = torch.load(ckpt, map_location=dev)
    net.load_state_dict(blob["state_dict"])
    net.eval()

    ds = build_dataset(labelled=True, max_bars=args.max_bars)
    X = torch.from_numpy(ds["X_flat"]).float()
    wrapper = AurumExportWrapper(net).eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), args.batch_size):
            outs.append(wrapper(X[i:i + args.batch_size].to(dev)).cpu().numpy())
    primary = np.concatenate(outs)

    # Causal context — both computable live by the EA with identical maths:
    #   realized_vol = std of the 128 past M5 log-returns (channel 0)
    #   atr_norm     = the latest M5 atr_norm channel (channel 7)
    realized_vol = ds["X"]["M5"][:, :, 0].std(axis=1).astype(np.float32)
    atr_norm = ds["X"]["M5"][:, -1, 7]
    feats = build_meta_features(primary, realized_vol, atr_norm)
    primary_dir = primary[:, SLICE_DIR[0]:SLICE_DIR[1]].argmax(1)
    target = build_meta_target(primary_dir, ds["y_dir"], ds["y_ret"])

    n = len(feats)
    n_val = int(n * VAL_SPLIT)
    gate = create_meta_gate(use_gpu=args.use_gpu)
    gate.fit(feats[:n - n_val], target[:n - n_val])
    pact = gate.predict_act_prob(feats[n - n_val:])
    from aurum.aurum_config import META_ACT_THRESHOLD
    acted = pact >= META_ACT_THRESHOLD
    val_target = target[n - n_val:]
    precision = (val_target[acted].mean() if acted.any() else 0.0)
    log.info("[meta] val precision @act=%.3f  act_rate=%.3f",
             float(precision), float(acted.mean()))
    gate.export_onnx(_ARTIFACT_DIR / f"M4GOLD_AURUM_META_GOLD.onnx")
    return 0


# ===========================================================================
# Phase 4 — conformal calibration
# ===========================================================================
def cmd_conformal(args) -> int:
    import torch
    from aurum.datamodule import build_dataset
    from aurum.model import create_aurum_net, AurumExportWrapper
    from aurum.conformal import calibrate_threshold, evaluate_coverage

    ckpt = _ARTIFACT_DIR / "aurum_net.pt"
    if not ckpt.exists():
        log.error("[conformal] %s missing — run `finetune` first", ckpt.name)
        return 1
    dev = _device()
    net = create_aurum_net().to(dev)
    net.load_state_dict(torch.load(ckpt, map_location=dev)["state_dict"])
    net.eval()
    wrapper = AurumExportWrapper(net).eval()

    ds = build_dataset(labelled=True, max_bars=args.max_bars)
    X = torch.from_numpy(ds["X_flat"]).float()
    n = len(X)
    cal_n = int(n * CONFORMAL_CAL_FRAC)
    cal_slice = slice(n - cal_n, n)            # most recent slice = calibration
    outs = []
    with torch.no_grad():
        xs = X[cal_slice]
        for i in range(0, len(xs), args.batch_size):
            outs.append(wrapper(xs[i:i + args.batch_size].to(dev)).cpu().numpy())
    primary = np.concatenate(outs)
    probs = primary[:, SLICE_DIR[0]:SLICE_DIR[1]]
    labels = ds["y_dir"][cal_slice]
    q_hat = calibrate_threshold(probs, labels)
    cov = evaluate_coverage(probs, labels, q_hat)
    log.info("[conformal] q_hat=%.4f  coverage=%.3f  singleton_frac=%.3f",
             q_hat, cov["coverage"], cov["singleton_frac"])
    (_ARTIFACT_DIR / "AURUM_conformal.json").write_text(
        json.dumps({"q_hat": q_hat, **cov}, indent=2))
    return 0


# ===========================================================================
# Export
# ===========================================================================
def cmd_export(args) -> int:
    import torch
    from aurum.model import create_aurum_net
    from aurum.export import export_main_net, write_spec

    ckpt = _ARTIFACT_DIR / "aurum_net.pt"
    if not ckpt.exists():
        log.error("[export] %s missing — run `finetune` first", ckpt.name)
        return 1
    blob = torch.load(ckpt, map_location="cpu")
    net = create_aurum_net()
    net.load_state_dict(blob["state_dict"])

    conf_path = _ARTIFACT_DIR / "AURUM_conformal.json"
    q_hat = (json.loads(conf_path.read_text())["q_hat"]
             if conf_path.exists() else 0.5)
    base_path = _ARTIFACT_DIR / "AURUM_baseline_report.json"
    cv_report = (json.loads(base_path.read_text())
                 if base_path.exists() else {})

    _, onnx_ok = export_main_net(net, _ARTIFACT_DIR)
    # Deploy gate — AURUM ships only if it beats the baseline on PF.
    aurum_pf = float(blob.get("val_pf", 0.0))
    best_base = float(cv_report.get("best_baseline_pf", 0.0))
    deploy = bool(onnx_ok and aurum_pf >= GATE_MIN_PF
                  and aurum_pf >= best_base + GATE_MIN_EXCESS_VS_BASELINE)
    write_spec(_ARTIFACT_DIR, conformal_q=q_hat, norm=blob["norm"],
               cv_report={"aurum_val_pf": aurum_pf,
                          "best_baseline_pf": best_base, **cv_report},
               deploy=deploy)
    log.info("[export] aurum_PF=%.3f  best_baseline_PF=%.3f  -> deploy=%s",
             aurum_pf, best_base, deploy)
    return 0


def cmd_all(args) -> int:
    for fn in (cmd_baseline, cmd_pretrain, cmd_finetune, cmd_meta,
               cmd_conformal, cmd_export):
        rc = fn(args)
        if rc != 0:
            log.error("[all] %s returned %d — stopping", fn.__name__, rc)
            return rc
    return 0


# ===========================================================================
# Helpers
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
        sp = sub.add_parser(name, help=fn.__doc__ or name)
        _common(sp)
        sp.set_defaults(func=fn)
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stdout, force=True)
    args = build_parser().parse_args(argv)
    t0 = time.time()
    rc = args.func(args)
    log.info("[%s] done in %.0fs (rc=%d)", args.command, time.time() - t0, rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
