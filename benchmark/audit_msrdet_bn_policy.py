"""Surgical BN-policy dry run for the pinned MSRDet reconstruction.

No dataset, checkpoint, training optimizer, detection evaluation or third-party
package import is used. Synthetic non-default source BN values make state-copy
assertions non-vacuous. The train-mode gradient check verifies a known BN null
effect; synthetic activation differences are not evidence of detection accuracy.

The pure policy_state function is usable with reviewed, real source tensors in a
later trainer. It always starts from the same saved target initialization.
"""
from pathlib import Path
import argparse
from copy import deepcopy
import hashlib
import json
import os
import sys
import traceback

import audit_msrdet_structure as structure

BN_AFFINE = ("weight", "bias")
BN_STATS = ("running_mean", "running_var", "num_batches_tracked")
POLICIES = {"A0R0": (False, False), "A1R0": (True, False),
            "A0R1": (False, True), "A1R1": (True, True)}


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode("utf-8"))
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def inspect_pairs(source, target, nn):
    """A narrow, explicit Conv wrapper rule, not general graph equivalence."""
    source_modules, target_modules = dict(source.named_modules()), dict(target.named_modules())
    source_state, target_state = source.state_dict(), target.state_dict()
    rows = []
    for path, bn in target_modules.items():
        if not isinstance(bn, nn.BatchNorm2d) or not path.endswith(".bn"):
            continue
        origin = source_modules.get(path)
        wrapper_path = path.rsplit(".", 1)[0]
        conv_path = wrapper_path + ".conv"
        conv, source_conv = target_modules.get(conv_path), source_modules.get(conv_path)
        if not isinstance(origin, nn.BatchNorm2d) or not isinstance(conv, nn.Conv2d) or not isinstance(source_conv, nn.Conv2d):
            continue
        keys = [path + "." + suffix for suffix in BN_AFFINE + BN_STATS]
        if not all(k in source_state and k in target_state and source_state[k].shape == target_state[k].shape for k in keys):
            continue
        conv_key = conv_path + ".weight"
        matched_conv = source_state[conv_key].shape == target_state[conv_key].shape
        operator_difference = any(getattr(conv, field) != getattr(source_conv, field)
                                  for field in ("kernel_size", "groups", "stride", "padding", "dilation"))
        rows.append({"bn_path": path, "channels": bn.num_features,
                     "bn_keys": keys, "conv_key": conv_key,
                     "conv_key_shape_matched": matched_conv,
                     "source_conv": structure.operator_signature(source_conv),
                     "target_conv": structure.operator_signature(conv),
                     "category": "matched_conv" if matched_conv else
                                 ("kernel_or_group_change" if operator_difference else "projection_input_width_change"),
                     "target_bn_momentum": bn.momentum})
    return rows


def policy_state(source_state, initial_target_state, affected_paths, affine_source, stats_source):
    """Return fixed-architecture key/shape transfer with local BN interventions.

    A0/R0 retain saved target initialization, including the constructor's own BN
    buffer values. They do not silently call reset_parameters/reset_running_stats.
    All tensor categories outside affected BN paths use exactly the same transfer.
    """
    state = {key: value.detach().clone() for key, value in initial_target_state.items()}
    copied = []
    for key, value in source_state.items():
        if key in state and value.shape == state[key].shape:
            state[key] = value.detach().clone()
            copied.append(key)
    for path in affected_paths:
        for suffix in BN_AFFINE + BN_STATS:
            key = path + "." + suffix
            if key not in initial_target_state or key not in source_state or initial_target_state[key].shape != source_state[key].shape:
                raise ValueError("Affected BN is not fully key/shape compatible: " + key)
            use_source = affine_source if suffix in BN_AFFINE else stats_source
            state[key] = (source_state[key] if use_source else initial_target_state[key]).detach().clone()
    return state, copied


def before_detect(model, x):
    """Read pre-head features; never invoke detector decoding/evaluation paths."""
    class StopBeforeDetect(Exception):
        pass
    features = []
    def intercept(module, inputs):
        features.extend(value.detach().clone() for value in inputs[0])
        raise StopBeforeDetect()
    handle = model.model[-1].register_forward_pre_hook(intercept)
    try:
        model(x)
    except StopBeforeDetect:
        pass
    finally:
        handle.remove()
    if not features:
        raise RuntimeError("No detection-head inputs were captured")
    return features


def forward_and_gradient(model, x, torch):
    model.train()
    output = model(x)
    if not isinstance(output, list):
        raise RuntimeError("Expected raw train-mode detection outputs")
    loss = sum(value.square().mean() for value in output)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite synthetic diagnostic loss")
    loss.backward()
    gradients = {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None}
    return [value.detach().clone() for value in output], gradients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if out.exists() or out.drive.upper() != "D:":
        raise SystemExit("Output must be a new file on D: " + str(out))
    for name in ("TEMP", "TMP", "TORCH_HOME", "MPLCONFIGDIR", "YOLO_CONFIG_DIR"):
        if not os.environ.get(name, "").upper().startswith("D:"):
            raise SystemExit("Dot-source project-env.ps1 first: " + name)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {"scope": "policy and BN-mechanism dry run only; synthetic inputs and source BN values; no dataset/checkpoint/optimizer/evaluation",
              "script_sha256": structure.digest(Path(__file__)),
              "structure_script_sha256": structure.digest(Path(structure.__file__)),
              "seed": args.seed, "status": "started", "source_manifest": {}}
    try:
        import torch
        import torch.nn as nn
        import yaml
        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        for relative, expected in structure.PINNED.items():
            actual = structure.digest(structure.REPO / relative)
            if actual != expected:
                raise RuntimeError("Pinned source hash mismatch: " + relative)
            result["source_manifest"][relative] = actual
        ns = structure.namespace(torch, nn, result)
        models = []
        with torch.no_grad(), torch.device("cpu"):
            for filename in ("yolov8.yaml", "MSRDet.yaml"):
                cfg = yaml.safe_load((structure.REPO / "ultralytics/cfg/models/v8" / filename).read_text(encoding="utf-8"))
                cfg["scale"] = "n"
                if cfg.get("activation"):
                    raise RuntimeError("Unreviewed YAML activation")
                allowed = {"Conv", "C2f", "C2fk", "SPPF", "GatedSPPF", "nn.Upsample", "Concat", "FEM", "HTEM", "Focus", "Fusion_2in_mod", "Detect"}
                if not {row[2] for row in cfg["backbone"] + cfg["head"]} <= allowed:
                    raise RuntimeError("Unreviewed YAML module")
                models.append(ns["DetectionModel"](deepcopy(cfg), verbose=False))
        source, target = models
        pairs = inspect_pairs(source, target, nn)
        affected = [row for row in pairs if not row["conv_key_shape_matched"]]
        if len(affected) != 12:
            raise RuntimeError("Pinned configuration no longer has the audited 12 BN sites")
        paths = [row["bn_path"] for row in affected]
        treatment_keys = {key for row in affected for key in row["bn_keys"]}
        initial = {key: value.detach().clone() for key, value in target.state_dict().items()}
        # Only BN tensors get non-default markers. No source convolutions or
        # model activations are trained, and no real pretrained values are used.
        with torch.no_grad():
            for index, (path, bn) in enumerate((x for x in source.named_modules() if isinstance(x[1], nn.BatchNorm2d))):
                channels = bn.num_features
                bn.weight.copy_(torch.linspace(0.6, 1.4, channels))
                bn.bias.copy_(torch.linspace(-0.2, 0.2, channels))
                bn.running_mean.copy_(torch.linspace(0.15, 0.35, channels))
                bn.running_var.copy_(torch.linspace(0.7, 1.7, channels))
                bn.num_batches_tracked.fill_(17 + index)
        source_state = source.state_dict()
        states, policy_audit = {}, {}
        for label, (affine, stats) in POLICIES.items():
            state, copied = policy_state(source_state, initial, paths, affine, stats)
            expected_from_source = set(copied) - {path + "." + suffix for path in paths
                                                   for suffix in (BN_AFFINE if not affine else ()) + (BN_STATS if not stats else ())}
            for key, value in state.items():
                expected = source_state[key] if key in expected_from_source else initial[key]
                if not torch.equal(value, expected):
                    raise AssertionError(label + " source/initial tensor check failed: " + key)
            changed = sorted(key for key in treatment_keys if not torch.equal(state[key], initial[key]))
            if len(changed) != (24 if affine else 0) + (36 if stats else 0):
                raise AssertionError("Synthetic BN marker check is vacuous")
            states[label] = state
            policy_audit[label] = {"affine_from_source": affine, "stats_from_source": stats,
                                   "state_sha256": state_hash(state),
                                   "treatment_keys_changed_from_saved_initial": changed,
                                   "source_copied_state_tensors": len(expected_from_source)}
        for key in initial:
            if key not in treatment_keys and not all(torch.equal(states["A0R0"][key], state[key]) for state in states.values()):
                raise AssertionError("A policy changed a nontreatment tensor: " + key)
        # A1R1 must equal the untouched original key/shape loader exactly.
        original_load_target = deepcopy(target)
        original_load_target.load(source, verbose=False)
        if not all(torch.equal(value, states["A1R1"][key]) for key, value in original_load_target.state_dict().items()):
            raise AssertionError("A1R1 differs from the pinned original BaseModel.load")
        # Critical null: in train mode, fixed-momentum running statistics and
        # counters are not inputs to the forward or gradient calculation.
        x = torch.rand((2, 3, 64, 64), generator=torch.Generator().manual_seed(args.seed + 1))
        a, b = deepcopy(target), deepcopy(target)
        a.load_state_dict(states["A0R0"], strict=True)
        b.load_state_dict(states["A0R1"], strict=True)
        outputs_a, grads_a = forward_and_gradient(a, x, torch)
        outputs_b, grads_b = forward_and_gradient(b, x, torch)
        outputs_equal = all(torch.equal(u, v) for u, v in zip(outputs_a, outputs_b))
        gradients_equal = grads_a.keys() == grads_b.keys() and all(torch.equal(grads_a[k], grads_b[k]) for k in grads_a)
        if not outputs_equal or not gradients_equal:
            raise AssertionError("Unexpected train-mode dependence on running buffers")
        decay_rows = []
        state_a, state_b = a.state_dict(), b.state_dict()
        for row in affected:
            path, momentum = row["bn_path"], row["target_bn_momentum"]
            if momentum is None:
                raise RuntimeError("Cumulative BN requires a different counter/decay analysis")
            errors = {}
            for suffix in ("running_mean", "running_var"):
                key = path + "." + suffix
                expected = (1 - momentum) * (states["A0R1"][key] - states["A0R0"][key])
                observed = state_b[key] - state_a[key]
                errors[suffix] = float((expected - observed).abs().max())
                if not torch.allclose(expected, observed, atol=2e-7, rtol=2e-6):
                    raise AssertionError("EMA initial-buffer decay check failed: " + key)
            decay_rows.append({"bn_path": path, "momentum": momentum, "one_forward_max_abs_errors": errors})
        # Eval-mode pre-head feature differences are a diagnostic of the API,
        # with synthetic non-neutral BN; they are not a detection evaluation.
        eval_features = {}
        for label in ("A0R0", "A0R1"):
            model = deepcopy(target)
            model.load_state_dict(states[label], strict=True)
            model.eval()
            with torch.no_grad():
                eval_features[label] = before_detect(model, x)
        feature_deltas = []
        for u, v in zip(eval_features["A0R0"], eval_features["A0R1"]):
            if not torch.isfinite(u).all() or not torch.isfinite(v).all():
                raise RuntimeError("Nonfinite synthetic eval features")
            feature_deltas.append({"shape": list(u.shape), "max_abs_difference": float((u - v).abs().max()),
                                   "relative_l2_difference": float((u - v).norm() / (u.norm() + 1e-12))})
        result.update({"status": "complete", "environment": {"python": sys.version, "torch": torch.__version__, "device": "cpu", "threads": 2},
                       "paired_bn_inventory": pairs, "affected_bn_sites": len(affected),
                       "affected_bn_channels": sum(row["channels"] for row in affected),
                       "affected_state_tensors": len(treatment_keys),
                       "affected_affine_parameter_tensors": 2 * len(affected),
                       "affected_affine_parameter_elements": 2 * sum(row["channels"] for row in affected),
                       "policy_audit": policy_audit, "outside_treatment_identical": True,
                       "A1R1_equals_original_BaseModel_load": True,
                       "train_mode_null": {"outputs_bitwise_equal": outputs_equal, "all_gradients_bitwise_equal": gradients_equal,
                                           "gradient_tensors_checked": len(grads_a), "one_forward_running_decay_checks": decay_rows},
                       "synthetic_eval_prehead_feature_differences": feature_deltas,
                       "full_ultralytics_package_imported": "ultralytics" in sys.modules,
                       "limitations": ["All source convolutions are random and BN values are synthetic markers, not learned pretrained weights.",
                                       "The narrow Conv-wrapper pairing rule does not prove semantic compatibility or recover all upstream dependencies.",
                                       "State checks and train-mode null do not establish benefit, harm, mAP effects or publication originality.",
                                       "No claim is made about whether the MSRDet authors used the audited loading path."]})
        if result["full_ultralytics_package_imported"] or result["inactive_class_constructor_calls"]:
            raise AssertionError("Unexpected third-party package import or guard constructor")
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(out), "status": result["status"], "sha256": structure.digest(out),
                          "affected_bn_sites": result.get("affected_bn_sites"),
                          "train_mode_null": {k: v for k, v in result.get("train_mode_null", {}).items() if k != "one_forward_running_decay_checks"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
