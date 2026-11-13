"""CPU-only, offline audit of the unmodified pinned MSRDet package.

Imports the full repository in a fresh process, then checks a six-class model,
real detection loss, one synthetic optimizer/EMA update and trainer setup order.
No dataset, detector evaluation, training run, installation or download is used.
All writes must remain on D:. Run only after dot-sourcing project-env.ps1.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import traceback


BASE = Path(__file__).resolve().parent
REPO = BASE / "external/MSRDet-7267acc2b1fb96d3af2dfe2e406d0f430e6abfc5"
WEIGHT_SHA = "31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--init-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir.resolve()
    if out.drive.upper() != "D:" or out.exists():
        raise SystemExit("A new D: output directory is required")
    for name in ("TEMP", "TMP", "TORCH_HOME", "MPLCONFIGDIR", "YOLO_CONFIG_DIR"):
        if not os.environ.get(name, "").upper().startswith("D:"):
            raise SystemExit("Dot-source project-env.ps1 first: " + name)
    if sha(args.weights) != WEIGHT_SHA:
        raise SystemExit("Unexpected checkpoint")
    out.mkdir(parents=True, exist_ok=False)
    config = out / "isolated_config"
    config.mkdir()
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", YOLO_AUTOINSTALL="false",
                      YOLO_CONFIG_DIR=str(config), YOLO_VERBOSE="false")
    sys.dont_write_bytecode = True
    blocked = []
    writes = []

    def audit(event, values):
        if event in ("socket.connect", "socket.bind", "socket.sendto", "socket.getaddrinfo",
                     "subprocess.Popen", "os.system", "os.posix_spawn"):
            blocked.append({"event": event, "target": repr(values)[:400]})
            # OSError is intentionally caught by the package connectivity probe.
            raise OSError("Offline CPU audit blocks " + event)
        if event == "open" and isinstance(values[0], (str, bytes)):
            path, mode, flags = values
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing:
                resolved = Path(os.fsdecode(path)).resolve()
                if resolved.drive.upper() != "D:":
                    blocked.append({"event": "write_outside_D", "target": str(resolved)})
                    raise OSError("CPU audit rejects write outside D:")
                if len(writes) < 100:
                    writes.append(str(resolved))

    sys.addaudithook(audit)
    result = {"status": "started", "scope": "full-package import and synthetic CPU mechanism audit only",
              "repository": str(REPO), "script_sha256": sha(__file__), "python": sys.version,
              "network_and_subprocess_blocked": True, "cuda_visible_devices": "-1",
              "blocked_events": blocked, "write_paths": writes, "packages": {}}
    for package in ("torch", "torchvision", "ultralytics", "numpy", "scipy", "pandas",
                    "matplotlib", "opencv-python", "thop", "ultralytics-thop", "pyyaml",
                    "tqdm", "requests", "psutil", "py-cpuinfo", "seaborn", "einops",
                    "timm", "lancedb", "duckdb", "albumentations"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    try:
        sys.path.insert(0, str(REPO))
        import torch
        import torch.nn as nn
        import yaml
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        import ultralytics
        result["package_import"] = {"file": str(Path(ultralytics.__file__).resolve()),
                                    "version": ultralytics.__version__}
        assert Path(ultralytics.__file__).resolve() == REPO / "ultralytics/__init__.py"
        from copy import deepcopy
        from types import SimpleNamespace
        from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight
        from ultralytics.utils.torch_utils import ModelEMA, strip_optimizer
        from ultralytics.models.yolo.detect.train import DetectionTrainer
        from ultralytics.utils import DEFAULT_CFG_DICT
        import audit_msrdet_bn_policy as policy

        record = json.loads((args.init_dir / "audit.json").read_text(encoding="utf-8"))
        initial_file = args.init_dir / "target_initial_state.pt"
        assert sha(initial_file) == record["target_initial_file_sha256"]
        initial = torch.load(initial_file, map_location="cpu", weights_only=True)
        assert policy.state_hash(initial) == record["target_initial_state_sha256"]
        # This uses the original package's own loader, not the installed modern package.
        source, checkpoint = attempt_load_one_weight(str(args.weights.resolve()), device="cpu", fuse=False)
        assert source.nc == 80
        assert policy.state_hash(source.state_dict()) == record["source_state_sha256"]
        cfg = yaml.safe_load((REPO / "ultralytics/cfg/models/v8/MSRDet.yaml").read_text(encoding="utf-8"))
        cfg["nc"], cfg["scale"] = 6, "n"
        torch.manual_seed(record["seed"])
        with torch.no_grad(), torch.device("cpu"):
            target = DetectionModel(deepcopy(cfg), verbose=False)
        target.load_state_dict(initial, strict=True)
        target.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        target.nc = 6
        assert target.model[-1].nc == 6 and target.model[-1].no == 70
        states = {}
        for label, (affine, stats) in policy.POLICIES.items():
            states[label], _ = policy.policy_state(source.state_dict(), initial,
                                                  record["affected_bn_paths"], affine, stats)
            assert policy.state_hash(states[label]) == record["policies"][label]["state_sha256"]
        result["source_and_target"] = {"source_nc": source.nc, "target_nc": target.nc,
                                        "target_head_nc": target.model[-1].nc,
                                        "target_state_tensors": len(initial),
                                        "source_hash_matches_prior_audit": True,
                                        "all_policy_hashes_match_prior_audit": True}

        generator = torch.Generator().manual_seed(20261103)
        batch = {"img": torch.rand(2, 3, 64, 64, generator=generator),
                 "batch_idx": torch.tensor([0., 1.]), "cls": torch.tensor([[0.], [5.]]),
                 "bboxes": torch.tensor([[.5, .5, .5, .5], [.45, .45, .4, .4]])}
        snapshots = {}
        details = {}
        for label in policy.POLICIES:
            model = deepcopy(target)
            model.load_state_dict(states[label], strict=True)
            model.train()
            ema = ModelEMA(model)
            assert policy.state_hash(ema.ema.state_dict()) == policy.state_hash(states[label])
            optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=1e-5)
            loss, components = model(batch)
            assert torch.isfinite(loss) and bool(torch.isfinite(components).all())
            assert model.criterion.nc == 6
            loss.backward()
            gradients = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
            assert all(bool(torch.isfinite(g).all()) for g in gradients.values())
            optimizer.step()
            ema.update(model)
            parameter_state = {k: p.detach().clone() for k, p in model.named_parameters()}
            ema_parameters = {k: p.detach().clone() for k, p in ema.ema.named_parameters()}
            snapshots[label] = {"parameters": parameter_state, "gradients": gradients,
                                "ema_parameters": ema_parameters,
                                "buffers": {k: v.detach().clone() for k, v in model.named_buffers()},
                                "ema_buffers": {k: v.detach().clone() for k, v in ema.ema.named_buffers()},
                                "loss": loss.detach().clone()}
            details[label] = {"loss": float(loss.detach()), "components": components.tolist(),
                              "criterion_nc": model.criterion.nc,
                              "gradient_tensors": len(gradients),
                              "parameter_sha256_after_one_step": policy.state_hash(parameter_state),
                              "ema_parameter_sha256_after_one_update": policy.state_hash(ema_parameters),
                              "ema_updates": ema.updates}
            del model, ema, optimizer
        result["synthetic_real_detection_loss"] = details
        pairs = {}
        for left, right in (("A0R0", "A0R1"), ("A1R0", "A1R1")):
            first, second = snapshots[left], snapshots[right]
            checks = {field + "_bitwise_equal": all(torch.equal(v, second[field][k]) for k, v in first[field].items())
                      for field in ("parameters", "gradients", "ema_parameters")}
            checks["loss_bitwise_equal"] = torch.equal(first["loss"], second["loss"])
            assert all(checks.values())
            checks["bn_running_buffers_still_differ"] = any(
                not torch.equal(first["buffers"][path + ".running_mean"], second["buffers"][path + ".running_mean"])
                for path in record["affected_bn_paths"])
            checks["ema_bn_running_buffers_still_differ"] = any(
                not torch.equal(first["ema_buffers"][path + ".running_mean"], second["ema_buffers"][path + ".running_mean"])
                for path in record["affected_bn_paths"])
            assert checks["bn_running_buffers_still_differ"] and checks["ema_bn_running_buffers_still_differ"]
            pairs[left + "_vs_" + right] = checks
        result["paired_running_stat_null"] = pairs

        # Exercise the actual _setup_train method with synthetic metadata only.
        # Constructor, dataset loader and validator are deliberately replaced so
        # this cannot discover or read a real data split, including the test set.
        class SyntheticSetupTrainer(DetectionTrainer):
            def __init__(self, label):
                self.label = label
                self.args = SimpleNamespace(**DEFAULT_CFG_DICT)
                for key, value in {"amp": False, "resume": False, "freeze": 0,
                                   "optimizer": "SGD", "lr0": .001, "task": "detect",
                                   "imgsz": 64, "plots": False, "nbs": 2,
                                   "epochs": 1, "patience": 0, "close_mosaic": 0}.items():
                    setattr(self.args, key, value)
                self.device = torch.device("cpu")
                self.model = "synthetic-saved-initialization"
                self.data = {"nc": 6, "names": {i: "synthetic_" + str(i) for i in range(6)}}
                self.trainset, self.testset = "synthetic-train", "synthetic-val"
                self.batch_size, self.epochs, self.start_epoch = 2, 1, 0
                self.resume, self.ema = False, None
                self.loss_names = "box_loss", "cls_loss", "dfl_loss"
                self.events = []

            def get_model(self, cfg=None, weights=None, verbose=True):
                assert cfg == "synthetic-saved-initialization" and weights is None
                model = deepcopy(target)
                model.load_state_dict(states[self.label], strict=True)
                self.events.append({"event": "get_model_policy_installed",
                                    "state_sha256": policy.state_hash(model.state_dict())})
                return model

            def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
                assert dataset_path in ("synthetic-train", "synthetic-val")
                self.events.append({"event": "synthetic_dataloader", "mode": mode})
                return SimpleNamespace(dataset=range(2))

            def get_validator(self):
                return SimpleNamespace(metrics=SimpleNamespace(keys=["synthetic_metric"]))

            def run_callbacks(self, event):
                row = {"event": event}
                if isinstance(self.model, nn.Module):
                    row["state_sha256"] = policy.state_hash(self.model.state_dict())
                self.events.append(row)

        setup_checks = {}
        for label in policy.POLICIES:
            trainer = SyntheticSetupTrainer(label)
            trainer._setup_train(1)
            observed = policy.state_hash(trainer.model.state_dict())
            ema_observed = policy.state_hash(trainer.ema.ema.state_dict())
            assert observed == record["policies"][label]["state_sha256"] == ema_observed
            assert trainer.model.nc == trainer.model.model[-1].nc == 6
            assert trainer.model.args is trainer.args
            setup_checks[label] = {"state_sha256_at_setup_end": observed,
                                   "ema_state_sha256_at_setup_end": ema_observed,
                                   "head_nc": trainer.model.model[-1].nc,
                                   "amp": trainer.amp, "events": trainer.events}
            del trainer
        result["actual_trainer_setup_synthetic_metadata"] = setup_checks

        # Demonstrate the callback ordering trap without launching the epoch loop.
        mode_model = deepcopy(target)
        for module in mode_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()  # what an on_train_epoch_start FrozenBN callback does
        frozen_before = all(not m.training for m in mode_model.modules() if isinstance(m, nn.BatchNorm2d))
        mode_model.train()  # original trainer.py immediately follows the callback with this
        overwritten = all(m.training for m in mode_model.modules() if isinstance(m, nn.BatchNorm2d))
        assert frozen_before and overwritten
        result["epoch_start_bn_mode_overwrite"] = {"callback_freezes_bn": frozen_before,
                                                  "following_model_train_restores_bn_training": overwritten}
        del mode_model

        # Test checkpoint compatibility using a tiny object checkpoint, not a
        # detector checkpoint or any train/validation/test artifact.
        tiny_path = out / "tiny_object_checkpoint.pt"
        torch.save({"model": nn.Linear(1, 1), "train_args": {}}, tiny_path)
        try:
            strip_optimizer(tiny_path)
            result["original_strip_optimizer_compatibility"] = {"status": "succeeded"}
        except Exception as exc:
            result["original_strip_optimizer_compatibility"] = {
                "status": "failed", "exception": type(exc).__name__, "message": str(exc),
                "implication": "Original final_eval calls this function; default object checkpoint loading is incompatible in this environment."}
        assert not torch.cuda.is_initialized()
        result["cuda_initialized"] = torch.cuda.is_initialized()
        result["status"] = "passed"
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
    finally:
        result["repository_modules_imported"] = {
            name: {"file": str(Path(module.__file__).resolve()), "sha256": sha(module.__file__)}
            for name, module in sorted(sys.modules.items()) if name.startswith("ultralytics")
            and getattr(module, "__file__", None) and Path(module.__file__).suffix == ".py"
            and Path(module.__file__).is_file()}
        (out / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": result["status"], "output": str(out / "audit.json"),
                          "error": result.get("error"), "blocked_events": blocked}, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
