"""Offline CPU diagnostic adapter for fixed six-class MSRDet BN policies.

Only short diagnostics are supported: no GPU, resume, augmentation, external
callbacks, automatic downloads, test split, or paper-scale training. The
synthetic check creates its own train/val images under a NEW D: output directory.
The original model, detection loss, EMA, optimizer step and trainer epoch loop
are used; data loading, validation copies and state saving are isolated here.
"""
from pathlib import Path
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
import random
import sys
import traceback

BASE = Path(__file__).resolve().parent
REPO = BASE / "external/MSRDet-7267acc2b1fb96d3af2dfe2e406d0f430e6abfc5"
WEIGHT_SHA = "31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95"
FORBIDDEN_PARTS = {"test", "testing", "heldout", "holdout", "testset", "test_set"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure_data_path(path, root=None):
    path = Path(path).resolve()
    if path.drive.upper() != "D:" or any(p.lower() in FORBIDDEN_PARTS for p in path.parts):
        raise ValueError("Only non-test D: data paths are accepted: " + str(path))
    if root is not None and not path.is_relative_to(root):
        raise ValueError("Data path escapes explicit root: " + str(path))
    if not path.is_file():
        raise ValueError("Explicit data file is missing: " + str(path))
    return path


def read_data_spec(path, yaml, max_images=32):
    """Read explicit lists only; never discover directories or fallback splits."""
    path = ensure_data_path(path)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or set(spec) - {"path", "train", "val", "names", "nc"}:
        raise ValueError("Data YAML permits only path/train/val/names/nc; test/download are rejected")
    if not all(spec.get(k) for k in ("path", "train", "val", "names")):
        raise ValueError("Explicit nonempty path/train/val/names are required")
    root = Path(spec["path"])
    if not root.is_absolute() or root.resolve().drive.upper() != "D:":
        raise ValueError("Data root must be absolute on D:")
    root = root.resolve()
    if any(p.lower() in FORBIDDEN_PARTS for p in root.parts):
        raise ValueError("A test-like root is not permitted")
    names = spec["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or set(names) != set(range(6)) or spec.get("nc", 6) != 6:
        raise ValueError("Exactly six names with class ids 0..5 are required")
    if any(not isinstance(name, str) or not name for name in names.values()) or len(set(names.values())) != 6:
        raise ValueError("Six nonempty unique class names are required")
    result = {"path": str(root), "nc": 6, "names": names}
    manifests, image_paths = {}, {}
    for split in ("train", "val"):
        entry = Path(spec[split])
        entry = ensure_data_path(entry if entry.is_absolute() else root / entry, root)
        if entry.suffix.lower() != ".txt":
            raise ValueError("Only explicit .txt image manifests are supported")
        lines = [line.strip() for line in entry.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not 1 <= len(lines) <= max_images:
            raise ValueError("Short diagnostic requires 1..32 images per split")
        rows = []
        for line in lines:
            image_path = Path(line)
            image_path = ensure_data_path(image_path if image_path.is_absolute() else root / image_path, root)
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                raise ValueError("Unsupported image extension")
            parts = list(image_path.parts)
            if parts.count("images") != 1:
                raise ValueError("Expected exactly one images component for explicit YOLO labels")
            parts[parts.index("images")] = "labels"
            label_path = ensure_data_path(Path(*parts).with_suffix(".txt"), root)
            boxes = []
            for line_number, label_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                if not label_line.strip():
                    continue
                values = [float(value) for value in label_line.split()]
                if len(values) != 5 or values[0] not in range(6):
                    raise ValueError("Detection labels need class and xywh: " + str(label_path))
                import math
                if not all(math.isfinite(value) for value in values) or not all(0 <= v <= 1 for v in values[1:]):
                    raise ValueError("Nonfinite or out-of-range detection label")
                if values[3] <= 0 or values[4] <= 0:
                    raise ValueError("Positive box width/height required")
                boxes.append(values)
            rows.append({"image": str(image_path), "label": str(label_path), "boxes": boxes,
                         "image_sha256": sha(image_path), "label_sha256": sha(label_path)})
        canonical = [row["image"].lower() for row in rows]
        if len(set(canonical)) != len(canonical):
            raise ValueError("Duplicate image path in " + split)
        manifests[split] = rows
        image_paths[split] = set(canonical)
        result[split] = str(entry)
    if image_paths["train"] & image_paths["val"]:
        raise ValueError("Train/val image paths overlap")
    if {row["image_sha256"] for row in manifests["train"]} & {row["image_sha256"] for row in manifests["val"]}:
        raise ValueError("Train/val image bytes overlap")
    return result, manifests


def create_synthetic_data(out, yaml, np, Image):
    root = out / "synthetic_data"
    root.mkdir()
    rng = np.random.default_rng(20261104)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        paths = []
        for index in range(2):
            image_path = root / "images" / split / (str(index) + ".png")
            pixels = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(image_path)
            (root / "labels" / split / (str(index) + ".txt")).write_text(
                str(index * 5) + " 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            paths.append(str(image_path))
        (root / (split + ".txt")).write_text("\n".join(paths) + "\n", encoding="utf-8")
    data_path = root / "development.yaml"
    data_path.write_text(yaml.safe_dump({"path": str(root), "train": "train.txt", "val": "val.txt",
                                        "nc": 6, "names": ["synthetic_" + str(i) for i in range(6)]}), encoding="utf-8")
    return data_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--init-dir", required=True, type=Path)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic-check", action="store_true")
    group.add_argument("--data", type=Path)
    ap.add_argument("--policy", choices=("A0R0", "A1R0", "A0R1", "A1R1"), default="A0R0")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--imgsz", type=int, choices=(64, 128), default=64)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()
    out = args.output_dir.resolve()
    if out.drive.upper() != "D:" or out.exists():
        raise SystemExit("A new D: output directory is required")
    if not 1 <= args.epochs <= 2 or not 1 <= args.batch <= 4:
        raise SystemExit("Only CPU diagnostics of 1..2 epochs and batch 1..4 are supported")
    if args.synthetic_check and (args.epochs != 1 or args.imgsz != 64 or args.batch != 2):
        raise SystemExit("Synthetic acceptance check is fixed to one epoch, 64px, batch=2")
    if any(not os.environ.get(k, "").upper().startswith("D:") for k in ("TEMP", "TMP", "TORCH_HOME", "YOLO_CONFIG_DIR")):
        raise SystemExit("Dot-source project-env.ps1 before starting Python")
    if "ultralytics" in sys.modules or any(os.environ.get(k, "-1") not in ("-1", "") for k in ("RANK", "LOCAL_RANK")):
        raise SystemExit("A fresh non-DDP Python process is required")
    if sha(args.weights) != WEIGHT_SHA:
        raise SystemExit("Unexpected real source checkpoint")
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", YOLO_AUTOINSTALL="false", YOLO_VERBOSE="false",
                      PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    sys.dont_write_bytecode = True
    for key, leaf in {"TEMP": "temp", "TMP": "temp", "TMPDIR": "temp", "TORCH_HOME": "torch",
                      "MPLCONFIGDIR": "matplotlib", "YOLO_CONFIG_DIR": "ultralytics",
                      "XDG_CACHE_HOME": "cache", "HF_HOME": "huggingface", "PIP_CACHE_DIR": "pip"}.items():
        path = out / "isolated_runtime" / leaf
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    blocked = []

    def audit(event, values):
        if event in ("socket.connect", "socket.bind", "socket.sendto", "socket.getaddrinfo",
                     "subprocess.Popen", "os.system", "os.posix_spawn"):
            blocked.append({"event": event, "target": repr(values)[:300]})
            raise OSError("Offline diagnostic blocks " + event)
        paths = []
        if event == "open" and isinstance(values[0], (str, bytes)):
            path, mode, flags = values
            if (isinstance(mode, str) and any(c in mode for c in "wax+")) or flags & (
                    os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                paths.append(path)
        elif event in ("os.mkdir", "os.remove", "os.rmdir"):
            paths.append(values[0])
        elif event == "os.rename":
            paths.extend(values[:2])
        for path in paths:
            resolved = Path(os.fsdecode(path)).resolve()
            if str(resolved).upper().startswith("\\\\.\\NUL"):
                continue  # Windows null device does not write a disk.
            if not resolved.is_relative_to(out):
                raise OSError("Diagnostic write outside its new output directory: " + str(resolved))

    sys.addaudithook(audit)
    report = {"status": "started", "scope": "short offline CPU diagnostic adapter; not a paper training protocol",
              "synthetic": args.synthetic_check, "script_sha256": sha(__file__), "blocked_events": blocked,
              "limitations": ["CPU only, no resume, no augmentation or calibration, no GPU/AMP/DDP acceptance",
                              "Explicit image/label manifests and deterministic RGB letterbox preprocessing only",
                              "Synthetic metrics are mechanism checks, never evidence of detection performance"]}
    try:
        import torch
        import torch.nn as nn
        import numpy as np
        import yaml
        from PIL import Image
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        import audit_msrdet_structure as structure
        import audit_msrdet_bn_policy as policy
        for relative, expected in structure.PINNED.items():
            assert sha(REPO / relative) == expected, relative
        sys.path.insert(0, str(REPO))
        import ultralytics
        from ultralytics.cfg import get_cfg
        from ultralytics.models.yolo.detect.train import DetectionTrainer
        from ultralytics.models.yolo.detect.val import DetectionValidator
        from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight
        from ultralytics.utils import DEFAULT_CFG, RANK
        from types import SimpleNamespace
        assert Path(ultralytics.__file__).resolve() == REPO / "ultralytics/__init__.py"
        assert RANK == -1

        def clone_cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {key: clone_cpu(child) for key, child in value.items()}
            if isinstance(value, tuple):
                return tuple(clone_cpu(child) for child in value)
            if isinstance(value, list):
                return [clone_cpu(child) for child in value]
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            raise TypeError("Non-tensor/primitive checkpoint value: " + type(value).__name__)

        def atomic_tensor_save(value, path):
            if path.exists():
                raise FileExistsError("Refusing checkpoint overwrite: " + str(path))
            pending = path.with_name(path.name + ".pending")
            with pending.open("xb") as handle:
                torch.save(value, handle, use_dill=False)
                handle.flush()
                os.fsync(handle.fileno())
            loaded = torch.load(pending, map_location="cpu", weights_only=True)
            os.replace(pending, path)
            return loaded

        def fp32_state(model):
            return {key: value.detach().cpu().float().clone() if value.is_floating_point()
                    else value.detach().cpu().clone() for key, value in model.state_dict().items()}

        def capture_rng(generators):
            state = np.random.get_state()
            return {"python": random.getstate(), "numpy": {"algorithm": state[0],
                    "keys": torch.tensor(state[1].astype(np.int64)), "position": state[2],
                    "has_gauss": state[3], "cached_gauss": state[4]}, "torch_cpu": torch.get_rng_state(),
                    "loaders": {key: generator.get_state() for key, generator in generators.items()}}

        def restore_rng(state, generators):
            random.setstate(state["python"])
            values = state["numpy"]
            np.random.set_state((values["algorithm"], values["keys"].numpy().astype(np.uint32),
                                 values["position"], values["has_gauss"], values["cached_gauss"]))
            torch.set_rng_state(state["torch_cpu"])
            for key, generator in generators.items():
                generator.set_state(state["loaders"][key])

        init_audit = json.loads((args.init_dir / "audit.json").read_text(encoding="utf-8"))
        initial_path = args.init_dir / "target_initial_state.pt"
        assert sha(initial_path) == init_audit["target_initial_file_sha256"]
        initial = torch.load(initial_path, map_location="cpu", weights_only=True)
        assert policy.state_hash(initial) == init_audit["target_initial_state_sha256"]
        source, _ = attempt_load_one_weight(str(args.weights.resolve()), device="cpu", fuse=False)
        assert policy.state_hash(source.state_dict()) == init_audit["source_state_sha256"]
        source_state = source.state_dict()
        cfg = yaml.safe_load((REPO / "ultralytics/cfg/models/v8/MSRDet.yaml").read_text(encoding="utf-8"))
        cfg["scale"], cfg["nc"] = "n", 6
        data_path = create_synthetic_data(out, yaml, np, Image) if args.synthetic_check else args.data
        data, manifests = read_data_spec(data_path, yaml)
        if len(manifests["train"]) < args.batch or len(manifests["train"]) % args.batch:
            raise ValueError("Diagnostic requires complete train batches without singleton tails")
        report["input_data_spec_sha256"] = sha(data_path)
        report["frozen_manifests"] = manifests
        report["preprocessing"] = "RGB, aspect-preserving centered letterbox with bilinear resize; fill=114; no augmentation"

        class FrozenManifestDataset(torch.utils.data.Dataset):
            def __init__(self, rows):
                self.rows = rows

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                row = self.rows[index]
                if sha(row["image"]) != row["image_sha256"] or sha(row["label"]) != row["label_sha256"]:
                    raise RuntimeError("Frozen image/label changed")
                with Image.open(row["image"]) as opened:
                    image = opened.convert("RGB")
                width, height = image.size
                gain = min(args.imgsz / height, args.imgsz / width)
                new_width, new_height = round(width * gain), round(height * gain)
                left, top = (args.imgsz - new_width) // 2, (args.imgsz - new_height) // 2
                canvas = Image.new("RGB", (args.imgsz, args.imgsz), (114, 114, 114))
                canvas.paste(image.resize((new_width, new_height), Image.Resampling.BILINEAR), (left, top))
                values = torch.tensor(row["boxes"], dtype=torch.float32).reshape(-1, 5)
                boxes = values[:, 1:].clone()
                boxes[:, 0] = (boxes[:, 0] * width * gain + left) / args.imgsz
                boxes[:, 1] = (boxes[:, 1] * height * gain + top) / args.imgsz
                boxes[:, 2] *= width * gain / args.imgsz
                boxes[:, 3] *= height * gain / args.imgsz
                return {"img": torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1),
                        "cls": values[:, :1], "bboxes": boxes, "batch_idx": torch.zeros(len(values)),
                        "im_file": row["image"], "ori_shape": (height, width),
                        "resized_shape": (args.imgsz, args.imgsz), "ratio_pad": ((gain, gain), (left, top))}

            @staticmethod
            def collate_fn(rows):
                batch = {key: tuple(row[key] for row in rows) for key in rows[0]}
                batch["img"] = torch.stack(batch["img"])
                for key in ("cls", "bboxes"):
                    batch[key] = torch.cat(batch[key])
                batch["batch_idx"] = torch.cat([value + index for index, value in enumerate(batch["batch_idx"])])
                return batch

        class OfflineCPUTrainer(DetectionTrainer):
            """Deliberately bypass BaseTrainer.__init__ auto-discovery/integrations."""
            def __init__(self, label):
                self.label = label
                self.save_dir = out / label
                self.save_dir.mkdir()
                self.wdir = self.save_dir / "weights"
                self.wdir.mkdir()
                overrides = {"task": "detect", "mode": "train", "model": str(REPO / "ultralytics/cfg/models/v8/MSRDet.yaml"),
                             "data": str(data_path), "device": "cpu", "workers": 0, "epochs": args.epochs,
                             "batch": args.batch, "imgsz": args.imgsz, "nbs": args.batch, "optimizer": "SGD",
                             "lr0": .001, "lrf": 1., "momentum": .9, "weight_decay": 0., "warmup_epochs": 0.,
                             "amp": False, "plots": False, "save": True, "val": False, "patience": 0,
                             "time": None, "close_mosaic": 0, "deterministic": True, "cache": False,
                             "multi_scale": False, "resume": False, "freeze": 0, "save_period": -1,
                             "seed": init_audit["seed"], "save_json": False, "save_txt": False, "split": "val"}
                self.args = get_cfg(DEFAULT_CFG, overrides)
                self.args.save_dir = str(self.save_dir)
                self.device = torch.device("cpu")
                self.validator, self.metrics, self.plots = None, None, {}
                self.last, self.best = self.wdir / "last_UNSUPPORTED.pt", self.wdir / "best_UNSUPPORTED.pt"
                self.save_period = -1
                self.batch_size, self.epochs, self.start_epoch = args.batch, args.epochs, 0
                self.model = self.args.model
                self.data = deepcopy(data)
                self.trainset, self.testset = data["train"], data["val"]  # inherited variable name means val only
                self.ema, self.lf, self.scheduler = None, None, None
                self.best_fitness, self.fitness, self.loss, self.tloss = None, None, None, None
                self.loss_names = "box_loss", "cls_loss", "dfl_loss"
                self.csv = self.save_dir / "results.csv"
                self.plot_idx = []
                self.resume = False
                self.callbacks = defaultdict(list)  # no external or even base integration callbacks
                self.generators, self.batches, self.saves, self.validation = {}, [], [], []
                self.optimizer_updates, self.bn_training_forwards = 0, 0
                self.add_callback("on_pretrain_routine_end", lambda trainer: trainer.check_setup())
                self.add_callback("on_train_start", lambda trainer: trainer.reset_training_rng())
                self.add_callback("on_train_batch_end", lambda trainer: trainer.check_batch_end())

            def get_model(self, cfg=None, weights=None, verbose=True):
                if weights is not None or self.resume:
                    raise RuntimeError("Resume/pretrained reload is unsupported")
                with torch.no_grad(), torch.device("cpu"):
                    model = DetectionModel(deepcopy(globals_cfg), nc=6, verbose=False)
                affine, stats = policy.POLICIES[self.label]
                state, _ = policy.policy_state(source_state, initial, init_audit["affected_bn_paths"], affine, stats)
                model.load_state_dict(state, strict=True)
                assert policy.state_hash(model.state_dict()) == init_audit["policies"][self.label]["state_sha256"]
                return model

            def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
                if mode not in ("train", "val") or dataset_path != self.data[mode] or rank not in (-1, 0):
                    raise ValueError("Only explicit train/val loaders are supported")
                generator = torch.Generator().manual_seed(init_audit["seed"] + (0 if mode == "train" else 1))
                self.generators[mode] = generator
                dataset = FrozenManifestDataset(manifests[mode])
                return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=mode == "train",
                                                   num_workers=0, pin_memory=False, drop_last=False,
                                                   collate_fn=dataset.collate_fn, generator=generator)

            def get_validator(self):
                return DetectionValidator(self.test_loader, save_dir=self.save_dir, args=deepcopy(self.args),
                                          _callbacks=defaultdict(list))

            def check_setup(self):
                expected = init_audit["policies"][self.label]["state_sha256"]
                self.setup_hashes = {"raw": policy.state_hash(self.model.state_dict()),
                                     "ema": policy.state_hash(self.ema.ema.state_dict())}
                assert set(self.setup_hashes.values()) == {expected}
                assert self.model.nc == self.model.model[-1].nc == 6
                assert not self.amp and not self.resume and self.device.type == "cpu"
                assert all(value.dtype == torch.float32 for value in self.model.state_dict().values() if value.is_floating_point())

            def reset_training_rng(self):
                random.seed(init_audit["seed"])
                np.random.seed(init_audit["seed"])
                torch.manual_seed(init_audit["seed"])
                self.initial_training_rng = capture_rng(self.generators)

            def preprocess_batch(self, batch):
                assert all(module.training for module in self.model.modules() if isinstance(module, nn.BatchNorm2d))
                self.bn_training_forwards += 1
                self.batches.append({"epoch": self.epoch, "paths": list(batch["im_file"]),
                                     "input_uint8_sha256": hashlib.sha256(batch["img"].numpy().tobytes()).hexdigest()})
                return super().preprocess_batch(batch)

            def optimizer_step(self):
                assert all(torch.isfinite(param.grad).all() for param in self.model.parameters() if param.grad is not None)
                self.last_gradient_hash = policy.state_hash({key: param.grad for key, param in self.model.named_parameters() if param.grad is not None})
                super().optimizer_step()
                self.optimizer_updates += 1

            def check_batch_end(self):
                assert torch.isfinite(self.loss) and bool(torch.isfinite(self.loss_items).all())

            def validate(self):
                before = {"raw": policy.state_hash(self.model.state_dict()),
                          "ema": policy.state_hash(self.ema.ema.state_dict())}
                rng = capture_rng(self.generators)
                readings = {}
                try:
                    for label, original in (("raw", self.model), ("ema", self.ema.ema)):
                        copied = deepcopy(original).cpu().float()
                        validator = self.get_validator()
                        proxy = SimpleNamespace(device=self.device, data=self.data,
                                                ema=SimpleNamespace(ema=copied), model=copied,
                                                loss_items=self.loss_items, stopper=self.stopper,
                                                epoch=self.epoch, epochs=self.epochs,
                                                label_loss_items=self.label_loss_items)
                        readings[label] = validator(proxy)
                        assert not validator.args.half
                        del copied, proxy, validator
                finally:
                    restore_rng(rng, self.generators)
                after = {"raw": policy.state_hash(self.model.state_dict()),
                         "ema": policy.state_hash(self.ema.ema.state_dict())}
                assert before == after
                fitness = float(readings["ema"].get("fitness", 0.))
                self.validation.append({"epoch": self.epoch, "readings": readings,
                                        "source_state_unchanged": before == after, "precision": "FP32 CPU copy"})
                metrics = {label + "/" + key: value for label, reading in readings.items() for key, value in reading.items()}
                return metrics, fitness

            def save_model(self):
                epoch = self.epoch + 1
                raw, ema = fp32_state(self.model), fp32_state(self.ema.ema)
                files = {}
                for label, state in (("raw", raw), ("ema", ema)):
                    path = self.wdir / ("epoch_%04d_%s_fp32.pt" % (epoch, label))
                    loaded = atomic_tensor_save(state, path)
                    assert policy.state_hash(loaded) == policy.state_hash(state)
                    assert all(value.dtype == torch.float32 for value in loaded.values() if value.is_floating_point())
                    files[label] = {"path": str(path), "file_sha256": sha(path), "state_sha256": policy.state_hash(loaded)}
                control = {"format": "offline-msrdet-cpu-diagnostic-v1", "epoch": epoch,
                           "policy": self.label, "optimizer_updates": self.optimizer_updates,
                           "bn_training_forwards": self.bn_training_forwards, "ema_updates": self.ema.updates,
                           "optimizer": clone_cpu(self.optimizer.state_dict()),
                           "scheduler": clone_cpu(self.scheduler.state_dict()),
                           "rng": capture_rng(self.generators), "initial_training_rng": self.initial_training_rng,
                           "source_checkpoint_sha256": WEIGHT_SHA,
                           "initial_state_sha256": init_audit["target_initial_state_sha256"]}
                path = self.wdir / ("epoch_%04d_control.pt" % epoch)
                atomic_tensor_save(control, path)
                files["control"] = {"path": str(path), "file_sha256": sha(path)}
                self.saves.append({"epoch": epoch, "files": files})

            def final_eval(self):
                # Validation already used FP32 copies at the fixed final epoch.
                # Never strip, replace raw with EMA, reload best, or overwrite state.
                assert self.saves and self.validation
                self.final_eval_avoided_strip = True

        globals_cfg = cfg
        labels = list(policy.POLICIES) if args.synthetic_check else [args.policy]
        report["runs"] = {}
        for label in labels:
            trainer = OfflineCPUTrainer(label)
            trainer.train()
            assert not torch.cuda.is_initialized()
            raw_parameters = {key: value.detach() for key, value in trainer.model.named_parameters()}
            ema_parameters = {key: value.detach() for key, value in trainer.ema.ema.named_parameters()}
            report["runs"][label] = {"setup_hashes": trainer.setup_hashes,
                                     "parameters_sha256": policy.state_hash(raw_parameters),
                                     "ema_parameters_sha256": policy.state_hash(ema_parameters),
                                     "last_gradient_sha256": trainer.last_gradient_hash,
                                     "loss": float(trainer.loss.detach()), "batches": trainer.batches,
                                     "optimizer_updates": trainer.optimizer_updates,
                                     "bn_training_forwards": trainer.bn_training_forwards,
                                     "ema_updates": trainer.ema.updates,
                                     "validation": trainer.validation, "checkpoints": trainer.saves,
                                     "final_eval_avoided_strip": trainer.final_eval_avoided_strip}
            del trainer
        if args.synthetic_check:
            report["paired_null_checks"] = {}
            for left, right in (("A0R0", "A0R1"), ("A1R0", "A1R1")):
                first, second = report["runs"][left], report["runs"][right]
                checks = {key: first[key] == second[key] for key in ("parameters_sha256", "ema_parameters_sha256",
                                                                    "last_gradient_sha256", "loss", "batches")}
                assert all(checks.values()), (left, right, checks)
                report["paired_null_checks"][left + "_vs_" + right] = checks
            # Ensure the strict spec rejects a test key before resolving any of its paths.
            invalid = yaml.safe_load(data_path.read_text(encoding="utf-8"))
            invalid["test"] = "NEVER_OPEN_THIS_PATH"
            bad_path = out / "rejected_spec.yaml"
            bad_path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
            try:
                read_data_spec(bad_path, yaml)
                raise AssertionError("Test key was accepted")
            except ValueError as exc:
                report["test_key_rejected_before_path_access"] = "test/download" in str(exc)
                assert report["test_key_rejected_before_path_access"]
        report["cuda_initialized"] = torch.cuda.is_initialized()
        report["package"] = {"file": str(ultralytics.__file__), "version": ultralytics.__version__,
                              "torch": torch.__version__, "numpy": np.__version__}
        report["repository_module_hashes"] = {
            name: sha(module.__file__) for name, module in sorted(sys.modules.items())
            if name.startswith("ultralytics") and getattr(module, "__file__", None)
            and Path(module.__file__).suffix == ".py" and Path(module.__file__).is_file()}
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
    finally:
        (out / "adapter_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": report["status"], "output": str(out / "adapter_audit.json"),
                          "error": report.get("error")}, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
