"""Build an audited, deliberately narrow shared-prefix initialization policy.

CPU construction/probes only. No training, data access, downloads, or framework load().
This policy approves model.0 only; downstream local-operator candidates are unapproved.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys

import torch
from torch import nn
import ultralytics
from ultralytics import YOLO
from ultralytics.nn.modules import Conv
from ultralytics.nn.tasks import BaseModel, DetectionModel
import ultralytics.nn.tasks as tasks

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "paper-try"))
from yolo_paper import register_modules

VARIANTS = ("base", "hwd-strip")
SOURCE_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
CONFIG_SHA256 = {
    "base": "5212ee3b9410c592437d32fab5a73b5fb1dfef68bb8b40eeaf4c7881a2bedaf3",
    "hwd-strip": "78264b1b436db2b287c9d880d88e8da7021cd51f3d0fb4c6da7973f4a4f331b7",
}
# These pins refer to locally reviewed source, not semantic proof from a hash alone.
CODE_SHA256 = {
    "Conv": "2805e7f091e90677407b0c05e255153fc74d44de9364666c2cc0eb9bf6aa8c10",
    "Conv2d": "566c7b4286b795bb035b52796f4582b4f1e71e7b901f794c375bbc60274be24d",
    "BatchNorm2d": "1294acc933b45db414e9757226a483fbef000abe51fcdba54ed805f078e500a4",
    "SiLU": "51c6f50019bf5cc42274e20dd2522174fa1ededf5672b5d1cca7b827a8aa44a1",
    "predict_once": "b932f70b581bec47a3546906f76677f11d1a4ce69f1f3d2e95a8a53ecd4b51d7",
    "initialize_weights": "e9d27bb15d2b6d56316ec27ac5692d1179f43cbb30c1fbff91aca86fe5a66adc",
}
FILE_SHA256 = {
    "tasks": "4dcf1e84f4dd87ec7a9d94b6365bb54fcf842c0e5267ab1bf02ccb31be197c18",
    "custom_modules": "20a975e50dd9c284ec63c72a15e08268f922b30af8b9cfd73a54bd53a6d64f22",
    "register": "e0a83a506b3707c3f9093668bfd58e48565120f6ff5e3c8410ae02507a57c3c4",
}
WHITE_KEYS = (
    "model.0.conv.weight", "model.0.bn.weight", "model.0.bn.bias",
    "model.0.bn.running_mean", "model.0.bn.running_var", "model.0.bn.num_batches_tracked",
)
WHITE_SHAPES = ((16, 3, 3, 3), (16,), (16,), (16,), (16,), ())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def json_hash(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return sha_bytes(str(value.dtype).encode() + str(tuple(value.shape)).encode() + value.numpy().tobytes())


def state_hashes(model):
    return {key: tensor_hash(value) for key, value in sorted(model.state_dict().items())}


def graph_signature(model):
    return [{"path": name, "type": type(module).__module__ + "." + type(module).__qualname__,
             "extra_repr": module.extra_repr(), "from": getattr(module, "f", None),
             "index": getattr(module, "i", None), "training": module.training,
             "parameters": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                                     "requires_grad": v.requires_grad}
                            for k, v in module.named_parameters(recurse=False)}}
            for name, module in model.named_modules()]


def assert_reviewed_environment():
    require(ultralytics.__version__ == "8.4.189", "Unreviewed Ultralytics version")
    require(torch.__version__ == "2.7.1+cu118", "Unreviewed PyTorch version")
    objects = {"Conv": Conv, "Conv2d": nn.Conv2d, "BatchNorm2d": nn.BatchNorm2d,
               "SiLU": nn.SiLU, "predict_once": BaseModel._predict_once,
               "initialize_weights": tasks.initialize_weights}
    for name, obj in objects.items():
        require(sha_bytes(inspect.getsource(obj).encode()) == CODE_SHA256[name], "Unreviewed code: " + name)
    files = {"tasks": Path(tasks.__file__), "custom_modules": BASE / "paper-try/yolo_paper/modules.py",
             "register": BASE / "paper-try/yolo_paper/register.py"}
    for name, path in files.items():
        require(sha_file(path) == FILE_SHA256[name], "Unreviewed file: " + name)
    for variant in VARIANTS:
        path = BASE / f"paper-try/configs/models/yolo11-{variant}.yaml"
        require(sha_file(path) == CONFIG_SHA256[variant], "Unreviewed configuration: " + variant)


def validate_stem(model, role):
    """Check the manually reviewed computation, graph entry, attributes and state schema."""
    require(role in (*VARIANTS, "source"), "Unknown model role")
    require(type(model) is DetectionModel, "Unexpected model type")
    require(model._predict_once.__func__ is BaseModel._predict_once, "Changed model execution")
    require("forward" not in model.__dict__, "Instance model forward override")
    stem = model.model[0]
    require(stem.i == 0 and stem.f == -1, "Stem is not the raw-input graph entry")
    require(type(stem) is Conv and set(stem._modules) == {"conv", "bn", "act"}, "Changed stem composition")
    require(stem.forward.__func__ is Conv.forward, "Changed stem forward")
    require(type(stem.conv) is nn.Conv2d and type(stem.bn) is nn.BatchNorm2d
            and type(stem.act) is nn.SiLU, "Changed leaf operators")
    for module in (model, model.model, *stem.modules()):
        require(not module._forward_hooks and not module._forward_pre_hooks, "Forward hooks require review")
        require("forward" not in module.__dict__, "Instance forward override requires review")
    c, b, a = stem.conv, stem.bn, stem.act
    require((c.in_channels, c.out_channels, c.kernel_size, c.stride, c.padding,
             c.dilation, c.groups, c.padding_mode, c.bias) ==
            (3, 16, (3, 3), (2, 2), (1, 1), (1, 1), 1, "zeros", None), "Changed Conv2d semantics")
    require((b.num_features, b.eps, b.momentum, b.affine, b.track_running_stats) ==
            (16, 0.001, 0.03, True, True), "Changed BatchNorm2d semantics")
    # Official checkpoint uses inplace=True; targets use False. Conv.forward feeds
    # SiLU only a fresh BN result with no second consumer. Probe values and gradients.
    require(a.inplace == (role == "source"), "Unreviewed SiLU inplace setting")
    expected = {key.removeprefix("model.0."): shape for key, shape in zip(WHITE_KEYS, WHITE_SHAPES)}
    require(set(stem.state_dict()) == set(expected), "Stem state schema changed")
    for key, value in stem.state_dict().items():
        require(tuple(value.shape) == expected[key], "Stem shape changed: " + key)
        require(value.device.type == "cpu", "This audit is CPU-only")
        require(value.dtype == (torch.int64 if key == "bn.num_batches_tracked" else torch.float32),
                "Stem dtype changed: " + key)
    return {"root": "model.0", "input": "same caller-provided RGB NCHW tensor; input preprocessing is external",
            "operation": "Conv2d(3,16,3,2,1,bias=False) -> BN(eps=.001,momentum=.03) -> SiLU",
            "input_graph_from": -1, "activation_inplace": a.inplace,
            "output_channels": 16, "output_spatial": "ceil(H/2), ceil(W/2)"}


def construct_targets(seed):
    assert_reviewed_environment()
    register_modules()
    models = {}
    for variant in VARIANTS:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            models[variant] = DetectionModel(
                str(BASE / f"paper-try/configs/models/yolo11n-{variant}.yaml"), nc=6, verbose=False).float().cpu()
        validate_stem(models[variant], variant)
    return models


def load_source():
    weight = BASE / "weights/yolo11n_coco_official.pt"
    require(sha_file(weight) == SOURCE_SHA256, "Source checkpoint SHA-256 changed")
    # Local trusted checkpoint only; this operation must not perturb scratch RNG.
    with torch.random.fork_rng(devices=[]):
        model = YOLO(str(weight)).model.float().cpu()
    validate_stem(model, "source")
    return model


def copy_white(target, source, target_role, source_role):
    """Copy exactly the reviewed keys, with bitwise preservation checks for every other key."""
    validate_stem(target, target_role)
    validate_stem(source, source_role)
    before = state_hashes(target)
    graph_before = json_hash(graph_signature(target))
    source_state, target_state = source.state_dict(), target.state_dict()
    with torch.no_grad():
        for key in WHITE_KEYS:
            require(source_state[key].shape == target_state[key].shape, "Whitelisted shape changed: " + key)
            target_state[key].copy_(source_state[key])
    after = state_hashes(target)
    require(graph_before == json_hash(graph_signature(target)), "Copy changed architecture/trainability/mode")
    for key in before:
        expected = tensor_hash(source_state[key]) if key in WHITE_KEYS else before[key]
        require(after[key] == expected, "Copy/preservation verification failed: " + key)
    return before, after


def probe_prefix(base, variant, source):
    """Synthetic stem probes strengthen code review; they are not a full-model equivalence claim."""
    results = []
    for training in (False, True):
        for shape in ((1, 3, 32, 32), (2, 3, 31, 35), (1, 3, 64, 48)):
            gen = torch.Generator().manual_seed(751 + int(training))
            value = torch.randn(shape, generator=gen)
            modules = [copy.deepcopy(m.model[0]).train(training) for m in (base, variant, source)]
            for module in modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
            outputs, grads, input_grads, buffers = [], [], [], []
            for module in modules:
                x = value.clone().requires_grad_(True)
                output = module(x)
                output.square().mean().backward()
                require(torch.equal(x.detach(), value), "Stem mutated its caller input")
                outputs.append(output.detach())
                input_grads.append(x.grad)
                grads.append({k: p.grad for k, p in module.named_parameters()})
                buffers.append({k: v.clone() for k, v in module.named_buffers()})
            for index in (1, 2):
                require(torch.equal(outputs[0], outputs[index]), "Stem forward mismatch")
                require(torch.equal(input_grads[0], input_grads[index]), "Stem input gradient mismatch")
                require(all(torch.equal(grads[0][k], grads[index][k]) for k in grads[0]), "Stem parameter gradient mismatch")
                require(all(torch.equal(buffers[0][k], buffers[index][k]) for k in buffers[0]), "Stem BN update mismatch")
            results.append({"training": training, "input_shape": list(shape), "output_shape": list(outputs[0].shape),
                            "forward_gradient_and_buffer_bitwise_equal": True})
    return results


def audit_rows(model, scratch_hashes, source, policy):
    params = dict(model.named_parameters())
    src, state = source.state_dict(), model.state_dict()
    rows = []
    for key, value in sorted(state.items()):
        copied = policy == "C_strict_prefix" and key in WHITE_KEYS
        reason = ("reviewed identical raw-input prefix" if copied else
                  "paired scratch initialization" if policy == "S_paired_prefix" else
                  "outside reviewed prefix; upstream computations diverge from layer 1")
        rows.append({"key": key, "kind": "parameter" if key in params else "buffer", "numel": value.numel(),
                     "target_shape": list(value.shape), "source_shape": list(src[key].shape) if key in src else None,
                     "source_sha256": tensor_hash(src[key]) if key in src else None,
                     "scratch_sha256": scratch_hashes[key], "after_sha256": tensor_hash(value),
                     "action": "copied" if copied else "retained", "reason": reason})
    return rows


def build(seed=17):
    torch.set_num_threads(1)
    targets = construct_targets(seed)
    pairing_before = state_hashes(targets["hwd-strip"])
    copy_white(targets["hwd-strip"], targets["base"], "hwd-strip", "base")
    scratch_hashes = {name: state_hashes(model) for name, model in targets.items()}
    source = load_source()
    result = {
        "status": "exploratory implementation verified; no training or dataset inference",
        "policy_id": "C_strict_prefix_v1", "seed": seed, "nc": 6, "device": "cpu",
        "scope": "only raw-input common prefix model.0; not a maximal shared-module policy",
        "source_weight_sha256": SOURCE_SHA256, "config_sha256": CONFIG_SHA256,
        "reviewed_code_sha256": CODE_SHA256, "reviewed_file_sha256": FILE_SHA256,
        "script_sha256": sha_file(__file__), "versions": {"torch": torch.__version__, "ultralytics": ultralytics.__version__},
        "whitelist": [{"source_key": k, "target_key": k, "shape": list(s)} for k, s in zip(WHITE_KEYS, WHITE_SHAPES)],
        "source_prefix": validate_stem(source, "source"), "targets": {},
        "pairing": {"source": "base scratch prefix", "target": "hwd-strip scratch prefix",
                    "explicit_copy": True, "before_sha256": pairing_before,
                    "after_sha256": scratch_hashes["hwd-strip"]},
    }
    transferred = {}
    for name, model in targets.items():
        clone = copy.deepcopy(model)
        before, after = copy_white(clone, source, name, "source")
        require(before == scratch_hashes[name], "Policy clone did not preserve S state")
        require(all(after[k] == before[k] for k in before if k not in WHITE_KEYS), "Nonwhite state changed")
        parameters = dict(model.named_parameters())
        white_p = sum(model.state_dict()[k].numel() for k in WHITE_KEYS if k in parameters)
        white_b = sum(model.state_dict()[k].numel() for k in WHITE_KEYS if k not in parameters)
        total = sum(p.numel() for p in model.parameters())
        result["targets"][name] = {
            "prefix": validate_stem(model, name), "total_parameter_elements": total,
            "copied_parameter_tensors": 3, "copied_parameter_elements": white_p,
            "copied_parameter_fraction": white_p / total,
            "copied_buffer_tensors": 3, "copied_buffer_elements": white_b,
            "scratch_state_sha256": json_hash(scratch_hashes[name]),
            "C_state_sha256": json_hash(after), "graph_signature_sha256": json_hash(graph_signature(model)),
            "policies": {"S_paired_prefix": audit_rows(model, scratch_hashes[name], source, "S_paired_prefix"),
                         "C_strict_prefix": audit_rows(clone, scratch_hashes[name], source, "C_strict_prefix")}}
        transferred[name] = clone
    result["synthetic_prefix_checks"] = probe_prefix(transferred["base"], transferred["hwd-strip"], source)
    result["limitations"] = [
        "464 parameter elements and 33 buffer elements only; do not interpret C as all common semantics.",
        "S matches only the reviewed prefix explicitly; no claim of paired new-module parameters.",
        "Synthetic probes are finite checks supported by source/graph review, not general semantic proof.",
        "No U/D policy implementation, trainer integration, pilot, or performance conclusion.",
        "Input preprocessing and training pipeline must be audited separately before actual training.",
    ]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    require(args.output.resolve().drive.upper() == "D:", "Output must be on D drive")
    require(not args.output.exists(), "Refusing overwrite: " + str(args.output))
    result = build(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "sha256": sha_file(args.output),
                      "parameters": {k: {a: v[a] for a in ("total_parameter_elements", "copied_parameter_elements", "copied_parameter_fraction")}
                                     for k, v in result["targets"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
