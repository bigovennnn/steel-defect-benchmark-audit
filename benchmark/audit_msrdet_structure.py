"""CPU structure/load audit of pinned MSRDet source; no package import or checkpoint.

Only the allowlisted, previously reviewed definitions are compiled verbatim from
the pinned files. Package initializers, imports, training, export, profiling and
evaluation paths are not executed. Inactive parser class names are unique guards
which fail if constructed. This is an isolated source reconstruction, not a full
installation test or a claim about the authors' training/loading procedure.
"""
from pathlib import Path
import argparse
import ast
import contextlib
from copy import deepcopy
import hashlib
import json
import logging
import math
import os
import sys
import traceback

BASE = Path(__file__).resolve().parent
REPO = BASE / "external/MSRDet-7267acc2b1fb96d3af2dfe2e406d0f430e6abfc5"
PINNED = {
    "ultralytics/nn/modules/conv.py": "b69ebd11a3c0bde3ded42873b3dd946c76ef44209abe8c07cb38cbb0f8dfb15e",
    "ultralytics/nn/modules/block.py": "a385ba1f06b9cb2507114479a57d04e88397f5865a0afceba02da5d42c490a45",
    "ultralytics/nn/modules/head.py": "2e89d5d9bed3acbc7ecca8474367c2a4a83b05f0beb5406d3e54b934d71f4ffc",
    "ultralytics/nn/tasks.py": "2b9e1b24311d91af78b89eab07c4f4004c78e5b31acde9515eaa5cd634a954b1",
    "ultralytics/utils/torch_utils.py": "77e0aeb78edbcb0cc6567d09e976d16e1b6f5844bc95dc1c43a4986bbc4bca91",
    "ultralytics/cfg/models/v8/yolov8.yaml": "56349efd5d869fd66926bc13b73d00ebfe88329e94bae80d891eda112851e6b7",
    "ultralytics/cfg/models/v8/MSRDet.yaml": "37d52a2aa691477c49796aac9abd9cb43bd98a8fba29b049d1420e6073d3d8db",
}
SELECTED = {
    "ultralytics/nn/modules/conv.py": ["autopad", "Conv", "Focus", "Concat", "h_sigmoid"],
    "ultralytics/nn/modules/block.py": ["DFL", "SPPF", "GatedSPPF", "C2f", "Bottleneck1", "C2fk", "Bottleneck", "HTEMBranch", "HTEM", "Fusion_2in_mod", "FEM"],
    "ultralytics/nn/modules/head.py": ["Detect"],
    "ultralytics/utils/torch_utils.py": ["initialize_weights", "make_divisible", "intersect_dicts"],
    "ultralytics/nn/tasks.py": ["BaseModel", "DetectionModel", "parse_model"],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guarded_unused_class(name, nn, calls):
    def fail(self, *args, **kwargs):
        calls.append(name)
        raise RuntimeError("Inactive parser symbol was constructed: " + name)
    return type(name, (nn.Module,), {"__init__": fail})


def namespace(torch, nn, result):
    ns = {"__name__": "audited_msrdet", "torch": torch, "nn": nn,
          "F": nn.functional, "math": math, "contextlib": contextlib,
          "deepcopy": deepcopy, "Path": Path, "thop": None,
          "LOGGER": logging.getLogger("msrdet_audit")}
    # Original parse_model compares every listed class in sets, even when that
    # class never occurs in the YAML. Unique guards preserve these comparisons.
    tree = ast.parse((REPO / "ultralytics/nn/tasks.py").read_text(encoding="utf-8"))
    imported_classes = next(n for n in tree.body if isinstance(n, ast.ImportFrom)
                            and n.module == "ultralytics.nn.modules")
    result["inactive_class_constructor_calls"] = []
    for alias in imported_classes.names:
        ns[alias.name] = guarded_unused_class(alias.name, nn, result["inactive_class_constructor_calls"])
    manifest = []
    for relative, names in SELECTED.items():
        path = REPO / relative
        source = path.read_text(encoding="utf-8")
        definitions = {n.name: n for n in ast.parse(source).body
                       if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        for name in names:
            node = definitions[name]
            # No AST edits, no execution of the original file's imports or
            # module-level statements. Original line positions are retained.
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
            manifest.append({"file": relative, "name": name,
                             "start_line": node.lineno, "end_line": node.end_lineno,
                             "definition_sha256": hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()})
    result["executed_definition_manifest"] = manifest
    return ns


def simple(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def operator_signature(module):
    fields = ("in_channels", "out_channels", "kernel_size", "stride", "padding",
              "dilation", "groups", "num_features", "eps", "momentum", "affine",
              "track_running_stats", "inplace", "c", "c1", "add", "nc", "nl", "reg_max")
    return {"class": type(module).__name__,
            "attributes": {field: simple(getattr(module, field)) for field in fields
                           if hasattr(module, field)},
            "children": [[name, type(child).__name__, len(list(child.children()))]
                         for name, child in module.named_children()],
            "subtree_parameter_elements": sum(p.numel() for p in module.parameters())}


def complete_layer_descriptor(layer):
    """Architecture/routing descriptor, not numerical equality of random weights."""
    return {"from": layer.f,
            "modules": {name: operator_signature(module) for name, module in layer.named_modules()},
            "parameter_schema": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                        "requires_grad": value.requires_grad}
                                 for key, value in layer.named_parameters()},
            "buffer_schema": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                              for key, value in layer.named_buffers()}}


def strict_prefix(source, target):
    layers = []
    first_difference = None
    for index, (source_layer, target_layer) in enumerate(zip(source.model, target.model)):
        a, b = complete_layer_descriptor(source_layer), complete_layer_descriptor(target_layer)
        if a != b:
            first_difference = {"layer": index,
                                "source_class": type(source_layer).__name__,
                                "target_class": type(target_layer).__name__,
                                "source_descriptor": a, "target_descriptor": b}
            break
        layers.append(index)
    prefix_keys = [key for key in target.state_dict() if int(key.split(".")[1]) in layers]
    prefix_parameters = {key: value for key, value in target.named_parameters()
                         if int(key.split(".")[1]) in layers}
    parameter_elements = sum(value.numel() for value in prefix_parameters.values())
    return {"definition": "Contiguous prefix with identical recursive classes, reviewed operator attributes, child counts, parameter/buffer schema and input routing; same verbatim extracted class definitions. Numerical random weights are not compared.",
            "common_layer_indices": layers,
            "common_state_tensors": len(prefix_keys),
            "common_parameter_tensors": len(prefix_parameters),
            "common_parameter_elements": parameter_elements,
            "fraction_of_target_parameter_elements": parameter_elements / sum(p.numel() for p in target.parameters()),
            "fraction_of_source_parameter_elements": parameter_elements / sum(p.numel() for p in source.parameters()),
            "first_difference": first_difference}


def context_for(model, key):
    modules = dict(model.named_modules())
    owner_name = key.rsplit(".", 1)[0]
    pieces = owner_name.split(".")
    ancestors = [{"path": ".".join(pieces[:i]),
                  "class": type(modules[".".join(pieces[:i])]).__name__}
                 for i in range(1, len(pieces) + 1)]
    owner = modules[owner_name]
    # For a BN tensor inside a Conv wrapper, this also records the paired
    # convolution: matching BN shapes can hide different kernels/groups.
    parent_name = owner_name.rsplit(".", 1)[0]
    parent = modules[parent_name]
    nearby = {name: operator_signature(child) for name, child in parent.named_children()}
    layer = int(pieces[1])
    top = model.model[layer]
    return {"owner": operator_signature(owner), "ancestors": ancestors,
            "siblings_in_parent": nearby,
            "top_layer": layer, "top_class": type(top).__name__,
            "top_from": top.f, "top_signature": operator_signature(top)}


def inventory(source, target):
    source_state, target_state = source.state_dict(), target.state_dict()
    source_parameters, target_parameters = dict(source.named_parameters()), dict(target.named_parameters())
    rows = []
    for key, value in target_state.items():
        origin = source_state.get(key)
        matched = origin is not None and origin.shape == value.shape
        item = {"key": key, "kind": "parameter" if key in target_parameters else "buffer",
                "target_shape": list(value.shape), "target_numel": value.numel(),
                "source_shape": None if origin is None else list(origin.shape),
                "same_key_and_shape": matched, "target_context": context_for(target, key)}
        if origin is not None:
            item["source_context"] = context_for(source, key)
        if matched:
            a, b = item["source_context"], item["target_context"]
            item["differences"] = {
                "owner_operator": a["owner"] != b["owner"],
                "ancestor_classes": a["ancestors"] != b["ancestors"],
                "sibling_operators": a["siblings_in_parent"] != b["siblings_in_parent"],
                "top_layer_structure": a["top_signature"] != b["top_signature"],
                "top_input_routing": a["top_from"] != b["top_from"],
            }
        rows.append(item)
    matched = [r for r in rows if r["same_key_and_shape"]]
    param = [r for r in rows if r["kind"] == "parameter"]
    matched_param = [r for r in matched if r["kind"] == "parameter"]
    parameter_elements = sum(p.numel() for p in target_parameters.values())
    matched_elements = sum(r["target_numel"] for r in matched_param)
    return {"source_state_tensors": len(source_state), "target_state_tensors": len(rows),
            "matched_state_tensors": len(matched),
            "source_parameter_elements": sum(p.numel() for p in source_parameters.values()),
            "target_parameter_elements": parameter_elements,
            "matched_parameter_tensors": len(matched_param),
            "matched_parameter_elements": matched_elements,
            "target_parameter_element_coverage": matched_elements / parameter_elements,
            "target_buffer_elements": sum(r["target_numel"] for r in rows if r["kind"] == "buffer"),
            "matched_buffer_elements": sum(r["target_numel"] for r in matched if r["kind"] == "buffer"),
            "difference_counts_among_matched_tensors": {
                category: sum(r["differences"][category] for r in matched)
                for category in ("owner_operator", "ancestor_classes", "sibling_operators", "top_layer_structure", "top_input_routing")},
            "tensors": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if out.exists():
        raise SystemExit("Refusing to overwrite audit output: " + str(out))
    if out.drive.upper() != "D:":
        raise SystemExit("Audit output must remain on D:")
    for variable in ("TEMP", "TMP", "TORCH_HOME", "MPLCONFIGDIR", "YOLO_CONFIG_DIR"):
        if not os.environ.get(variable, "").upper().startswith("D:"):
            raise SystemExit("Dot-source project-env.ps1 first: " + variable)
    result = {"scope": "pinned-source AST reconstruction and synthetic marker load; no checkpoint, data, training or paper reproduction",
              "commit": "7267acc2b1fb96d3af2dfe2e406d0f430e6abfc5",
              "script_sha256": digest(Path(__file__)),
              "status": "started", "source_manifest": {},
              "limitations": ["Package initializers/dependency chain are bypassed; this does not establish full-package runtime compatibility.",
                              "Matching-context flags are structural differences, not proof of erroneous transfer or a performance effect.",
                              "A random model with source markers is used, not a pretrained YOLOv8 checkpoint.",
                              "No evidence here establishes that the paper authors invoked BaseModel.load in their experiments."]}
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        for relative, expected in PINNED.items():
            actual = digest(REPO / relative)
            if actual != expected:
                raise RuntimeError("Pinned source hash mismatch: " + relative)
            result["source_manifest"][relative] = actual
        import torch
        import torch.nn as nn
        import yaml
        result["environment"] = {"python": sys.version, "torch": torch.__version__,
                                 "device": "cpu", "threads": 2,
                                 "temp": os.environ["TEMP"]}
        torch.set_num_threads(2)
        torch.manual_seed(20261031)
        with torch.device("cpu"):
            ns = namespace(torch, nn, result)
            configs = {}
            for label, filename in (("source", "yolov8.yaml"), ("target", "MSRDet.yaml")):
                cfg = yaml.safe_load((REPO / "ultralytics/cfg/models/v8" / filename).read_text(encoding="utf-8"))
                cfg["scale"] = "n"
                assert not cfg.get("activation"), "Unreviewed eval activation forbidden"
                allowed = {"Conv", "C2f", "C2fk", "SPPF", "GatedSPPF", "nn.Upsample", "Concat", "FEM", "HTEM", "Focus", "Fusion_2in_mod", "Detect"}
                assert {row[2] for row in cfg["backbone"] + cfg["head"]} <= allowed
                configs[label] = cfg
            result["configs"] = deepcopy(configs)
            # Execute the untouched DetectionModel constructor, including the
            # original 1x3x256x256 zeros stride forward, and initialize_weights.
            with torch.no_grad():
                source = ns["DetectionModel"](deepcopy(configs["source"]), verbose=False)
                target = ns["DetectionModel"](deepcopy(configs["target"]), verbose=False)
            assert all(t.device.type == "cpu" for m in (source, target) for t in m.state_dict().values())
            result["construction"] = {"source_stride": source.stride.tolist(), "target_stride": target.stride.tolist(),
                                      "source_layers": len(source.model), "target_layers": len(target.model),
                                      "constructor_forward": "original zeros(1,3,256,256); no dataset"}
            result["strict_prefix"] = strict_prefix(source, target)
            result["audit"] = inventory(source, target)
            # Distinct source markers prove load execution for BN defaults too.
            with torch.no_grad():
                for index, value in enumerate(source.state_dict().values()):
                    value.fill_(1000 + index / 1000 if value.is_floating_point() else 1000 + index)
            before = {key: value.clone() for key, value in target.state_dict().items()}
            before_ids = {name: id(module) for name, module in target.named_modules()}
            target.load(source, verbose=False)  # exact original BaseModel.load
            after, source_state = target.state_dict(), source.state_dict()
            copied = unchanged = changed_matched = 0
            for row in result["audit"]["tensors"]:
                key = row["key"]
                if row["same_key_and_shape"]:
                    row["source_marker_copied"] = torch.equal(after[key], source_state[key])
                    row["target_changed"] = not torch.equal(before[key], after[key])
                    assert row["source_marker_copied"] and row["target_changed"], key
                    copied += 1
                    changed_matched += int(row["target_changed"])
                else:
                    row["unmatched_target_unchanged"] = torch.equal(before[key], after[key])
                    assert row["unmatched_target_unchanged"], key
                    unchanged += 1
            assert before_ids == {name: id(module) for name, module in target.named_modules()}
            assert not result["inactive_class_constructor_calls"]
            result["load_verification"] = {"all_matched_tensors_copied": copied,
                                            "matched_tensors_changed_from_before": changed_matched,
                                            "all_unmatched_tensors_unchanged": unchanged,
                                            "module_instances_preserved": True,
                                            "source_values": "synthetic unique constant markers; not pretrained weights"}
        assert "ultralytics" not in sys.modules
        result["full_ultralytics_package_imported"] = False
        result["status"] = "complete"
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("audit_output", str(out), "sha256", digest(out), flush=True)
        print(json.dumps({"status": result["status"],
                          "construction": result.get("construction"),
                          "summary": {k: v for k, v in result.get("audit", {}).items() if k != "tensors"},
                          "load_verification": result.get("load_verification")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
