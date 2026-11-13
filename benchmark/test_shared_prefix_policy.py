"""Meaningful positive/negative checks for the manually reviewed shared-prefix policy."""
import argparse
import copy
import json
from pathlib import Path
import unittest

import torch
import shared_prefix_policy as policy


class SharedPrefixChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.targets = policy.construct_targets(17)
        cls.source = policy.load_source()

    def test_source_loading_and_construction_preserve_caller_rng(self):
        torch.manual_seed(805)
        before = torch.get_rng_state().clone()
        policy.construct_targets(19)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        policy.load_source()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_explicit_pairing_with_independent_random_initializations(self):
        other = policy.construct_targets(18)["hwd-strip"]
        original = policy.state_hashes(other)
        base_hashes = policy.state_hashes(self.targets["base"])
        self.assertNotEqual(original[policy.WHITE_KEYS[0]], base_hashes[policy.WHITE_KEYS[0]])
        _, after = policy.copy_white(other, self.targets["base"], "hwd-strip", "base")
        for key in after:
            self.assertEqual(after[key], base_hashes[key] if key in policy.WHITE_KEYS else original[key])

    def test_copy_values_all_skipped_state_architecture_and_trainability(self):
        for name, scratch in self.targets.items():
            model = copy.deepcopy(scratch)
            old_hashes = policy.state_hashes(model)
            old_graph = policy.graph_signature(model)
            before, after = policy.copy_white(model, self.source, name, "source")
            self.assertEqual(before, old_hashes)
            self.assertEqual(policy.graph_signature(model), old_graph)
            source_hashes = policy.state_hashes(self.source)
            for key in after:
                self.assertEqual(after[key], source_hashes[key] if key in policy.WHITE_KEYS else before[key])
            self.assertEqual(policy.state_hashes(scratch), old_hashes)

    def test_reject_same_type_same_shape_changed_stride(self):
        model = copy.deepcopy(self.targets["hwd-strip"])
        model.model[0].conv.stride = (1, 1)
        before = policy.state_hashes(model)
        with self.assertRaisesRegex(ValueError, "Conv2d semantics"):
            policy.copy_white(model, self.source, "hwd-strip", "source")
        self.assertEqual(before, policy.state_hashes(model))

    def test_reject_same_type_same_shape_changed_bn_semantics(self):
        model = copy.deepcopy(self.targets["base"])
        model.model[0].bn.eps = 0.01
        with self.assertRaisesRegex(ValueError, "BatchNorm2d semantics"):
            policy.validate_stem(model, "base")

    def test_reject_changed_graph_input_and_hook(self):
        model = copy.deepcopy(self.targets["base"])
        model.model[0].f = 1
        with self.assertRaisesRegex(ValueError, "raw-input graph entry"):
            policy.validate_stem(model, "base")
        model.model[0].f = -1
        handle = model.model[0].register_forward_pre_hook(lambda module, inputs: inputs)
        with self.assertRaisesRegex(ValueError, "Forward hooks"):
            policy.validate_stem(model, "base")
        handle.remove()
        policy.validate_stem(model, "base")

    def test_reject_fused_or_instance_overridden_forward(self):
        model = copy.deepcopy(self.targets["base"])
        model.model[0].forward = model.model[0].forward_fuse
        with self.assertRaisesRegex(ValueError, "Changed stem forward"):
            policy.validate_stem(model, "base")

    def test_train_eval_forward_gradients_bn_updates_and_source_inplace(self):
        models = {name: copy.deepcopy(model) for name, model in self.targets.items()}
        for name, model in models.items():
            policy.copy_white(model, self.source, name, "source")
        rows = policy.probe_prefix(models["base"], models["hwd-strip"], self.source)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["forward_gradient_and_buffer_bitwise_equal"] for row in rows))

    def test_repeatable_build_and_full_audit_accounting(self):
        first, second = policy.build(23), policy.build(23)
        self.assertEqual(first, second)
        for name, item in first["targets"].items():
            self.assertEqual(item["copied_parameter_elements"], 464)
            self.assertEqual(item["copied_buffer_elements"], 33)
            self.assertEqual(item["total_parameter_elements"], 2591010 if name == "base" else 2215402)
            rows = item["policies"]["C_strict_prefix"]
            self.assertEqual({r["key"] for r in rows if r["action"] == "copied"}, set(policy.WHITE_KEYS))
            self.assertTrue(all(r["after_sha256"] == (r["source_sha256"] if r["action"] == "copied"
                                                     else r["scratch_sha256"]) for r in rows))
            self.assertTrue(all(r["action"] == "retained" for r in item["policies"]["S_paired_prefix"]))


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed_names = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed_names.append(test.id())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy.require(args.output.resolve().drive.upper() == "D:", "Output must be on D drive")
    policy.require(not args.output.exists(), "Refusing overwrite: " + str(args.output))
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SharedPrefixChecks)
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)
    report = {"success": result.wasSuccessful(), "tests_run": result.testsRun,
              "passed": sorted(result.passed_names),
              "failures": [{"name": t.id(), "traceback": s} for t, s in result.failures + result.errors],
              "script_sha256": policy.sha_file(__file__),
              "policy_script_sha256": policy.sha_file(policy.__file__),
              "scope": "CPU-only construction, positive/negative copy checks, synthetic prefix probes; no training"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
