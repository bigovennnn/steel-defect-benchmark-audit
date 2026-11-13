"""CPU-only exploratory transfer audit of local package variants; no training or test inference."""
from pathlib import Path
import argparse, hashlib, json, sys
import torch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / 'paper-try'))
from yolo_paper import register_modules
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import ultralytics

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def inventory(model, source):
    target = model.state_dict()
    params = dict(model.named_parameters())
    head = f'model.{len(model.model)-1}.cv3.'
    rows = []
    for key, value in sorted(target.items()):
        src = source.get(key)
        matched = src is not None and src.shape == value.shape
        rows.append({'key': key, 'target_shape': list(value.shape),
                     'source_shape': list(src.shape) if src is not None else None,
                     'kind': 'parameter' if key in params else 'buffer',
                     'numel': value.numel(), 'same_key_and_shape': matched,
                     'partial_transfer_eligible': matched and not key.startswith(head),
                     'classification_branch': key.startswith(head)})
    param_rows = [r for r in rows if r['kind'] == 'parameter']
    return {'parameters': sum(r['numel'] for r in param_rows),
            'eligible_parameter_elements': sum(r['numel'] for r in param_rows if r['partial_transfer_eligible']),
            'matching_parameter_elements_including_head': sum(r['numel'] for r in param_rows if r['same_key_and_shape']),
            'tensors': rows}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit('Refusing to overwrite output: ' + str(out))
    register_modules()
    weight = BASE / 'weights/yolo11n_coco_official.pt'
    expected = '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'
    if digest(weight) != expected:
        raise SystemExit('Source weight SHA-256 mismatch')
    pretrained = YOLO(str(weight)).model.float().cpu()
    source = pretrained.state_dict()
    result = {'status': 'exploratory; same key/shape is not evidence of semantic equivalence',
              'purpose': 'inspect local supplied variants, not published paper implementations',
              'ultralytics': ultralytics.__version__, 'torch': torch.__version__,
              'source_weight_sha256': expected, 'nc': 6,
              'partial_policy': 'matching shape and key excluding entire detection cv3 classification branch',
              'variants': {}}
    for variant in ('base', 'hwd', 'strip', 'hwd-strip'):
        actual = BASE / 'paper-try/configs/models' / f'yolo11-{variant}.yaml'
        cfg = actual.with_name(f'yolo11n-{variant}.yaml')
        model = DetectionModel(str(cfg), nc=6, verbose=False).cpu()
        item = inventory(model, source)
        item['yaml_sha256'] = digest(actual)
        # Ultralytics may rebuild parts of the target on load; inventory that separately.
        model.load(pretrained, verbose=False)
        after = inventory(model, source)
        item['after_framework_load'] = {'parameters': after['parameters'],
                                         'matching_parameter_elements_including_head': after['matching_parameter_elements_including_head'],
                                         'classification_shapes': {r['key']: r['target_shape'] for r in after['tensors'] if r['classification_branch']}}
        result['variants'][variant] = item
        print(variant, 'parameters', item['parameters'], 'eligible', item['eligible_parameter_elements'],
              'after_default_load', after['parameters'], flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('output', out, 'sha256', digest(out))

if __name__ == '__main__':
    main()
