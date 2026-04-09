"""Evaluate Icelandic inflection benchmark.
Usage:
  python eval_inflection.py --model Qwen/Qwen3.5-4B [--checkpoint PATH] [--difficulty easy] [--num_samples 50]
"""
import argparse, json, torch, os, sys
os.environ['HF_HOME'] = os.environ.get('HF_HOME', '/dtu/p1/vestsn/isft/hf_cache')

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name, checkpoint=None, device='cuda:0'):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.environ['HF_HOME'])

    if checkpoint:
        from loqt.LoQT import LoQTModel
        with open(f'{checkpoint}/loqt_config.json') as f:
            cfg = json.load(f)
        base = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, trust_remote_code=True,
            cache_dir=os.environ['HF_HOME'])
        model = LoQTModel(
            base, r=cfg['r'], lora_alpha=cfg['lora_alpha'],
            target_modules=cfg['target_modules'],
            quantize_w=cfg.get('quantize_w'), use_double_quant=cfg.get('use_double_quant', False),
            device=torch.device(device), proj_type=cfg.get('proj_type', 'std'),
            compute_dtype=torch.bfloat16,
            quantize_projection_matrix=cfg.get('quantize_projection_matrix'),
            compensate_quant_error_iterations=cfg.get('compensate_quant_error_iterations', 0),
            use_offloading=True, is_single_gpu=True, model_config={},
            use_eigenh_for_projection=False, init_lora_AB_as_random_and_zeros=False,
            train_projection_matrix=False, update_steps=[100],
            grad_accumulation_steps=cfg.get('grad_accumulation_steps', 16))
        sd = torch.load(f'{checkpoint}/pytorch_model_full.pth', map_location=device, weights_only=False)
        model.load_state_dict(sd, strict=False)
        del sd
        model.to(device).eval()
        return model.wrapped_model, tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, trust_remote_code=True,
            cache_dir=os.environ['HF_HOME'])
        model = model.to(device).eval()
        return model, tokenizer


def evaluate(model, tokenizer, difficulty='easy', num_samples=50, device='cuda:0'):
    ds = load_dataset(f'mideind/icelandic-inflection-{difficulty}', split='train')
    samples = list(ds.select(range(min(num_samples, len(ds)))))

    correct = 0
    total = 0
    field_correct = 0
    field_total = 0

    for i, sample in enumerate(samples):
        prompt = sample['input']
        ideal = sample['ideal']

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        response = tokenizer.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

        # Try to extract JSON from response
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                pred = json.loads(response[start:end])
                ideal_parsed = json.loads(ideal)

                # Exact match (normalized)
                if json.dumps(pred, sort_keys=True, ensure_ascii=False).lower() == json.dumps(ideal_parsed, sort_keys=True, ensure_ascii=False).lower():
                    correct += 1

                # Field-level: check each inflected form appears in the prediction
                for num in ['et', 'ft']:
                    for case in ['nf', 'þf', 'þgf', 'ef']:
                        field_total += 1
                        ideal_form = ideal_parsed.get(num, {}).get(case, '').strip().lower()
                        # Check nested format first, then flat
                        pred_form = ''
                        if isinstance(pred.get(num), dict):
                            pred_form = pred.get(num, {}).get(case, '').strip().lower()
                        else:
                            # Flat format: check if the ideal form appears anywhere in pred values
                            for v in pred.values():
                                if isinstance(v, str) and v.strip().lower() == ideal_form:
                                    pred_form = ideal_form
                                    break
                                elif isinstance(v, dict):
                                    for vv in v.values():
                                        if isinstance(vv, str) and vv.strip().lower() == ideal_form:
                                            pred_form = ideal_form
                                            break
                        if pred_form == ideal_form and ideal_form:
                            field_correct += 1
        except (json.JSONDecodeError, Exception):
            # Count fields even if JSON parse fails
            ideal_parsed = json.loads(ideal)
            for num in ['et', 'ft']:
                for case in ['nf', 'þf', 'þgf', 'ef']:
                    field_total += 1

        total += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{num_samples}] exact={correct}/{total} ({100*correct/total:.1f}%) fields={field_correct}/{field_total} ({100*field_correct/field_total:.1f}%)")

    exact_acc = 100 * correct / total if total > 0 else 0
    field_acc = 100 * field_correct / field_total if field_total > 0 else 0
    return exact_acc, field_acc, total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen3.5-4B')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--difficulty', type=str, default='easy', choices=['easy', 'medium', 'hard'])
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint or 'None (base model)'}")
    print(f"Difficulty: {args.difficulty}, Samples: {args.num_samples}")
    print()

    model, tokenizer = load_model(args.model, args.checkpoint, args.device)
    exact, field, total = evaluate(model, tokenizer, args.difficulty, args.num_samples, args.device)

    print(f"\n{'='*50}")
    print(f"Results ({args.difficulty}, n={total}):")
    print(f"  Exact match: {exact:.1f}%")
    print(f"  Field accuracy: {field:.1f}%")
    print(f"{'='*50}")
