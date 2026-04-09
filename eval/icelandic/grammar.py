"""Icelandic grammar quality evaluation using GreynirCorrect.

Generates text from a model given Icelandic prompts, then measures grammar
error rate using GreynirCorrect (https://github.com/mideind/GreynirCorrect).

Metrics:
- errors_per_sentence: Average number of grammar errors per generated sentence
- error_free_rate: Proportion of sentences with zero grammar errors
- error_types: Distribution of error categories

Usage:
    python -m eval.icelandic.grammar --model Qwen/Qwen3.5-4B [--checkpoint PATH]
"""
import argparse
import json
import os
import sys
import torch
from collections import Counter
from typing import Optional

os.environ['HF_HOME'] = os.environ.get('HF_HOME', '/dtu/p1/vestsn/isft/hf_cache')


PROMPTS = [
    "Segðu mér frá sögu Íslands.",
    "Lýstu veðrinu á Íslandi.",
    "Hvað er sérstakt við íslenska menningu?",
    "Hvernig virkar íslenska heilbrigðiskerfið?",
    "Segðu mér frá íslenskum bókmenntum.",
    "Lýstu landslagi Íslands.",
    "Hvað er Alþingi og hvers vegna er það mikilvægt?",
    "Segðu mér frá sjávarútvegi á Íslandi.",
    "Hvernig er menntakerfið á Íslandi?",
    "Lýstu áhrifum loftslagsbreytinga á Ísland.",
    "Segðu mér frá jafnrétti kynjanna á Íslandi.",
    "Hvað gerir íslensku ólíka öðrum norrænum tungumálum?",
    "Lýstu ferðamennsku á Íslandi.",
    "Segðu mér frá orkumálum á Íslandi.",
    "Hvernig hefur tæknin breytt íslensku samfélagi?",
]


def load_model(model_name: str, checkpoint: Optional[str] = None, device: str = 'cuda:0'):
    """Load base or fine-tuned model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
        return model.to(device).eval(), tokenizer


def generate_texts(model, tokenizer, prompts: list[str], device: str = 'cuda:0',
                   max_new_tokens: int = 200) -> list[str]:
    """Generate responses for each prompt."""
    texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer([text], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        response = tokenizer.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        texts.append(response)
    return texts


def check_grammar(texts: list[str]) -> dict:
    """Run GreynirCorrect on generated texts and return error statistics."""
    from reynir_correct import check_single

    total_sentences = 0
    total_errors = 0
    error_free = 0
    error_types = Counter()
    all_errors = []

    for text in texts:
        # Split into sentences (rough split on period/question mark)
        sentences = [s.strip() for s in text.replace('?', '.').replace('!', '.').split('.')
                     if s.strip() and len(s.strip()) > 5]

        for sent in sentences:
            total_sentences += 1
            try:
                result = check_single(sent)
                annotations = list(result.annotations)
                n_errors = len(annotations)
                total_errors += n_errors
                if n_errors == 0:
                    error_free += 1
                for ann in annotations:
                    error_types[ann.code] += 1
                    all_errors.append({
                        'sentence': sent,
                        'code': ann.code,
                        'text': ann.text,
                        'suggest': ann.suggest,
                    })
            except Exception:
                pass  # Skip sentences that can't be parsed

    return {
        'total_sentences': total_sentences,
        'total_errors': total_errors,
        'errors_per_sentence': total_errors / max(total_sentences, 1),
        'error_free_rate': error_free / max(total_sentences, 1),
        'error_types': dict(error_types.most_common(20)),
        'sample_errors': all_errors[:10],
    }


def evaluate(model_name: str, checkpoint: Optional[str] = None,
             device: str = 'cuda:0', num_prompts: int = 10) -> dict:
    """Full grammar evaluation pipeline."""
    model, tokenizer = load_model(model_name, checkpoint, device)
    prompts = PROMPTS[:num_prompts]

    print(f"Generating {len(prompts)} responses...")
    texts = generate_texts(model, tokenizer, prompts, device)

    print("Checking grammar...")
    results = check_grammar(texts)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen3.5-4B')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_prompts', type=int, default=10)
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint or 'None (base)'}")
    print()

    results = evaluate(args.model, args.checkpoint, args.device, args.num_prompts)

    print(f"\n{'='*50}")
    print(f"Grammar Evaluation Results")
    print(f"{'='*50}")
    print(f"  Sentences analyzed: {results['total_sentences']}")
    print(f"  Total errors: {results['total_errors']}")
    print(f"  Errors/sentence: {results['errors_per_sentence']:.2f}")
    print(f"  Error-free rate: {results['error_free_rate']:.1%}")
    print(f"\n  Top error types:")
    for code, count in list(results['error_types'].items())[:10]:
        print(f"    {code}: {count}")
    if results['sample_errors']:
        print(f"\n  Sample errors:")
        for err in results['sample_errors'][:3]:
            print(f"    [{err['code']}] \"{err['text']}\" -> \"{err['suggest']}\"")
    print(f"{'='*50}")
