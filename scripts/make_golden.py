#!/usr/bin/env python3
"""Generate per-token NumPy logits and greedy token IDs for a tiny F32 model."""

import argparse
from pathlib import Path

import numpy as np

from reference_forward import forward_token, load_model


ROOT = Path(__file__).resolve().parents[1]
BOS, EOS = 1, 2


def encode_prompt(prompt):
    """BOS followed by ASCII character IDs; reject unsupported text explicitly."""
    return [BOS, *prompt.encode("ascii")]


def make_golden(model_dir, output_dir, prompt="hello", n_predict=4):
    model = load_model(model_dir)
    tokens = encode_prompt(prompt)
    if n_predict < 0 or len(tokens) + n_predict > model.config["max_seq_len"]:
        raise ValueError("requested tokens exceed context window or negative budget")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step = 0

    def forward_and_dump(token):
        nonlocal step
        logits = forward_token(model, step, token)
        logits.tofile(output_dir / f"python_logits_step{step}.bin")
        step += 1
        return logits

    for token in tokens:
        logits = forward_and_dump(token)
    generated = []
    for index in range(n_predict):
        # NumPy argmax returns the first maximum, matching C++'s smaller-ID tie.
        token = int(np.argmax(logits))
        generated.append(token)
        if token == EOS or index + 1 == n_predict:
            break
        # The first generated token comes directly from the final prefill logits.
        logits = forward_and_dump(token)
    (output_dir / "python_generation_tokens.txt").write_text(
        " ".join(map(str, generated)) + "\n"
    )
    return step, generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/tiny")
    parser.add_argument("--output-dir", "--output", type=Path, required=True)
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--n-predict", type=int, default=4)
    args = parser.parse_args()
    steps, generated = make_golden(args.model, args.output_dir, args.prompt, args.n_predict)
    print(f"Python: {steps} logits steps, generated tokens: {generated}")


if __name__ == "__main__":
    main()
