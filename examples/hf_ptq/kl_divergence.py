# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""KL divergence of a quantized model from its unquantized self, scored like llama.cpp.

The text is split into consecutive ``seq_len``-token chunks and only the second half of each
chunk is scored, as ``llama-perplexity --kl-divergence`` does, so the numbers are directly
comparable with llama.cpp's for the same model and text.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from datasets import load_dataset

__all__ = [
    "KLDivergenceReference",
    "collect_reference",
    "format_kl_divergence",
    "kl_divergence",
    "load_eval_tokens",
]


@dataclass
class KLDivergenceReference:
    """The scored chunks and the unquantized model's log-probabilities on them."""

    chunks: torch.Tensor  # (num_chunks, seq_len) token ids
    log_probs: torch.Tensor  # (num_chunks, seq_len - 1 - seq_len // 2, vocab) float16, on CPU


def load_eval_tokens(tokenizer, text_file: str | None = None) -> torch.Tensor:
    """Token ids of ``text_file``, or of wikitext-2 test laid out as llama.cpp's wiki.test.raw."""
    if text_file is None:
        rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
        text = "".join(row or " \n" for row in rows)  # wiki.test.raw writes blank rows as " \n"
    else:
        with open(text_file, encoding="utf-8") as f:
            text = f.read()
    return torch.tensor(tokenizer(text, add_special_tokens=False)["input_ids"])


@torch.no_grad()
def collect_reference(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    num_chunks: int = 100,
    seq_len: int = 512,
    bos_token_id: int | None = None,
) -> KLDivergenceReference:
    """Score ``model`` before quantization; pass the BOS id if its tokenizer prepends one."""
    num_chunks = min(num_chunks, tokens.numel() // seq_len)
    if num_chunks == 0:
        raise ValueError(f"The evaluation text is shorter than one {seq_len}-token chunk.")
    chunks = tokens[: num_chunks * seq_len].reshape(num_chunks, seq_len).clone()
    if bos_token_id is not None:
        chunks[:, 0] = bos_token_id
    log_probs = None
    for i, chunk in enumerate(chunks):
        chunk_log_probs = _scored_log_probs(model, chunk).half().cpu()
        if log_probs is None:
            log_probs = chunk_log_probs.new_empty((num_chunks, *chunk_log_probs.shape))
        log_probs[i] = chunk_log_probs
    return KLDivergenceReference(chunks, log_probs)


@torch.no_grad()
def kl_divergence(model: torch.nn.Module, reference: KLDivergenceReference) -> dict[str, float]:
    """Mean KL(reference || model) per scored token, with both models' perplexities.

    Returns:
        ``kld``, ``ppl``, ``ppl_base`` and ``same_top`` (the fraction of tokens whose most
        likely next token is unchanged), each with a ``*_stderr``, and ``num_tokens``.
    """
    kld, nll, nll_base, same_top = [], [], [], []
    for chunk, base in zip(reference.chunks, reference.log_probs):
        log_probs = _scored_log_probs(model, chunk)
        base = base.to(log_probs.device, torch.float32)
        targets = chunk[chunk.numel() // 2 + 1 :, None].to(log_probs.device)
        kld.append(F.kl_div(log_probs, base, reduction="none", log_target=True).sum(-1))
        nll.append(-log_probs.gather(-1, targets)[:, 0])
        nll_base.append(-base.gather(-1, targets)[:, 0])
        same_top.append((log_probs.argmax(-1) == base.argmax(-1)).float())
    kld, nll, nll_base, same_top = (
        torch.cat(v).double().cpu() for v in (kld, nll, nll_base, same_top)
    )

    def stderr(values):
        return (values.var() / values.numel()).sqrt().item()

    ppl, ppl_base = nll.mean().exp().item(), nll_base.mean().exp().item()
    return {
        "kld": kld.mean().item(),
        "kld_stderr": stderr(kld),
        "ppl": ppl,
        "ppl_stderr": ppl * stderr(nll),
        "ppl_base": ppl_base,
        "ppl_base_stderr": ppl_base * stderr(nll_base),
        "same_top": same_top.mean().item(),
        "same_top_stderr": stderr(same_top),
        "num_tokens": kld.numel(),
    }


def format_kl_divergence(result: dict[str, float]) -> str:
    """The result in llama-perplexity's summary wording."""
    return "\n".join(
        [
            f"KL divergence vs. the unquantized model over {result['num_tokens']} scored tokens:",
            f"  Mean PPL(Q)    : {result['ppl']:10.6f} ± {result['ppl_stderr']:.6f}",
            f"  Mean PPL(base) : {result['ppl_base']:10.6f} ± {result['ppl_base_stderr']:.6f}",
            f"  Mean KLD       : {result['kld']:10.6f} ± {result['kld_stderr']:.6f}",
            f"  Same top p     : {100 * result['same_top']:10.3f} ± "
            f"{100 * result['same_top_stderr']:.3f} %",
        ]
    )


def _scored_log_probs(model: torch.nn.Module, chunk: torch.Tensor) -> torch.Tensor:
    """Log-softmax of the logits that predict the second half of ``chunk``."""
    logits = model(input_ids=chunk[None].to(model.device), use_cache=False).logits[0]
    return F.log_softmax(logits[chunk.numel() // 2 : -1].float(), dim=-1)
