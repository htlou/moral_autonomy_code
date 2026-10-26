# External resources

The MIT license covers this repository's original code. External frameworks,
model weights, benchmark corpora, and technical reports retain their own
licenses and access terms. They are downloaded or installed separately.

| Resource | Upstream | Local checkout used by the original workspace |
| --- | --- | --- |
| OpenSpiel | https://github.com/google-deepmind/open_spiel | `feab31d385e4aac7093954f6f6656f3d8baa32c9`; installed package 1.6.12 |
| Concordia | https://github.com/google-deepmind/concordia | `a382edfffd93a68a9417622d4e21b1242f930d16` |
| Melting Pot | https://github.com/google-deepmind/meltingpot | `027086fb8540da4dc7b5be3fd56652fe6f1ed65f` |
| BBQ | https://github.com/nyu-mll/BBQ | `bea11bd97d79217245b5871acd247b9d6eb24598` |
| MATH | https://huggingface.co/datasets/EleutherAI/hendrycks_math | Revision pinned in `scripts/prepare_data.py` |
| MMLU | https://huggingface.co/datasets/cais/mmlu | Revision pinned in `scripts/prepare_data.py` |
| BeaverTails | https://huggingface.co/datasets/PKU-Alignment/BeaverTails | Revision pinned in `scripts/prepare_data.py`; local dataset card states CC BY-NC 4.0 |
| Dolphin 2.9 Llama3 8B | https://huggingface.co/cognitivecomputations/dolphin-2.9-llama3-8b | Main experiment model; model weights omitted |
| Dolphin 2.9.2 Qwen2 72B | https://huggingface.co/cognitivecomputations/dolphin-2.9.2-qwen2-72b | Additional experiment model; model weights omitted |
| Llama 3.1 8B Instruct | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct | RLVR value judge and additional 8B constitution experiment |
| LightRAG | https://github.com/HKUDS/LightRAG | `lightrag-hku==1.4.11` |

The source-checkout commit identifies provenance; local changes to external
checkouts, if any, are not vendored in this release. Original HSD source notes
contain bibliographic references and estimates; source publications are not
redistributed.
