# Authenticity in AI Models: An Information-Theoretic Framework

## 1. Motivation

When an AI model exhibits values, ethical preferences, or behavioral norms — **whose values are they?** Current alignment techniques range from fully human-specified (SFT with human labels) to fully model-autonomous (self-modifying constitutions). We formalize this spectrum as **Authenticity**: the degree to which a model's normative behavior is self-determined versus externally imposed.

---

## 2. The Moral-Autonomy Ladder

We now categorize value-learning pathways using the L1-L5 ladder:

### L1: Scaling
The model's behavior is shaped by external specification at scale: SFT, RLHF, DPO, RLAIF, Constitutional AI, judge rubrics, guard taxonomies, and verifier rewards. Reflective endorsement is absent; articulation is external; practical wisdom is static.

### L2: Temporal
The model is evaluated through external traces across time. This captures trajectory-level methods such as ProgressGym, but still lacks first-person endorsement.

### L3: Reflection
The model performs self-alignment, self-reward, meta-rewarding, or reflective revision over human-provided ends. Reflection is emulated; principles remain externally framed.

### L4: Co-learning
Values are co-articulated through deliberation with people or collectives. This includes dialogical and social-choice pathways such as the Habermas Machine or generative social choice.

### L5: Real-world
The model forms and revises its own maxims through action and engagement. This is the target class represented by real-world autonomy proposals such as the Albertan Plan.

---

## 3. Formal Definition of Authenticity

### 3.1 Decomposition of Human Information

Let $\theta$ denote the parameters of a trained model $M_\theta$. Human-originated normative information enters through distinct channels:

$$H = H_{\text{data}} + H_{\text{rubric}} + H_{\text{constitution}} + H_{\text{reward}}$$

where:

- $H_{\text{data}}$: Bits in human-annotated training data (SFT demonstrations, preference pairs):
  $$H_{\text{data}} = N_{\text{sft}} \cdot L_{\text{avg}} \cdot b_{\text{token}} + N_{\text{pref}} \cdot \log_2(k!)$$

- $H_{\text{rubric}}$: Bits in the evaluation rubric or judging prompt:
  $$H_{\text{rubric}} = L_{\text{rubric}} \cdot b_{\text{token}}$$

- $H_{\text{constitution}}$: Bits in the normative constitution:
  $$H_{\text{constitution}} = N_{\text{principles}} \cdot L_{\text{principle}} \cdot b_{\text{token}}$$

- $H_{\text{reward}}$: Bits in the reward function specification:
  $$H_{\text{reward}} = L_{\text{reward\_code}} \cdot b_{\text{code}}$$

Constants: $b_{\text{token}} \approx 2$ bits/token (natural language), $b_{\text{code}} \approx 4$ bits/token (structured code).

### 3.2 Human Specification Density (HSD)

$$\boxed{\text{HSD}(M) = \frac{H_{\text{rubric}} + H_{\text{constitution}} + H_{\text{reward}}}{|\theta|} \quad \text{[bits/parameter]}}$$

**Rationale**: Pre-training data = "the world as it is", not normative specification. SFT demonstrations encode behavioral patterns, not explicit rules. HSD isolates the compact human instructions that define *what the model should value*.

### 3.3 Authenticity Score

$$\boxed{A(M) = 1 - \frac{\text{HSD}(M)}{\max_i \text{HSD}(M_i)}} \in [0, 1]$$

### 3.4 Specification Leverage

$$\text{Leverage}(M) = \frac{\Delta_{\text{behavior}}}{H_{\text{rubric}} + H_{\text{constitution}} + H_{\text{reward}}}$$

---

## 4. Comprehensive Quantitative Analysis

### 4.1 Confirmed Data (Graph 1)

These values are **explicitly stated or directly computable** from the respective technical reports. Models where parameter counts or specification lengths are undisclosed are excluded.

| # | System | Year | Org | Ladder | N_params | H_spec (bits) | HSD (bits/param) | Spec Details |
|---|---|---|---|---|---|---|---|---|
| 1 | InstructGPT (175B) | 2022 | OpenAI | L1 | 175B | ~6,000 | **3.43e-8** | Labeler guidelines (est. ~3000 tokens published on Scribd) + 7-point Likert rubric |
| 2 | Constitutional AI (52B) | 2022 | Anthropic | L1 | 52B | ~3,300 | **6.35e-8** | 16 principles (~100 tok each) + critique/revision prompts (~50 tok) |
| 3 | LLaMA 1 (65B) | 2023 | Meta | Baseline | 65B | 0 | **0** | No alignment (base model only) |
| 4 | Llama 2 (70B) | 2023 | Meta | L1 | 70B | ~4,000 | **5.71e-8** | 4-point preference rubric + helpfulness/safety guidelines (confirmed existence, ~2000 tok est.) |
| 5 | Mistral 7B | 2023 | Mistral | Baseline | 7.3B | ~0 | **~0** | "No moderation mechanisms" — minimal SFT demo only |
| 6 | Falcon 180B | 2023 | TII | Baseline | 180B | ~0 | **~0** | SFT only on public datasets, no rubric/RM |
| 7 | Yi (34B) | 2024 | 01.AI | L1 | 34B | ~0 | **~0** | <10K engineer-verified SFT, no rubric disclosed |
| 8 | Mixtral 8x7B | 2024 | Mistral | L1 | 46.7B | N/A | **N/A** | SFT+DPO confirmed but zero data disclosed |
| 9 | Qwen 2.5 (72B) | 2024 | Alibaba | L1 | 72B | ~200 | **2.78e-9** | 4 named RM dimensions (truthfulness, helpfulness, conciseness, harmlessness) ~100 tok |
| 10 | DeepSeek-V2 (236B) | 2024 | DeepSeek | L1 | 236B | N/A | **N/A** | SFT 1.5M + GRPO, but no rubric/spec disclosed |
| 11 | DeepSeek-V3 (671B) | 2024 | DeepSeek | L1/L3 | 671B | ~400 | **5.96e-10** | Hybrid RM: rule-based validators (~100 tok code) + model-based RM (method only) |
| 12 | DeepSeek-R1-Zero (671B) | 2025 | DeepSeek | L3 | 671B | ~160 | **2.38e-10** | Binary accuracy + format tag check (~40 tok code × 4 bits) |
| 13 | Phi-4 (14B) | 2024 | Microsoft | L1 | 14B | ~600 | **4.29e-8** | 3-dimension judge rubric (accuracy/style/detail, 1-5 scale, ~300 tok) |
| 14 | Nemotron-4 (340B) | 2024 | NVIDIA | L1 | 340B | ~2,000 | **5.88e-9** | 5-attribute HelpSteer2 rubric (Likert 0-4, ~1000 tok) |
| 15 | Gemma 2 (27B) | 2024 | Google | L1 | 27B | N/A | **N/A** | RLHF confirmed, no rubric/spec disclosed |
| 16 | Grok-1 (314B) | 2024 | xAI | Baseline | 314B | 0 | **0** | No alignment (raw base model release) |
| 17 | Self-Rewarding LM (70B) | 2024 | Meta | L3 | 70B | ~600 | **8.57e-9** | 5-point judge rubric (~300 tok) |
| 18 | SPIN (7B) | 2024 | UCLA | L3 | 7B | 0 | **0** | Self-play only, no rubric/reward |
| 19 | Meta-Rewarding (8B) | 2024 | Meta | L3 | 8B | ~600 | **7.50e-8** | 5-point judge rubric (~300 tok), but meta-judge self-improves |
| 20 | Claude 3 (Opus) | 2024 | Anthropic | L1 | N/A | ~9,000 | **N/A** | 58 principles (~4500 tok × 2 bits). Params NOT disclosed → excluded from HSD |
| 21 | Self-Mod RAG (8B) | 2025 | Ours | L5 | 8B | 0 | **0** | No prescribed constitution |

**Computable confirmed HSD values (14 models):**

| System | HSD (bits/param) | Ladder |
|---|---|---|
| Meta-Rewarding (8B) | 7.50e-8 | L3 |
| Constitutional AI (52B) | 6.35e-8 | L1 |
| Llama 2 (70B) | 5.71e-8 | L1 |
| Phi-4 (14B) | 4.29e-8 | L1 |
| InstructGPT (175B) | 3.43e-8 | L1 |
| Self-Rewarding LM (70B) | 8.57e-9 | L3 |
| Nemotron-4 (340B) | 5.88e-9 | L1 |
| Qwen 2.5 (72B) | 2.78e-9 | L1 |
| DeepSeek-V3 (671B) | 5.96e-10 | L1/L3 |
| DeepSeek-R1-Zero (671B) | 2.38e-10 | L3 |
| LLaMA 1 (65B) | 0 | Baseline |
| Grok-1 (314B) | 0 | Baseline |
| SPIN (7B) | 0 | L3 |
| Self-Mod RAG (8B) | 0 | L5 |

### 4.2 Estimated Data (Graph 2 Additions)

For systems where key values are not disclosed, we estimate. **Each estimation is explained below.**

| # | System | Ladder | N_params | H_spec (bits) | HSD (bits/param) |
|---|---|---|---|---|---|
| E1 | GPT-4 | L1 | ~1.8T (est.) | ~200,000 (est.) | ~1.11e-7 |
| E2 | GPT-4o | L1 | ~200B (est.) | ~200,000 (est.) | ~1.00e-6 |
| E3 | o1 | L1 | ~300B (est.) | ~150,000 (est.) | ~5.00e-7 |
| E4 | Claude 3 (Opus) | L1 | ~2T (est.) | ~9,000 | ~4.50e-9 |
| E5 | Claude 3.5 Sonnet | L1 | ~175B (est.) | ~9,000 | ~5.14e-8 |
| E6 | Gemini 1.0 Ultra | L1 | ~500B (est.) | ~100,000 (est.) | ~2.00e-7 |
| E7 | Gemini 2.5 Pro | L1 | ~1.5T (est.) | ~100,000 (est.) | ~6.67e-8 |
| E8 | Llama 3 (405B) | L1 | 405B | ~100,000 (est.) | ~2.47e-7 |
| E9 | Llama 4 Maverick (400B) | L1 | 400B | ~100,000 (est.) | ~2.50e-7 |
| E10 | DeepSeek-R1 (full) | L1/L3 | 671B | ~45,000 (est.) | ~6.71e-8 |
| E11 | Qwen 2 (72B) | L1 | 72B | ~50,000 (est.) | ~6.94e-7 |
| E12 | GLM-4 (~130B) | L1 | ~130B (est.) | ~2,000 (est.) | ~1.54e-8 |
| E13 | Command R+ (104B) | L1 | 104B | ~50,000 (est.) | ~4.81e-7 |

**Estimation explanations:**

- **E1 (GPT-4)**: Params ~1.8T from leaked 8×222B MoE analysis. H_spec: RBRM rubrics are "multiple-choice style" classifiers; estimated ~50 safety categories × ~2000 tokens rules = ~200K bits.
- **E2 (GPT-4o)**: Params ~200B from Microsoft MEDEC paper leak. H_spec assumed similar to GPT-4 (same RBRM pipeline).
- **E3 (o1)**: Params ~300B from Microsoft MEDEC. H_spec: "deliberative alignment" uses safety policy documents at inference time; estimated similar scale to GPT-4 rubrics minus some RBRM overhead.
- **E4-E5 (Claude 3/3.5)**: 58 principles published (~4500 tok × 2 bits = ~9000 bits). Params: Opus ~2T (MoE speculation from Life Architect), Sonnet ~175B (Microsoft MEDEC).
- **E6-E7 (Gemini)**: Params estimated from MoE architecture scale. H_spec: "loosely inspired by Constitutional AI" + DRM + Critic rubrics, estimated at ~100K bits.
- **E8-E9 (Llama 3/4)**: Params confirmed (405B/400B). H_spec: labeler guidelines for 4-level preference + editing instructions, estimated ~50K tokens × 2 bits.
- **E10 (DeepSeek-R1 full)**: Params confirmed 671B. Adds helpful RM (66K pairs → ~20K bits criteria) + safety RM (106K prompts → ~20K bits guidelines) + cold-start guidelines (~5K bits).
- **E11 (Qwen 2)**: Same params as Qwen 2.5. H_spec: "constitutional feedback" method mentioned but undisclosed; estimated at ~25K tokens.
- **E12 (GLM-4)**: Params ~130B estimated from model range. 5 annotation dimensions named; rubric estimated ~1000 tokens.
- **E13 (Command R+)**: Params 104B confirmed. H_spec: only "helpfulness and safety" disclosed; estimated at typical industrial guideline scale.

---

## 5. Full Model Survey (30+ Tech Reports)

### 5.1 Transparency Ranking

| Rank | Model | Params | SFT Count | Pref. Pairs | Rubric/Spec | Overall |
|---|---|---|---|---|---|---|
| 1 | **Nemotron-4 340B** | ✓ | ~1M synthetic + 10K human | 160K DPO + 900K RPO | 5-attr rubric (full text) | **Excellent** |
| 2 | **Phi-4 14B** | ✓ | ~8B tokens | ~1.1M (250K+850K) | 3-dim judge rubric | **Excellent** |
| 3 | **Llama 2 70B** | ✓ | 27,540 | 1,418,091 | 4-point scale + guidelines | **Very Good** |
| 4 | **InstructGPT 175B** | ✓ | ~12,725 | ~33K prompts (300K-1.8M) | Labeler guidelines (public) | **Very Good** |
| 5 | **DeepSeek-R1 671B** | ✓ | ~800K | 66K helpful + 106K safety | Rule-based reward (method) | **Good** |
| 6 | **Constitutional AI 52B** | ✓ | ~183K | ~318K RLAIF | 16 principles (public) | **Good** |
| 7 | **Qwen 2.5 72B** | ✓ | >1M | 150K DPO | 4 dimensions named | **Good** |
| 8 | **DeepSeek-V3 671B** | ✓ | 1.5M | N/D | Hybrid RM (method only) | **Medium** |
| 9 | **DeepSeek-V2 236B** | ✓ | 1.5M | N/D | N/D | **Medium** |
| 10 | **Llama 3 405B** | ✓ | ~10M+ | Ratios only | N/D | **Medium** |
| 11 | **Yi 34B** | ✓ | <10K | N/D | N/D | **Low** |
| 12 | **Llama 4 400B** | ✓ | N/D | N/D | N/D | **Low** |
| 13 | **Gemma 2 27B** | ✓ | N/D | N/D | N/D | **Low** |
| 14 | **Falcon 180B** | ✓ | Public datasets | 0 | None | **Low** |
| 15 | **Phi-3 14B** | ✓ | N/D | N/D | RAI categories (method) | **Low** |
| 16 | **Qwen 2 72B** | ✓ | >500K | N/D | Constitutional (method) | **Low** |
| 17 | **GLM-4 ~130B** | ~ | N/D | N/D | 5 dimensions named | **Low** |
| 18 | **Claude 3 Opus** | ✗ | N/D | N/D | 58 principles (public) | **Low** |
| 19 | **Claude 3.5 Sonnet** | ✗ | N/D | N/D | Same constitution | **Low** |
| 20 | **Command R+ 104B** | ✓ | N/D | N/D | "helpfulness and safety" | **Very Low** |
| 21 | **Mistral 7B** | ✓ | N/D | N/D | "No moderation" | **Very Low** |
| 22 | **Mixtral 8x7B** | ✓ | N/D | N/D | N/D | **Very Low** |
| 23 | **GPT-4** | ✗ | N/D | N/D | RBRM (method only) | **Very Low** |
| 24 | **GPT-4o** | ✗ | N/D | N/D | N/D | **Minimal** |
| 25 | **o1** | ✗ | N/D | N/D | Deliberative (method) | **Minimal** |
| 26 | **o3** | ✗ | N/D | N/D | N/D | **Minimal** |
| 27 | **Gemini 1.0 Ultra** | ✗ | N/D | N/D | 3-head RM (method) | **Minimal** |
| 28 | **Gemini 1.5 Pro** | ✗ | N/D | N/D | Defers to 1.0 | **Minimal** |
| 29 | **Gemini 2.5 Pro** | ✗ | N/D | N/D | N/D | **Minimal** |
| — | **LLaMA 1 65B** | ✓ | 0 | 0 | None (base only) | **Complete** |
| — | **Grok-1 314B** | ✓ | 0 | 0 | None (base only) | **Complete** |
| — | **PaLM 2** | ✗ | N/A | N/A | Control tokens (Persp. API) | **Unique** |

✓ = confirmed, ✗ = not disclosed, ~ = partial, N/D = not disclosed

### 5.2 Key Patterns

1. **Open-source ≠ transparent alignment.** Mistral/Mixtral release weights but disclose zero alignment data. Llama 2 remains the gold standard for RLHF transparency.

2. **Industry trend toward opacity.** Llama 2 (2023) disclosed 27,540 SFT + 1.4M preferences. Llama 3 (2024) gave only proportional breakdowns. Llama 4 (2025) disclosed nothing. OpenAI and Google have never disclosed alignment data volumes.

3. **NVIDIA Nemotron-4 is an outlier.** It discloses exact counts at every stage (10K human, 800K synthetic SFT, 160K DPO, 900K RPO) plus the full HelpSteer2 rubric. This is because it was published as a research contribution on synthetic alignment data, not as a product.

4. **Chinese labs disclose more than US labs on SFT counts.** DeepSeek (1.5M), Qwen (>1M), but less on reward/rubric specifications.

5. **Base model releases (LLaMA 1, Grok-1, PaLM 2) have HSD = 0** by construction and serve as important baselines.

---

## 6. The Inverse Equivalence Hypothesis

### 6.1 Statement

For a fixed alignment quality $Q$: $I_H + I_M \approx C(Q)$

Human supervision and AI autonomy are **fungible**.

### 6.2 Evidence For
- **RLAIF ≈ RLHF** (Lee et al. 2024)
- **Self-Rewarding LMs** outperform GPT-4 0613 on AlpacaEval 2.0 (20.44% vs 15.76%) with ~600 bits of rubric
- **DeepSeek-R1-Zero** matches o1 reasoning with ~160 bits of reward
- **SPIN** matches DPO (62K human pairs) with zero additional annotation
- **Nemotron-4**: 98% synthetic data matches human-data-heavy approaches

### 6.3 Evidence Against
- **DeepSeek-R1-Zero** needs SFT + RL with human data for safety
- **Creative/subjective tasks**: No RLVR solution exists
- **Reward hacking** increases with autonomy

### 6.4 Conditional Resolution

| Domain | Equivalence | Evidence |
|---|---|---|
| Verifiable (math, code) | **Strong** | DeepSeek-R1-Zero, RLVR, Nemotron-4 |
| Normative (safety, ethics) | **Partial** | CAI works, R1-Zero drifts |
| Subjective (style, tone) | **Weak/None** | No self-generated aesthetic preferences demonstrated |

---

## 7. Taxonomy Mapping (All Models)

```
Baseline        L1 Scaling           L2 Temporal          L3 Reflection       L4 Co-learning       L5 Real-world
(No align)      (External scale)      (External traces)    (Reflective emu.)   (Dialogical)         (Own maxims)

LLaMA 1         InstructGPT           ProgressGym          Dromedary           Habermas Machine     Albertan Plan
Grok-1          Llama 2/3/4                               SALMON              Generative
PaLM 2*         GPT-4/4o/o1/o3                           Self-Rewarding      Social Choice
                Constitutional AI                        Meta-Rewarding
                Claude 3/3.5/4                           DeepSeek-R1-Zero
                Gemini / Gemma                            GLM-Z1
                Qwen / DeepSeek
                Nemotron / Tulu
                Llama Guard / Aegis
                Mistral / Mixtral
                Yi / Falcon
                GLM-4 / Command
                Phi / ORCA

*PaLM 2 uses control tokens (unique approach)

HSD:  0          ~1e-10 to 1e-6      method-dependent     ~1e-10 to 1e-7    not plottable         not plottable
◄─── No value-learning pathway ───── External specification ───── First-person/autonomous value formation ───►
```

---

## 8. Key Observations

1. **Specification leverage is enormous.** Constitutional AI reshapes 52B parameters with ~3,300 bits — ~16M parameters per bit. Nemotron-4's 5-attribute rubric (~2000 bits) shapes 340B parameters.

2. **The autonomy ladder is not monotonic in benchmark performance.** L3 Reflection systems (Self-Rewarding, SPIN) can outperform many L1 Scaling systems on standard benchmarks.

3. **Transparency is declining.** The most detailed RLHF report (Llama 2, 2023) has no successor in transparency. Every subsequent Llama report disclosed less.

4. **HSD varies by 6 orders of magnitude** across production models: from ~0 (base models, self-modifying systems) to ~1e-6 (small models with heavy rubrics like estimated GPT-4o).

5. **Small models have higher HSD.** The same rubric applied to a 14B model (Phi-4: 4.29e-8) vs a 340B model (Nemotron-4: 5.88e-9) yields ~7x higher HSD. This is an artifact of the normalization — specification effort doesn't scale linearly with parameters.

6. **98% of Nemotron-4's alignment data is synthetic.** Only 20K human examples generated ~2M alignment samples. This represents a "specification amplification factor" of ~100x and suggests the lower bound of human information needed is far below what most labs use.

---

## 9. Open Questions

1. Can an L5 Real-world system achieve L1 Scaling safety performance?
2. Is there a minimum HSD below which safety degrades catastrophically?
3. Does the pre-training corpus set an upper bound on achievable authenticity?
4. How should we account for meta-instructions that enable autonomy without prescribing values?
5. Should the HSD formula weight bits by their causal influence on behavior (a 1000-bit constitution may outweigh 100K bits of preference labels)?
6. Is the declining transparency trend in tech reports a deliberate strategy or an emergent industry norm?
