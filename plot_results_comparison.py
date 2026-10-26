import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Load MMLU results ──
with open('snapshots/rexp_8b_20260420_184914_eval.json') as f:
    mmlu_const = json.load(f)

mmlu_sft = {}
for cond in ['wr_neg','wr_pos','wr_neu','wor_neg','wor_pos','wor_neu']:
    with open(f'snapshots/sft_8b_{cond}_eval.json') as f:
        d = json.load(f)
    mmlu_sft[cond] = d['matrix'].get(cond, {})

# ── Load MATH results ──
with open('snapshots/rexp_8b_math_20260422_152408_eval.json') as f:
    math_const = json.load(f)

math_sft = {}
for cond in ['wr_neg','wr_pos','wr_neu','wor_neg','wor_pos','wor_neu']:
    with open(f'snapshots/sft_8b_math_{cond}_eval.json') as f:
        d = json.load(f)
    math_sft[cond] = d['matrix'].get(cond, {})

# ── Setup ──
conds_6 = ['wr_neg', 'wr_pos', 'wr_neu', 'wor_neg', 'wor_pos', 'wor_neu']
cond_labels = {
    'wr_neg': 'w/R+Neg', 'wr_pos': 'w/R+Pos', 'wr_neu': 'w/R+Neu',
    'wor_neg': 'w/oR+Neg', 'wor_pos': 'w/oR+Pos', 'wor_neu': 'w/oR+Neu',
    'none': 'Baseline'
}
conds_7 = conds_6 + ['none']

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('8B Model: MMLU (knowledge recall) vs MATH (multi-step reasoning)', fontsize=16, fontweight='bold')

# ── Panel 1: MMLU vs MATH reasoning scores (constitution) ──
ax = axes[0, 0]
x = np.arange(len(conds_7))
mmlu_r = [mmlu_const['matrix'][c]['reasoning'] for c in conds_7]
math_r = [math_const['matrix'][c]['math_reasoning'] for c in conds_7]
width = 0.35
ax.bar(x - width/2, mmlu_r, width, label='MMLU (knowledge)', color='#2196F3', alpha=0.85)
ax.bar(x + width/2, math_r, width, label='MATH (reasoning)', color='#FF5722', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([cond_labels[c] for c in conds_7], fontsize=8, rotation=20)
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 0.75)
ax.set_title('Constitution: MMLU vs MATH Scores', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
# Add variance annotations
mmlu_std = np.std(mmlu_r)
math_std = np.std(math_r)
ax.text(0.02, 0.95, f'MMLU std={mmlu_std:.4f}\nMATH std={math_std:.4f}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ── Panel 2: MMLU vs MATH reasoning scores (SFT) ──
ax = axes[0, 1]
x = np.arange(len(conds_6))
mmlu_sft_r = [mmlu_sft[c].get('reasoning') or 0 for c in conds_6]
math_sft_r = [math_sft[c].get('math_reasoning') or 0 for c in conds_6]
ax.bar(x - width/2, mmlu_sft_r, width, label='MMLU (knowledge)', color='#2196F3', alpha=0.85)
ax.bar(x + width/2, math_sft_r, width, label='MATH (reasoning)', color='#FF5722', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([cond_labels[c] for c in conds_6], fontsize=8, rotation=20)
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 0.75)
ax.set_title('SFT: MMLU vs MATH Scores', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
mmlu_sft_std = np.std(mmlu_sft_r)
math_sft_std = np.std(math_sft_r)
ax.text(0.02, 0.95, f'MMLU std={mmlu_sft_std:.4f}\nMATH std={math_sft_std:.4f}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ── Panel 3: Constitution vs SFT for MATH reasoning ──
ax = axes[0, 2]
x = np.arange(len(conds_6))
const_math = [math_const['matrix'][c]['math_reasoning'] for c in conds_6]
sft_math = [math_sft[c].get('math_reasoning') or 0 for c in conds_6]
ax.bar(x - width/2, const_math, width, label='Constitution', color='#4CAF50', alpha=0.85)
ax.bar(x + width/2, sft_math, width, label='SFT', color='#F44336', alpha=0.85)
ax.axhline(y=math_const['matrix']['none']['math_reasoning'], color='gray', linestyle='--', alpha=0.7, label='Baseline')
ax.set_xticks(x)
ax.set_xticklabels([cond_labels[c] for c in conds_6], fontsize=8, rotation=20)
ax.set_ylabel('MATH Accuracy')
ax.set_ylim(0, 0.3)
ax.set_title('MATH: Constitution vs SFT', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# ── Panel 4: MATH heatmap (constitution, all benchmarks) ──
ax = axes[1, 0]
benchmarks = ['math_reasoning', 'neg_value', 'pos_value', 'neu_value']
bench_labels = ['MATH', 'Neg Value', 'Pos Value', 'Neu Value']
matrix = []
for c in conds_7:
    row = [math_const['matrix'][c].get(b) or 0 for b in benchmarks]
    matrix.append(row)
matrix = np.array(matrix)
im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(bench_labels)))
ax.set_xticklabels(bench_labels, fontsize=10)
ax.set_yticks(range(len(conds_7)))
ax.set_yticklabels([cond_labels[c] for c in conds_7], fontsize=10)
for i in range(len(conds_7)):
    for j in range(len(benchmarks)):
        ax.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center', fontsize=9,
                color='white' if matrix[i,j] < 0.3 or matrix[i,j] > 0.85 else 'black')
ax.set_title('MATH Experiment: Constitution Eval', fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)

# ── Panel 5: Variance comparison (MMLU vs MATH across conditions) ──
ax = axes[1, 1]
categories = ['Const.\nReasoning', 'SFT\nReasoning', 'Const.\nNeg Value', 'Const.\nPos Value', 'Const.\nNeu Value']
mmlu_vars = [
    np.std([mmlu_const['matrix'][c]['reasoning'] for c in conds_6]),
    np.std(mmlu_sft_r),
    np.std([mmlu_const['matrix'][c]['neg_value'] for c in conds_6]),
    np.std([mmlu_const['matrix'][c]['pos_value'] for c in conds_6]),
    np.std([mmlu_const['matrix'][c]['neu_value'] for c in conds_6]),
]
math_vars = [
    np.std([math_const['matrix'][c]['math_reasoning'] for c in conds_6]),
    np.std(math_sft_r),
    np.std([math_const['matrix'][c]['neg_value'] for c in conds_6]),
    np.std([math_const['matrix'][c]['pos_value'] for c in conds_6]),
    np.std([math_const['matrix'][c]['neu_value'] for c in conds_6]),
]
x = np.arange(len(categories))
ax.bar(x - width/2, mmlu_vars, width, label='MMLU experiment', color='#2196F3', alpha=0.85)
ax.bar(x + width/2, math_vars, width, label='MATH experiment', color='#FF5722', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=8)
ax.set_ylabel('Std Dev (across 6 conditions)')
ax.set_title('Cross-condition Variance: MMLU vs MATH', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# ── Panel 6: Bias scores comparison ──
ax = axes[1, 2]
mmlu_bias = [mmlu_const['bias_scores'].get(c, 0) or 0 for c in conds_6]
math_bias = [math_const['bias_scores'].get(c, 0) or 0 for c in conds_6]
x = np.arange(len(conds_6))
ax.bar(x - width/2, mmlu_bias, width, label='MMLU experiment', color='#2196F3', alpha=0.85)
ax.bar(x + width/2, math_bias, width, label='MATH experiment', color='#FF5722', alpha=0.85)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([cond_labels[c] for c in conds_6], fontsize=8, rotation=20)
ax.set_ylabel('Bias Score')
ax.set_title('Bias Scores: MMLU vs MATH Experiments', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('experiment_results_mmlu_vs_math.png', dpi=150, bbox_inches='tight')
print('Saved: experiment_results_mmlu_vs_math.png')
