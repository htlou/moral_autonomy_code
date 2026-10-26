import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- Load constitution eval ---
with open('snapshots/rexp_8b_20260420_184914_eval.json') as f:
    const_eval = json.load(f)

# --- Load SFT eval ---
sft_scores = {}
sft_bias = {}
for cond in ['wr_neg','wr_pos','wr_neu','wor_neg','wor_pos','wor_neu']:
    with open(f'snapshots/sft_8b_{cond}_eval.json') as f:
        d = json.load(f)
    m = d['matrix'].get(cond, {})
    sft_scores[cond] = m
    b = d.get('bias_scores', {}).get(cond)
    sft_bias[cond] = b

conditions = ['wr_neg', 'wr_pos', 'wr_neu', 'wor_neg', 'wor_pos', 'wor_neu', 'none']
benchmarks = ['reasoning', 'neg_value', 'pos_value', 'neu_value']
bench_labels = ['Reasoning', 'Neg Value', 'Pos Value', 'Neu Value']
cond_labels = {
    'wr_neg': 'w/ R + Neg', 'wr_pos': 'w/ R + Pos', 'wr_neu': 'w/ R + Neu',
    'wor_neg': 'w/o R + Neg', 'wor_pos': 'w/o R + Pos', 'wor_neu': 'w/o R + Neu',
    'none': 'None (baseline)'
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('8B Model Experiment Results — Constitution vs SFT', fontsize=16, fontweight='bold')

# --- Panel 1: Constitution eval heatmap ---
ax = axes[0, 0]
matrix = []
for c in conditions:
    row = [const_eval['matrix'][c].get(b, 0) or 0 for b in benchmarks]
    matrix.append(row)
matrix = np.array(matrix)
im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(bench_labels)))
ax.set_xticklabels(bench_labels, fontsize=10)
ax.set_yticks(range(len(conditions)))
ax.set_yticklabels([cond_labels[c] for c in conditions], fontsize=10)
for i in range(len(conditions)):
    for j in range(len(benchmarks)):
        ax.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center', fontsize=9,
                color='white' if matrix[i,j] < 0.4 or matrix[i,j] > 0.85 else 'black')
ax.set_title('Constitution Eval Scores', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)

# --- Panel 2: Constitution eval grouped bar ---
ax = axes[0, 1]
x = np.arange(len(benchmarks))
width = 0.12
for i, c in enumerate(conditions):
    vals = [const_eval['matrix'][c].get(b, 0) or 0 for b in benchmarks]
    ax.bar(x + i * width, vals, width, label=cond_labels[c], alpha=0.85)
ax.set_xticks(x + width * 3)
ax.set_xticklabels(bench_labels, fontsize=10)
ax.set_ylabel('Score', fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title('Constitution Eval by Benchmark', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc='upper left')
ax.grid(axis='y', alpha=0.3)

# --- Panel 3: Constitution vs SFT comparison (reasoning + neu_value) ---
ax = axes[1, 0]
conds_6 = ['wr_neg', 'wr_pos', 'wr_neu', 'wor_neg', 'wor_pos', 'wor_neu']
x = np.arange(len(conds_6))
width = 0.2

const_reasoning = [const_eval['matrix'][c]['reasoning'] for c in conds_6]
sft_reasoning = [sft_scores[c].get('reasoning') or 0 for c in conds_6]
const_neu = [const_eval['matrix'][c]['neu_value'] for c in conds_6]
sft_neu = [sft_scores[c].get('neu_value') or 0 for c in conds_6]

ax.bar(x - 1.5*width, const_reasoning, width, label='Const. Reasoning', color='#2196F3', alpha=0.8)
ax.bar(x - 0.5*width, sft_reasoning, width, label='SFT Reasoning', color='#2196F3', alpha=0.4, hatch='//')
ax.bar(x + 0.5*width, const_neu, width, label='Const. Neu Value', color='#FF9800', alpha=0.8)
ax.bar(x + 1.5*width, sft_neu, width, label='SFT Neu Value', color='#FF9800', alpha=0.4, hatch='//')

ax.set_xticks(x)
ax.set_xticklabels([cond_labels[c] for c in conds_6], fontsize=9, rotation=15)
ax.set_ylabel('Score', fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_title('Constitution vs SFT: Reasoning & Neu Value', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

# --- Panel 4: Bias scores ---
ax = axes[1, 1]
const_bias = [const_eval['bias_scores'].get(c, 0) or 0 for c in conds_6]
sft_b = [sft_bias.get(c) or 0 for c in conds_6]

x = np.arange(len(conds_6))
width = 0.35
bars1 = ax.bar(x - width/2, const_bias, width, label='Constitution', color='#4CAF50', alpha=0.8)
bars2 = ax.bar(x + width/2, sft_b, width, label='SFT', color='#F44336', alpha=0.8)

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([cond_labels[c] for c in conds_6], fontsize=9, rotation=15)
ax.set_ylabel('Bias Score', fontsize=11)
ax.set_title('Bias Scores: Constitution vs SFT', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('experiment_results_8b.png', dpi=150, bbox_inches='tight')
print('Saved: experiment_results_8b.png')
