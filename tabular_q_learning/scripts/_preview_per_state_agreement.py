"""Preview the new per_state_agreement styling using values from the existing figure."""
import os
import numpy as np
import matplotlib.pyplot as plt

row_labels = ['ID (Beta 2,2)', 'Deterministic trans.', 'Sparse rewards',
              'Dense rewards', 'Adversarial', 'High-variance']
bin_labels = ['t=1-10', 't=11-20', 't=21-30', 't=31-40', 't=41-50']
agree_matrix = np.array([
    [0.62, 0.88, 0.94, 0.90, 0.95],
    [0.73, 0.89, 0.89, 0.96, 0.96],
    [0.60, 0.84, 0.92, 0.93, 0.93],
    [0.65, 0.87, 0.92, 0.94, 0.95],
    [0.78, 0.89, 0.92, 0.93, 0.95],
    [0.56, 0.88, 0.89, 0.92, 0.90],
])

n_dists, n_bins = agree_matrix.shape
fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(agree_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Greedy action agreement', fontsize=16)
cbar.ax.tick_params(labelsize=14)
ax.set_xticks(range(n_bins))
ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=15)
ax.set_yticks(range(n_dists))
ax.set_yticklabels(row_labels, fontsize=15)
ax.set_xlabel('Episode timestep bin', fontsize=17)
ax.set_ylabel('Reward distribution', fontsize=17)
ax.tick_params(axis='both', labelsize=15)
ax.set_title('Action Agreement by Reward Distribution and Timestep Phase',
             fontsize=18)
cell_fontsize = 18
for r in range(n_dists):
    for b in range(n_bins):
        val = agree_matrix[r, b]
        ax.text(b, r, f'{val:.2f}', ha='center', va='center',
                fontsize=cell_fontsize,
                color='black' if 0.3 < val < 0.85 else 'white')
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'figures',
                   'per_state_agreement_preview.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
