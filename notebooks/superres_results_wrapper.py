# %%
import os
import sys
sys.path.insert(0, '../')
from qgan_superres_results_runner import RCPS_Results_Runner

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set_theme()
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
sns.set_style("white")

from PIL import Image
import torch

import pandas as pd
import all_utils

%load_ext autoreload
%autoreload 2




# %%
base_dir = 'assets'
exp_name = 'super_resolution'
model_name = 'models/superres_alpha_0.1.pt'

resize_factors = [1, 16, 32]
difficulty_levels = ['easy', 'medium', 'hard']
results_obj = RCPS_Results_Runner(base_dir=base_dir, exp_name=exp_name, model_name=model_name, resize_factors=resize_factors, difficulty_levels=difficulty_levels, norm_scheme='mean')

# %%
results_obj.compute_losses_prediction_sets()

# %%
results_obj.calibrate_all_difficulty_levels(total_runs=100)

# %% [markdown]
# ## Set sizes across resolutions

# %%
set_sizes_plot = results_obj.plot_set_sizes()
plt.gcf().set_size_inches(8, 6)

# %% [markdown]
# ## Residuals across resolutions

# %%
residuals_plot = results_obj.plot_residuals()
plt.gcf().set_size_inches(8, 6)

# %% [markdown]
# ## Empirical risk before/after calibration

# %%
test_idx_list = results_obj.test_idx_list[results_obj.difficulty_levels[0]]
calibrated_risks = results_obj.all_rcps_stats[results_obj.difficulty_levels[0]]['mean_emp_risks']
uncalibrated_risks = []

plt.figure(figsize=(10, 8))
for test_idx_per_run in test_idx_list:
    # Find test indices that overlap with all difficulty
    overlap_indices = []
    for idx, masking_grade in enumerate(results_obj.difficulty_levels):
        all_indices_this_difficulty = results_obj.indices_per_difficulty_level[masking_grade]
        overlap_indices.extend(list(set(all_indices_this_difficulty).intersection(set(test_idx_per_run))))
    uncalibrated_risks.append(results_obj.all_losses_per_lambda[masking_grade][0][np.where(results_obj.lambda_values == 1)[0][0]][overlap_indices, :].mean())
        
        
print(len(uncalibrated_risks), len(calibrated_risks))
        
bins = np.linspace(0.0, 0.5, 100)

plt.hist(uncalibrated_risks, color='r',  bins=bins, label='Before', alpha=0.7)
n, bins, patches = plt.hist(calibrated_risks, color='g', bins=bins, label='After', alpha=0.7)

plt.axvline(x=0.1, linewidth=1, color='k', linestyle='--')
plt.xlim([0.05, 0.2])
plt.ylim([0, 60])
plt.gca().axes.yaxis.set_ticklabels([])
plt.gcf().set_size_inches(8, 6)
all_utils.decorate_plot(
    plt.gca(), xlabel='Risk', ylabel='Histogram density', title_str='FFHQ Super-Resolution: Empirical risk before/after calibration', xticks=np.asarray([0.05, 0.10, 0.15, 0.20, 0.25]))

# %% [markdown]
# ## Downsampling by 1x

# %%
# run_index=-2,  image_num=308 / argsorts[22], model: real_generated_mixed_with_image_mse_alpha_{0.1}_resume , 800000
image_num = 10
run_index = -1
img_desc = f'sample_image'

image_index= results_obj.test_idx_list['easy'][run_index][image_num]
plotted_fig, _ = results_obj.show_outputs_across_difficulty_levels(difficulty_level='easy', image_index=image_index, run_index=run_index)
display(Image.fromarray(plotted_fig))

# %% [markdown]
# ## Downsampling by 32x

# %%
plotted_fig, _ = results_obj.show_outputs_across_difficulty_levels(difficulty_level='medium', image_index=image_index,run_index=run_index)
display(Image.fromarray(plotted_fig))

# %%
plotted_fig, factor_array_hard = results_obj.show_outputs_across_difficulty_levels(difficulty_level='hard', image_index=image_index,run_index=run_index)
display(Image.fromarray(plotted_fig))

# %%
display(Image.fromarray(np.vstack(factor_array_hard)))

# %%
difficulty_level = 'easy'
out4 = results_obj.visualize_style_space_intervals(difficulty_level=difficulty_level, image_index=image_index, plot_type='calibration_interval')
plt.savefig(f'factor_intervals_{difficulty_level}_calibration_interval.png', dpi=300, bbox_inches='tight')

# %%



