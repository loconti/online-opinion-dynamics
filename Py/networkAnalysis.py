import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid display issues
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define paths
data_dir = Path(__file__).parent.parent / "Data"
comments_csv = data_dir / "diffusions_comments_labeled.csv"
submissions_csv = data_dir / "diffusions_submissions_labeled.csv"

# Read the labeled CSV files
print("Reading labeled data files...")
comments_df = pd.read_csv(comments_csv)
submissions_df = pd.read_csv(submissions_csv)

print(f"Comments shape: {comments_df.shape}")
print(f"Submissions shape: {submissions_df.shape}")

# Display column names
print("\nComments columns:", comments_df.columns.tolist())
print("Submissions columns:", submissions_df.columns.tolist())

# Categorize Pro vs Anti based on stance column
# Mapping stance to Pro/Anti categories
def categorize_stance(stance):
    if 'Pro' in stance:
        return 'Pro'
    elif 'Against' in stance:
        return 'Anti'
    else:
        return 'Neutral'

# Apply categorization
comments_df['category'] = comments_df['stance'].apply(categorize_stance)
submissions_df['category'] = submissions_df['stance'].apply(categorize_stance)

# Get statistics for Pro vs Anti
print("\n=== Comments Statistics ===")
comments_stats = comments_df['category'].value_counts()
print(comments_stats)

print("\n=== Submissions Statistics ===")
submissions_stats = submissions_df['category'].value_counts()
print(submissions_stats)

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Comments Pro vs Anti
categories_c = comments_stats.index.tolist()
values_c = comments_stats.values.tolist()
colors_c = ['#2ecc71' if c == 'Pro' else '#e74c3c' if c == 'Anti' else '#95a5a6' for c in categories_c]

axes[0].bar(categories_c, values_c, color=colors_c, edgecolor='black', linewidth=1.5)
axes[0].set_title('Comments: Pro vs Anti', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_xlabel('Stance Category', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (cat, val) in enumerate(zip(categories_c, values_c)):
    axes[0].text(i, val, str(val), ha='center', va='bottom', fontweight='bold')

# Plot 2: Submissions Pro vs Anti
categories_s = submissions_stats.index.tolist()
values_s = submissions_stats.values.tolist()
colors_s = ['#2ecc71' if c == 'Pro' else '#e74c3c' if c == 'Anti' else '#95a5a6' for c in categories_s]

axes[1].bar(categories_s, values_s, color=colors_s, edgecolor='black', linewidth=1.5)
axes[1].set_title('Submissions: Pro vs Anti', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_xlabel('Stance Category', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (cat, val) in enumerate(zip(categories_s, values_s)):
    axes[1].text(i, val, str(val), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(data_dir / "pro_vs_anti_comparison.png", dpi=300, bbox_inches='tight')
print("\n✓ Bar graph saved to: Data/pro_vs_anti_comparison.png")
