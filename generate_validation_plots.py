"""
Generate validation plots for feature extraction analysis.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

def load_sequence_lengths():
    """Load sequence lengths from all splits."""
    features_root = Path("data/processed")
    all_lengths = []
    split_lengths = {}

    for split in ['train', 'dev', 'test']:
        split_dir = features_root / split
        files = list(split_dir.glob('*.npy'))

        lengths = []
        for f in files:
            features = np.load(f)
            lengths.append(len(features))

        all_lengths.extend(lengths)
        split_lengths[split] = lengths

    return all_lengths, split_lengths

def generate_plots():
    """Generate comprehensive validation plots."""

    print("Loading sequence length data...")
    all_lengths, split_lengths = load_sequence_lengths()
    all_lengths_array = np.array(all_lengths)

    # Load validation report
    report_file = Path('validation_report.json')
    if report_file.exists():
        with open(report_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = {}

    print(f"Loaded {len(all_lengths)} sequences")

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Feature Extraction Validation Report\nRWTH-PHOENIX-Weather 2014',
                 fontsize=18, fontweight='bold', y=0.98)

    # 1. Sequence length histogram
    ax1 = fig.add_subplot(gs[0, :2])
    counts, bins, patches = ax1.hist(all_lengths, bins=50, edgecolor='black',
                                      alpha=0.7, color='steelblue')
    mean_val = np.mean(all_lengths)
    median_val = np.median(all_lengths)
    ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_val:.1f} frames')
    ax1.axvline(median_val, color='green', linestyle='--', linewidth=2,
                label=f'Median: {median_val:.1f} frames')
    ax1.set_xlabel('Sequence Length (frames)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Sequence Length Distribution (All Splits)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Add statistics text box
    textstr = f'n = {len(all_lengths):,}\nStd = {np.std(all_lengths):.1f}\nMin = {np.min(all_lengths)}\nMax = {np.max(all_lengths)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.97, 0.97, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    # 2. Box plot by split
    ax2 = fig.add_subplot(gs[0, 2])
    split_data = [split_lengths['train'], split_lengths['dev'], split_lengths['test']]
    bp = ax2.boxplot(split_data, labels=['Train', 'Dev', 'Test'],
                     patch_artist=True, notch=True, showmeans=True)

    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax2.set_ylabel('Sequence Length (frames)', fontsize=11, fontweight='bold')
    ax2.set_title('Length Distribution by Split', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Cumulative distribution
    ax3 = fig.add_subplot(gs[1, :2])
    sorted_lengths = np.sort(all_lengths)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
    ax3.plot(sorted_lengths, cumulative, linewidth=2.5, color='navy', label='Cumulative %')

    # Mark truncation thresholds
    thresholds = [150, 200, 250, 300]
    colors_thresh = ['red', 'orange', 'green', 'blue']
    for threshold, color in zip(thresholds, colors_thresh):
        pct = 100 * np.mean(all_lengths_array <= threshold)
        ax3.axhline(pct, color=color, linestyle=':', alpha=0.6, linewidth=1.5)
        ax3.axvline(threshold, color=color, linestyle=':', alpha=0.6, linewidth=1.5)
        ax3.plot(threshold, pct, 'o', color=color, markersize=8)
        ax3.text(threshold + 5, pct - 4, f'{threshold}f: {pct:.1f}%',
                 fontsize=8, color=color, fontweight='bold')

    ax3.set_xlabel('Sequence Length (frames)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Cumulative Percentage (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Cumulative Distribution Function (Truncation Analysis)', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 320])

    # 4. Split comparison - mean and std
    ax4 = fig.add_subplot(gs[1, 2])
    splits = ['train', 'dev', 'test']
    means = [np.mean(split_lengths[s]) for s in splits]
    stds = [np.std(split_lengths[s]) for s in splits]

    x = np.arange(len(splits))
    bars = ax4.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                   color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax4.set_xticks(x)
    ax4.set_xticklabels(splits)
    ax4.set_ylabel('Mean Sequence Length (frames)', fontsize=11, fontweight='bold')
    ax4.set_title('Mean Length by Split (± Std)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. File count validation
    ax5 = fig.add_subplot(gs[2, 0])
    if 'file_counts' in stats:
        splits_data = []
        for s in splits:
            if s in stats['file_counts']:
                splits_data.append({
                    'split': s,
                    'extracted': stats['file_counts'][s]['extracted'],
                    'expected': stats['file_counts'][s]['expected']
                })

        x = np.arange(len(splits_data))
        width = 0.35
        extracted = [d['extracted'] for d in splits_data]
        expected = [d['expected'] for d in splits_data]

        bars1 = ax5.bar(x - width/2, extracted, width, label='Extracted',
                        alpha=0.8, edgecolor='black', color='green')
        bars2 = ax5.bar(x + width/2, expected, width, label='Expected',
                        alpha=0.8, edgecolor='black', color='blue')

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', fontsize=8)

        ax5.set_xticks(x)
        ax5.set_xticklabels([d['split'] for d in splits_data])
        ax5.set_ylabel('Number of Sequences', fontsize=11, fontweight='bold')
        ax5.set_title('Extraction Success Rate', fontsize=13, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3, axis='y')

    # 6. Storage requirements
    ax6 = fig.add_subplot(gs[2, 1])
    if 'storage' in stats:
        storage_data = []
        for s in splits:
            if s in stats['storage']:
                storage_data.append({
                    'split': s,
                    'size_mb': stats['storage'][s]['size_mb']
                })

        x = np.arange(len(storage_data))
        sizes = [d['size_mb'] for d in storage_data]
        bars = ax6.bar(x, sizes, alpha=0.7, edgecolor='black',
                       color=colors, linewidth=1.5)

        # Add value labels
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{size:.1f} MB',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax6.set_xticks(x)
        ax6.set_xticklabels([d['split'] for d in storage_data])
        ax6.set_ylabel('Storage Size (MB)', fontsize=11, fontweight='bold')
        ax6.set_title('Storage Requirements by Split', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')

    # 7. Frame distribution
    ax7 = fig.add_subplot(gs[2, 2])
    if 'file_counts' in stats:
        frame_data = []
        for s in splits:
            if s in stats['file_counts']:
                frame_data.append({
                    'split': s,
                    'frames': stats['file_counts'][s]['frames']
                })

        x = np.arange(len(frame_data))
        frames = [d['frames'] for d in frame_data]
        total_frames = sum(frames)
        percentages = [100 * f / total_frames for f in frames]

        bars = ax7.bar(x, [f/1000 for f in frames], alpha=0.7,
                       edgecolor='black', color=colors, linewidth=1.5)

        # Add value labels
        for bar, frame, pct in zip(bars, frames, percentages):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{frame:,}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax7.set_xticks(x)
        ax7.set_xticklabels([d['split'] for d in frame_data])
        ax7.set_ylabel('Total Frames (thousands)', fontsize=11, fontweight='bold')
        ax7.set_title('Frame Distribution by Split', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    output_file = Path('validation_plots.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {output_file.absolute()}")

    plt.close()

    # Generate a second figure focused on truncation analysis
    print("Generating truncation analysis plot...")
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Truncation Strategy Analysis', fontsize=16, fontweight='bold')

    # Truncation impact
    ax = axes[0]
    thresholds = range(50, 350, 10)
    percentages = [100 * np.mean(all_lengths_array <= t) for t in thresholds]
    sequences_kept = [np.sum(all_lengths_array <= t) for t in thresholds]

    ax.plot(thresholds, percentages, linewidth=2.5, color='darkblue', marker='o', markersize=3)
    ax.axhline(90, color='red', linestyle='--', alpha=0.7, label='90% threshold')
    ax.axhline(95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
    ax.axhline(99, color='green', linestyle='--', alpha=0.7, label='99% threshold')

    # Mark recommended values
    for target_pct, color in [(90, 'red'), (95, 'orange'), (99, 'green')]:
        # Find threshold closest to target
        idx = np.argmin(np.abs(np.array(percentages) - target_pct))
        threshold_val = thresholds[idx]
        actual_pct = percentages[idx]
        ax.plot(threshold_val, actual_pct, 'o', color=color, markersize=10)
        ax.annotate(f'{threshold_val}f\n{actual_pct:.1f}%',
                   xy=(threshold_val, actual_pct),
                   xytext=(threshold_val + 20, actual_pct - 5),
                   fontsize=9, fontweight='bold', color=color)

    ax.set_xlabel('Truncation Threshold (frames)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage of Sequences Kept (%)', fontsize=11, fontweight='bold')
    ax.set_title('Data Retention vs Truncation Threshold', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([50, 350])

    # Sequences discarded
    ax = axes[1]
    sequences_discarded = [len(all_lengths_array) - s for s in sequences_kept]
    ax.plot(thresholds, sequences_discarded, linewidth=2.5, color='darkred', marker='o', markersize=3)

    # Mark recommended values
    recommended = [200, 250, 300]
    for rec in recommended:
        if rec in thresholds:
            idx = thresholds.index(rec)
            val = sequences_discarded[idx]
            ax.plot(rec, val, 'o', color='blue', markersize=10)
            ax.annotate(f'{rec}f:\n{val} lost',
                       xy=(rec, val),
                       xytext=(rec, val + 50),
                       fontsize=9, fontweight='bold', color='blue')

    ax.set_xlabel('Truncation Threshold (frames)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Sequences Truncated', fontsize=11, fontweight='bold')
    ax.set_title('Sequences Affected by Truncation', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([50, 350])

    plt.tight_layout()
    output_file2 = Path('truncation_analysis.png')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"Truncation analysis saved to: {output_file2.absolute()}")

    plt.close()

    print("\nVisualization complete!")
    print(f"\nGenerated plots:")
    print(f"  1. {Path('validation_plots.png').absolute()}")
    print(f"  2. {Path('truncation_analysis.png').absolute()}")

if __name__ == "__main__":
    generate_plots()
