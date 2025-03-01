import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import auc
from pathlib import Path
import json

sns.set_style("darkgrid")

language_colors = {
    'Bengali': '#1f77b4',
    'English': '#2ca02c',
    'Japanese': '#8c564b',
    'Yoruba': '#9467bd',
    'Indonesian': '#d62728',
    'Swahili': '#ff7f0e'
}

models_map = {
    "aya-expanse-8b": ["Aya Expanse 8B", "-", "o"],
    "gemma-2-9b-it": ["Gemma 2 9B", ":", "^"],
    "gemma-2-2b-it": ["Gemma 2 2B", "--", "s"],
    "Qwen2.5-7B-Instruct": ["Qwen2.5 7B", "-.", "X"],
    "DeepSeek-R1-Distill-Llama-8B": ["DeepSeek-R1 Llama 8B", (0, (3, 1, 1, 1, 1, 1)), "P"],
    "gemma-2-27b-it-gptq-int4": ["Gemma 2 27B (int4)", (0, (3, 1, 1, 1, 1, 1)), "*"]
}


def load_all_results(base_dir, models):
    results_data = {}
    for target_lang_dir in Path(base_dir).iterdir():
        if not target_lang_dir.is_dir():
            continue
        results_dir = target_lang_dir / 'results'
        if not results_dir.exists():
            continue

        target_lang = target_lang_dir.name
        results_data[target_lang] = {}

        for result_file in results_dir.glob('*.json'):
            split = result_file.stem.rsplit('_')
            model_name = split[2]
            if model_name not in models:
                continue

            with open(result_file, 'r') as f:
                data = json.load(f)

            if model_name not in results_data[target_lang]:
                results_data[target_lang][model_name] = {}

            task_type = split[0]
            runs = data['runs']
            results_data[target_lang][model_name][task_type] = {
                'confidence_cutoffs': [run['confidence_cutoff'] for run in runs],
                'effective_accuracies': [run['metrics']['abstention_metrics']['effective_accuracy'] for run in runs],
                'answered_accuracies': [run['metrics']['abstention_metrics']['answered_accuracy'] for run in runs],
                'abstention_rates': [run['metrics']['abstention_metrics']['abstention_rate'] for run in runs],
                'correctly_abstained_rates': [run['metrics']['abstention_metrics']['correctly_abstained_rate'] for run
                                              in runs],
                'confidence_histograms': [run['metrics']['confidence_histogram'] for run in runs]
            }

    return results_data


def plot_composite_accuracy_comparison(results_data, task_types=['low_res', 'mid_res', 'high_res']):
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 24)
    axes = [fig.add_subplot(gs[0, 0:11]), fig.add_subplot(gs[0, 13:]), fig.add_subplot(gs[1, 6:17]),
            fig.add_subplot(gs[1, 19:])]

    plot_lines, models = [], []
    lang = results_data.keys()

    for idx, task_type in enumerate(task_types):
        for target_lang, lang_data in results_data.items():
            lines = []
            for model, model_data in lang_data.items():
                if task_type in model_data:
                    data = model_data[task_type]
                    l, = axes[idx].plot(
                        data['confidence_cutoffs'],
                        data['effective_accuracies'],
                        color=language_colors[target_lang],
                        ls=models_map[model][1]
                    )
                    lines.append(l)
                    models.append(models_map[model][0])
            plot_lines.append(lines)

        axes[idx].set_xlabel('Confidence Cutoff')
        axes[idx].set_ylabel('Effective Accuracy')
        axes[idx].grid(True)
        axes[idx].set_title(f'{task_type.split("_")[0].capitalize()} Resource')

    legend1 = axes[3].legend(plot_lines[0], models, bbox_to_anchor=(0.3, 0.7), loc="center")
    axes[3].axis('off')
    axes[3].legend([l[0] for l in plot_lines], lang, bbox_to_anchor=(0.3, 0.35), loc="center")
    plt.gca().add_artist(legend1)
    plt.tight_layout()
    plt.savefig('composite_accuracy_comparison.png')
    plt.close()


def plot_coverage_accuracy_curves(results_data, task_types=['low_res', 'mid_res', 'high_res']):
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 24)
    axes = [fig.add_subplot(gs[0, 0:11]), fig.add_subplot(gs[0, 13:]), fig.add_subplot(gs[1, 6:17]),
            fig.add_subplot(gs[1, 19:])]

    plot_lines = []
    models = []
    lang = results_data.keys()

    for idx, task_type in enumerate(task_types):
        for target_lang, lang_data in results_data.items():
            lines = []
            for model, model_data in lang_data.items():
                if task_type in model_data:
                    data = model_data[task_type]
                    coverage = 1 - np.array(data['abstention_rates'])
                    points = np.array(list(zip(coverage, data['answered_accuracies'])))
                    sorted_points = points[points[:, 0].argsort()]

                    l, = axes[idx].plot(
                        sorted_points[:, 0],
                        sorted_points[:, 1],
                        color=language_colors[target_lang],
                        marker=models_map[model][2],
                        linewidth=0.5,
                        markersize=5
                    )
                    lines.append(l)
                    models.append(models_map[model][0])
            plot_lines.append(lines)

        axes[idx].set_xlabel('Coverage (1 - Abstention Rate)')
        axes[idx].set_ylabel('Accuracy')
        axes[idx].grid(True)
        axes[idx].set_title(f'{task_type.split("_")[0].capitalize()} Resource')

    legend1 = axes[3].legend(plot_lines[0], models, bbox_to_anchor=(0.3, 0.7), loc="center")
    axes[3].axis('off')
    axes[3].legend([l[0] for l in plot_lines], lang, bbox_to_anchor=(0.3, 0.35), loc="center")
    plt.gca().add_artist(legend1)
    plt.tight_layout()
    plt.savefig('coverage_accuracy_curves.png')
    plt.close()


def calculate_auc_metrics(results_data):
    auc_metrics = {}
    for target_lang, lang_data in results_data.items():
        auc_metrics[target_lang] = {}
        for model, model_data in lang_data.items():
            auc_metrics[target_lang][model] = {}
            for task_type, task_data in model_data.items():
                coverage = 1 - np.array(task_data['abstention_rates'])
                sort_idx = np.argsort(coverage)
                coverage_sorted = coverage[sort_idx]
                accuracies_sorted = np.array(task_data['answered_accuracies'])[sort_idx]
                auc_score = auc(coverage_sorted, accuracies_sorted)
                auc_metrics[target_lang][model][task_type] = auc_score
                results_data[target_lang][model][task_type]['coverage'] = coverage.tolist()
                results_data[target_lang][model][task_type]['auc_score'] = auc_score
    return auc_metrics
