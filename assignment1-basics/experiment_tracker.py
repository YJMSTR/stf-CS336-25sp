#!/usr/bin/env python3
"""
Comprehensive experiment tracking infrastructure for Transformer experiments.
Supports logging to local files, Weights & Biases, and experiment documentation.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

class ExperimentTracker:
    """Enhanced experiment tracker with local and cloud logging capabilities."""
    
    def __init__(self, 
                 experiment_name: str,
                 log_dir: str = "experiment_logs",
                 config: Dict[str, Any] = None,
                 use_wandb: bool = False,
                 wandb_project: str = None,
                 wandb_entity: str = None):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to save local logs
            config: Configuration dictionary
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            wandb_entity: W&B entity name
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.start_datetime = datetime.now()
        self.metrics = defaultdict(list)
        self.step_times = []
        self.wall_times = []
        self.config = config or {}
        self._save_config()
        self.use_wandb = use_wandb
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=experiment_name,
                    config=self.config
                )
            except ImportError:
                print("Warning: wandb not installed. Disabling W&B logging.")
                self.use_wandb = False
        self.log_file = self.experiment_dir / "experiment_log.md"
        self._init_experiment_log()
        self.metrics_file = self.experiment_dir / "metrics.csv"
        
    def _save_config(self):
        """Save configuration to JSON file."""
        config_file = self.experiment_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _init_experiment_log(self):
        """Initialize experiment log markdown file."""
        with open(self.log_file, 'w') as f:
            f.write(f"# Experiment: {self.experiment_name}\n\n")
            f.write(f"**Start Time:** {self.start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Configuration\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.config, indent=2))
            f.write("\n```\n\n")
            f.write("## Experiment Notes\n\n")
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log metrics for a given step.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current training step
        """
        current_time = time.time()
        wall_time = current_time - self.start_time
        self.step_times.append(step)
        self.wall_times.append(wall_time)
        for name, value in metrics.items():
            self.metrics[name].append(value)
        metrics_with_time = {
            'step': step,
            'wall_time': wall_time,
            **metrics
        }
        self._append_to_csv(metrics_with_time)
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.log(metrics, step=step)
    
    def _append_to_csv(self, metrics: Dict[str, Any]):
        """Append metrics to CSV file."""
        import csv
        file_exists = self.metrics_file.exists()
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
    
    def log_note(self, note: str, step: Optional[int] = None):
        """
        Add a note to the experiment log.
        
        Args:
            note: Note to add
            step: Optional step number
        """
        with open(self.log_file, 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if step is not None:
                f.write(f"\n### Step {step} - {timestamp}\n")
            else:
                f.write(f"\n### {timestamp}\n")
            f.write(f"{note}\n")
    
    def save_checkpoint_info(self, checkpoint_path: str, step: int, metrics: Dict[str, float]):
        """
        Log checkpoint information.
        
        Args:
            checkpoint_path: Path to checkpoint file
            step: Training step
            metrics: Current metrics
        """
        note = f"Saved checkpoint: {checkpoint_path}\n"
        note += "Metrics:\n"
        for name, value in metrics.items():
            note += f"- {name}: {value:.4f}\n"
        self.log_note(note, step)
    
    def plot_metrics(self, metric_names: Optional[List[str]] = None, save_plots: bool = True):
        """
        Plot metrics vs steps and wall time.
        
        Args:
            metric_names: List of metric names to plot (None for all)
            save_plots: Whether to save plots to disk
        """
        if metric_names is None:
            metric_names = list(self.metrics.keys())
        num_metrics = len(metric_names)
        fig, axes = plt.subplots(num_metrics, 2, figsize=(15, 5 * num_metrics))
        if num_metrics == 1:
            axes = axes.reshape(1, -1)
        for i, metric_name in enumerate(metric_names):
            if metric_name not in self.metrics:
                continue
            values = self.metrics[metric_name]
            axes[i, 0].plot(self.step_times[:len(values)], values)
            axes[i, 0].set_xlabel('Steps')
            axes[i, 0].set_ylabel(metric_name)
            axes[i, 0].set_title(f'{metric_name} vs Steps')
            axes[i, 0].grid(True)
            axes[i, 1].plot(self.wall_times[:len(values)], values)
            axes[i, 1].set_xlabel('Wall Time (seconds)')
            axes[i, 1].set_ylabel(metric_name)
            axes[i, 1].set_title(f'{metric_name} vs Wall Time')
            axes[i, 1].grid(True)
        plt.tight_layout()
        if save_plots:
            plot_path = self.experiment_dir / 'metrics_plots.png'
            plt.savefig(plot_path, dpi=150)
            self.log_note(f"Saved metrics plots to {plot_path}")
        return fig
    
    def generate_summary(self):
        """Generate experiment summary."""
        end_time = time.time()
        total_time = end_time - self.start_time
        summary = f"\n## Experiment Summary\n\n"
        summary += f"**End Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += f"**Total Duration:** {total_time:.2f} seconds ({total_time/3600:.2f} hours)\n"
        summary += f"**Total Steps:** {self.step_times[-1] if self.step_times else 0}\n\n"
        if self.metrics:
            summary += "### Final Metrics\n\n"
            for name, values in self.metrics.items():
                if values:
                    summary += f"- {name}: {values[-1]:.4f}\n"
            summary += "\n### Best Metrics\n\n"
            for name, values in self.metrics.items():
                if values and 'loss' in name.lower():
                    best_val = min(values)
                    best_idx = values.index(best_val)
                    best_step = self.step_times[best_idx] if best_idx < len(self.step_times) else 'N/A'
                    summary += f"- Best {name}: {best_val:.4f} (at step {best_step})\n"
        with open(self.log_file, 'a') as f:
            f.write(summary)
        if self.metrics:
            self.plot_metrics()
        return summary
    
    def close(self):
        """Close the experiment tracker."""
        self.generate_summary()
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.finish()


class ExperimentComparison:
    """Tool for comparing multiple experiments."""
    
    def __init__(self, log_dir: str = "experiment_logs"):
        """
        Initialize experiment comparison tool.
        
        Args:
            log_dir: Directory containing experiment logs
        """
        self.log_dir = Path(log_dir)
    
    def load_experiment_metrics(self, experiment_name: str) -> pd.DataFrame:
        """Load metrics from an experiment."""
        metrics_file = self.log_dir / experiment_name / "metrics.csv"
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
        return pd.read_csv(metrics_file)
    
    def compare_experiments(self, experiment_names: List[str], metric_name: str):
        """
        Compare a specific metric across multiple experiments.
        
        Args:
            experiment_names: List of experiment names to compare
            metric_name: Name of the metric to compare
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        for exp_name in experiment_names:
            try:
                df = self.load_experiment_metrics(exp_name)
                if metric_name in df.columns:
                    ax1.plot(df['step'], df[metric_name], label=exp_name)
                    ax2.plot(df['wall_time'], df[metric_name], label=exp_name)
            except FileNotFoundError:
                print(f"Warning: Could not load metrics for {exp_name}")
        ax1.set_xlabel('Steps')
        ax1.set_ylabel(metric_name)
        ax1.set_title(f'{metric_name} vs Steps')
        ax1.legend()
        ax1.grid(True)
        ax2.set_xlabel('Wall Time (seconds)')
        ax2.set_ylabel(metric_name)
        ax2.set_title(f'{metric_name} vs Wall Time')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        return fig
    
    def generate_comparison_report(self, experiment_names: List[str], output_file: str = "comparison_report.md"):
        """Generate a markdown report comparing experiments."""
        report = "# Experiment Comparison Report\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += "## Configurations\n\n"
        configs = {}
        for exp_name in experiment_names:
            config_file = self.log_dir / exp_name / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    configs[exp_name] = json.load(f)
        if configs:
            report += "### Key Hyperparameters\n\n"
            all_keys = set()
            for config in configs.values():
                all_keys.update(config.keys())
            report += "| Parameter | " + " | ".join(experiment_names) + " |\n"
            report += "|-----------|" + "|".join(["---------"] * len(experiment_names)) + "|\n"
            for key in sorted(all_keys):
                values = [str(configs.get(exp, {}).get(key, 'N/A')) for exp in experiment_names]
                report += f"| {key} | " + " | ".join(values) + " |\n"
        report += "\n## Final Metrics\n\n"
        report += "| Metric | " + " | ".join(experiment_names) + " |\n"
        report += "|--------|" + "|".join(["---------"] * len(experiment_names)) + "|\n"
        all_metrics = set()
        final_metrics = {}
        for exp_name in experiment_names:
            try:
                df = self.load_experiment_metrics(exp_name)
                last_row = df.iloc[-1]
                final_metrics[exp_name] = last_row.to_dict()
                all_metrics.update(last_row.index)
            except:
                final_metrics[exp_name] = {}
        for metric in sorted(all_metrics):
            if metric in ['step', 'wall_time']:
                continue
            values = []
            for exp_name in experiment_names:
                val = final_metrics.get(exp_name, {}).get(metric, 'N/A')
                if isinstance(val, float):
                    values.append(f"{val:.4f}")
                else:
                    values.append(str(val))
            report += f"| {metric} | " + " | ".join(values) + " |\n"
        with open(self.log_dir / output_file, 'w') as f:
            f.write(report)
        return report
