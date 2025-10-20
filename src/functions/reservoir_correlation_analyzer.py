import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

class ReservoirCorrelationAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df_selected = None
        self.corr_matrix = None

        # Base features + targets
        self.base_features = [
            "reservoir_pressure_initial (psi)",
            "bottomhole_pressure_psi (psi)",
            "permeability_md (mD)",
            "net_pay_thickness_ft (ft)",
            "bubble_point_pressure (psi)",
            "porosity_percent (%)",
            "choke_size_percent (%)",
        ]

        self.targets = [
            "oil_rate_bopd (BOPD)",
            "gas_rate_mscf_day (MSCF/day)",
            "water_rate_bwpd (BWPD)"
        ]

        # Use all together by default
        self.selected_features = self.base_features + self.targets

    def preprocess(self):
        # Keep only selected features that exist in dataset
        self.df_selected = self.df[[c for c in self.selected_features if c in self.df.columns]].copy()
        self.df_selected = self.df_selected.dropna()

    def calculate_correlation(self):
        if self.df_selected is None:
            self.preprocess()
        self.corr_matrix = self.df_selected.corr()
        return self.corr_matrix

    def save_correlation_excel(self, filename="./src/plot/cor/reservoir_correlation_matrix.xlsx"):
        if self.corr_matrix is None:
            self.calculate_correlation()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.corr_matrix.to_excel(filename)
        print(f"Correlation matrix saved to {filename}")

    def plot_heatmap(self, filename="./src/plot/cor/reservoir_heatmap.png", mask=False):
        if self.corr_matrix is None:
            self.calculate_correlation()
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        plt.figure(figsize=(12, 10))  # Bigger figure

        # Apply mask if requested
        mask_matrix = np.triu(np.ones_like(self.corr_matrix, dtype=bool)) if mask else None

        sns.heatmap(
            self.corr_matrix,
            mask=mask_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            annot_kws={"size": 12},  # Bigger annotation numbers
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
        )
        plt.title("Correlation Heatmap (Reservoir Parameters + Rates)", fontsize=18, pad=20)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        print(f"Correlation heatmap saved to {filename} (mask={mask})")

    def plot_target_correlation(self, target, filename="./src/plot/cor/target_correlation_bar.png"):
        if self.corr_matrix is None:
            self.calculate_correlation()

        if target not in self.df_selected.columns:
            raise ValueError(f"Target '{target}' not found in dataset.")

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Extract correlations with target
        target_corr = self.corr_matrix[target].drop(target)

        # Sort by absolute strength
        target_corr_sorted = target_corr.reindex(target_corr.abs().sort_values(ascending=False).index)

        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=target_corr_sorted.values, y=target_corr_sorted.index, palette="coolwarm")

        # Axis labels and title with larger fonts
        plt.xlabel(f"Correlation with {target}", fontsize=14, labelpad=10)
        plt.ylabel("Feature", fontsize=14, labelpad=10)
        plt.title(f"Feature Correlation with {target}", fontsize=16, pad=15)

        # Tick labels larger
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        print(f"{target} correlation bar chart saved to {filename}")
