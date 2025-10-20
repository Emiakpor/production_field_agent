import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

class SelectedCorrelationAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df_selected = None
        self.corr_matrix = None

        # Only the required features
        self.selected_features = [
            "reservoir_pressure_initial (psi)",
            "bottomhole_pressure_psi (psi)",
            "permeability_md (mD)",
            "net_pay_thickness_ft (ft)",
            "bubble_point_pressure (psi)",
            "porosity_percent (%)",
            "oil_rate_bopd (BOPD)",   # historical production volume
            "gas_rate_mscf_day (MSCF/day)",   # historical production volume
            "water_rate_bwpd (BWPD)",   # historical production volume
            "choke_size_percent (%)"
        ]

    def preprocess(self):
        # Keep only selected features that exist in dataset
        self.df_selected = self.df[[c for c in self.selected_features if c in self.df.columns]].copy()
        # Drop rows with missing values
        self.df_selected = self.df_selected.dropna()

    def calculate_correlation(self):
        if self.df_selected is None:
            self.preprocess()
        self.corr_matrix = self.df_selected.corr()
        return self.corr_matrix

    def save_correlation_excel(self, filename="./src/plot/cor/selected_correlation_matrix.xlsx"):
        if self.corr_matrix is None:
            self.calculate_correlation()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.corr_matrix.to_excel(filename)
        print(f"Correlation matrix saved to {filename}")

    def plot_heatmaps(self, filename="./src/plot/cor/selected_correlation_heatmap.png"):
        if self.corr_matrix is None:
            self.calculate_correlation()
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        mask = np.triu(np.ones_like(self.corr_matrix, dtype=bool))

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            self.corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
        )
        plt.title("Correlation Heatmap (Reservoir Parameters)", fontsize=14, pad=15)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
    
    def plot_heatmap(self, filename="./src/plot/cor/selected_correlation_heatmap.png", mask=False):
        if self.corr_matrix is None:
            self.calculate_correlation()
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        plt.figure(figsize=(10, 8))

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
            annot_kws={"size": 8},
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
        )
        plt.title("Correlation Heatmap", fontsize=16, pad=15)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        print(f"Correlation heatmap saved to {filename} (mask={mask})")


    def plot_oil_rate_correlation(self, filename="./src/plot/cor/oil_rate_correlation_bar.png", target="oil_rate_bopd (BOPD)"):
        if self.corr_matrix is None:
            self.calculate_correlation()

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Extract correlations with oil_rate_bopd
        oil_corr = self.corr_matrix[target].drop(target)

        # Sort by absolute correlation strength
        oil_corr_sorted = oil_corr.reindex(oil_corr.abs().sort_values(ascending=False).index)

        # Plot
        plt.figure(figsize=(8, 5))
        sns.barplot(x=oil_corr_sorted.values, y=oil_corr_sorted.index, palette="coolwarm")
        plt.xlabel("Correlation with Oil Rate (BOPD)")
        plt.ylabel("Feature")
        plt.title("Feature Correlation with Production", fontsize=14, pad=12)
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        print(f"Oil rate correlation bar chart saved to {filename}")
