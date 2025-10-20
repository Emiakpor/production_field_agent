import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

class CorrelationAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df_selected = None
        self.corr_matrix = None

        # Define only the features you want
        self.selected_features = [
            "reservoir_pressure_initial",
            "bottomhole_pressure_psi",
            "permeability_md",
            "net_pay_thickness_ft",
            "bubble_point_pressure",
            "porosity_percent",
            "oil_rate_bopd",          # historical production volume
            "choke_size_percent"
        ]

    def preprocess(self):
        # Keep only selected features if they exist in the dataset
        self.df_selected = self.df[[c for c in self.selected_features if c in self.df.columns]].copy()

        # Drop rows with missing values in these columns
        self.df_selected = self.df_selected.dropna()

    def calculate_correlation(self):
        if self.df_selected is None:
            self.preprocess()
        self.corr_matrix = self.df_selected.corr()
        return self.corr_matrix

    def save_correlation_excel(self, filename="./src/plot/cor/correlation_matrix.xlsx"):
        if self.corr_matrix is None:
            self.calculate_correlation()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.corr_matrix.to_excel(filename)
        print(f"Correlation matrix saved to {filename}")

    def plot_heatmap(self, filename="./src/plot/cor/correlation_heatmap.png"):
        if self.corr_matrix is None:
            self.calculate_correlation()
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        mask = np.triu(np.ones_like(self.corr_matrix, dtype=bool))

        plt.figure(figsize=(10, 8))
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
        plt.title("Correlation Heatmap (Selected Reservoir Parameters)", fontsize=14, pad=15)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Correlation heatmap saved to {filename}")
