import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np  # add this at the top

class CorrelationAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df_clean = None
        self.corr_matrix = None

    def preprocess(self):
        remove_cols = [
            "well_id",
            "timestamp",
            "completion_type",
            "lift_type",
            "region",
            "shift",
            "field"
        ]

        # Convert boolean columns to int
        bool_cols = ["maintenance_event", "shutdown_flag"]
        for col in bool_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(int)

        # Remove duplicate columns
        self.df = self.df.loc[:, ~self.df.columns.duplicated()]

        # Drop unwanted categorical columns
        self.df_clean = self.df.drop(
            columns=[c for c in remove_cols if c in self.df.columns],
            errors="ignore"
        )

        # Keep only numeric
        self.df_clean = self.df_clean.select_dtypes(include=["number"])

    def calculate_correlation(self):
        if self.df_clean is None:
            self.preprocess()
        self.corr_matrix = self.df_clean.corr()
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

        # Mask upper triangle
        mask = np.triu(np.ones_like(self.corr_matrix, dtype=bool))

        plt.figure(figsize=(18, 14))
        sns.heatmap(
            self.corr_matrix,
            mask=mask,
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
        )
        plt.title("Correlation Heatmap of Well Parameters", fontsize=18, pad=20)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        print(f"Improved correlation heatmap saved to {filename}")

    def plot_top_correlations(self, target="oil_rate_bopd", top_n=15, filename="./src/plot/cor/top_correlations.png"):
        if self.corr_matrix is None:
            self.calculate_correlation()

        if target not in self.corr_matrix.columns:
            raise ValueError(f"Target column '{target}' not found in dataset.")

        # Get correlations with target
        target_corr = self.corr_matrix[target].drop(target).sort_values(key=lambda x: abs(x), ascending=False)

        # Select top N
        top_corr = target_corr.head(top_n)

        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x=top_corr.values, y=top_corr.index, palette="coolwarm", orient="h")
        plt.title(f"Top {top_n} Correlated Features with {target}", fontsize=16, pad=15)
        plt.xlabel("Correlation Coefficient")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        print(f"Top {top_n} correlations with '{target}' saved to {filename}")
