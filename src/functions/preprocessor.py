import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the preprocessor with a DataFrame.
        """
        self.df = df.copy()
        self.df_scaled = None
        self.correlations = None
        self.scaler = MinMaxScaler()
        self.id_like_cols = []   # UUIDs, datetimes, unique IDs
        self.meta_cols = []      # system capacity, panel type, etc.
        self.features = []

    def _detect_features(self):
        """
        Detect numeric feature columns automatically (excluding timestamp).
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if "timestamp" in numeric_cols:
            numeric_cols.remove("timestamp")
        return numeric_cols
    
    def handle_missing_data(self):
        """Fill missing values using linear interpolation and forward/backward fill."""
        self.df = self.df.interpolate(method='linear')
        self.df = self.df.ffill().bfill()
        return self

    def remove_outliers(self):
        """Replace values exceeding ±3 standard deviations with column mean."""
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if col in self.meta_cols:   # skip meta columns like system capacity
                continue
            mean = self.df[col].mean()
            std = self.df[col].std()
            upper, lower = mean + 3 * std, mean - 3 * std
            self.df[col] = np.where(
                (self.df[col] > upper) | (self.df[col] < lower),
                mean,
                self.df[col]
            )
        return self

    def detect_non_scalable_columns(self):
        """
        Detect columns that should not be scaled:
        """
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                self.id_like_cols.append(col)
            elif np.issubdtype(self.df[col].dtype, np.datetime64):
                self.id_like_cols.append(col)
            elif self.df[col].nunique() == len(self.df):  # unique per row
                self.id_like_cols.append(col)

        # Manually tag known metadata columns
        for col in ["timestamp","depth_md (ft)","depth_tvd (ft)", "bubble_point_pressure (psi)",
                    "porosity_percent (%)", ]:
            if col in self.df.columns:
                self.meta_cols.append(col)
        return self

    def normalize(self):
        """Apply Min-Max scaling (0–1) only to numeric features that are not IDs/metadata."""
        self.detect_non_scalable_columns()

        # Numeric columns to scale (excluding id-like + meta cols)
        numeric_cols = [
            col for col in self.df.select_dtypes(include=[np.number]).columns
            if col not in self.id_like_cols and col not in self.meta_cols
        ]

        # Scale selected numeric features
        scaled_values = self.scaler.fit_transform(self.df[numeric_cols])
        self.df_scaled = pd.DataFrame(scaled_values,
                                      columns=numeric_cols,
                                      index=self.df.index)

        # Add back excluded columns unchanged
        for col in self.df.columns:
            if col not in numeric_cols:
                self.df_scaled[col] = self.df[col]

        return self

    def compute_correlation(self):
        """Compute Pearson correlation matrix only on scaled numeric features."""
        numeric_cols = self.df_scaled.select_dtypes(include=[np.number]).columns

        # Drop meta columns from correlation
        numeric_cols = [c for c in numeric_cols if c not in self.meta_cols]
        self.features = numeric_cols

        self.correlations = self.df_scaled[numeric_cols].corr(method='pearson')
        return self

    def run_pipeline(self):
        """Run all preprocessing steps in sequence."""
        return (self.handle_missing_data()
                    .remove_outliers()
                    .normalize()
                    .compute_correlation())

    def get_processed_data(self):
        """Return the processed DataFrame."""
        return self.df_scaled

    def get_correlation_matrix(self):
        """Return the Pearson correlation matrix."""
        return self.correlations

    def plot_correlation(self):
        """Correlation heatmap using matplotlib only (no seaborn)."""
        corr = self.df[self.features].corr()
        fig, ax = plt.subplots(figsize=(8,6))
        cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(len(corr.columns))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(corr.columns, rotation=45, ha="left")
        ax.set_yticklabels(corr.columns)
        plt.title("Feature Correlation Heatmap", pad=20)
        plt.show()