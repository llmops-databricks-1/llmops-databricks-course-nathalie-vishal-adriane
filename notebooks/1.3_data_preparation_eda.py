# Databricks notebook source
"""
Data Preparation & EDA for the Agent
Loads olist_ecomerce_data from workspace.default and performs exploratory data analysis.
"""

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# COMMAND ----------

# Load data from Databricks table
TABLE_NAME = "workspace.default.olist_ecomerce_data"

df_spark = spark.table(TABLE_NAME)  # noqa: F821 - spark is a Databricks runtime global
print(f"Table   : {TABLE_NAME}")
print(f"Row count: {df_spark.count():,}")
print(f"Columns : {len(df_spark.columns)}")
df_spark.printSchema()

# COMMAND ----------

# Convert to pandas for EDA
SAMPLE_ROWS = 10_000

df = df_spark.limit(SAMPLE_ROWS).toPandas()
print(f"Converted to pandas: {df.shape}")
df.head()

# COMMAND ----------

# --- Shape & dtypes ---
print("Shape:", df.shape)
print()
print(df.dtypes)

# COMMAND ----------

# --- Missing values ---
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_summary = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_summary = missing_summary[missing_summary["missing_count"] > 0].sort_values(
    "missing_pct", ascending=False
)
print("Columns with missing values:")
print(missing_summary if not missing_summary.empty else "None")

# COMMAND ----------

# --- Descriptive statistics (numeric) ---
_ = df.describe(include="number").T  # Display in notebook

# COMMAND ----------

# --- Descriptive statistics (object / categorical) ---
_ = df.describe(include="object").T  # Display in notebook

# COMMAND ----------

# --- Cardinality of categorical columns ---
cat_cols = df.select_dtypes(include="object").columns.tolist()
cardinality = {col: df[col].nunique() for col in cat_cols}
cardinality_df = pd.DataFrame.from_dict(
    cardinality, orient="index", columns=["unique_values"]
).sort_values("unique_values", ascending=False)
print("Cardinality of categorical columns:")
_ = cardinality_df  # Display in notebook

# COMMAND ----------

# --- Distribution of key numeric columns ---
numeric_cols = df.select_dtypes(include="number").columns.tolist()[:8]  # cap at 8
if numeric_cols:
    fig, axes = plt.subplots(
        nrows=(len(numeric_cols) + 1) // 2,
        ncols=2,
        figsize=(14, 4 * ((len(numeric_cols) + 1) // 2)),
    )
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        df[col].dropna().hist(bins=40, ax=axes[i], color="steelblue", edgecolor="white")
        axes[i].set_title(col)
        axes[i].set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Numeric Column Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# --- Value counts for top categorical columns (top 5 values each) ---
top_cat_cols = cardinality_df[cardinality_df["unique_values"] <= 50].index.tolist()[:4]
for col in top_cat_cols:
    print(f"\n{col} — top 10 values:")
    print(df[col].value_counts().head(10).to_string())

# COMMAND ----------

# --- Correlation heatmap (numeric columns) ---
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

# COMMAND ----------

print("EDA complete. Dataset is ready for agent preparation.")
