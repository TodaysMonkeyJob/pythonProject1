import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load all sheets from the Excel file
file_path = "multi_model_sentence_data.xlsx"  # <-- replace with actual file path
sheets = ['Tiny', 'Large 2', 'Whisper 1', 'Whisper 1 en']

# Load data and tag with model name
data = pd.concat(
    [pd.read_excel(file_path, sheet_name=sheet).assign(Model=sheet) for sheet in sheets],
    ignore_index=True
)

# Ensure numeric data types
data['Accurancy'] = pd.to_numeric(data['Accurancy'], errors='coerce')
data['Sentence Length'] = pd.to_numeric(data['Sentence Length'], errors='coerce')
print(data.to_excel("test.xlsx"))
# === Global Comparison ===
agg_data = data.groupby(['Model', 'Sentence Length']).agg({
    'Accurancy': 'mean',
    'Processing Time (s)': 'mean',
    'Memory Usage (MB)': 'mean'
}).reset_index()
#
# # Plot: Accuracy vs Sentence Length (global)
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=agg_data, x='Sentence Length', y='Accurancy', hue='Model', marker='o')
# plt.title("Accuracy by Sentence Length (All Models)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # Plot: Processing Time vs Sentence Length
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=agg_data, x='Sentence Length', y='Processing Time (s)', hue='Model', marker='o')
# plt.title("Processing Time by Sentence Length")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # Plot: Memory Usage vs Sentence Length
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=agg_data, x='Sentence Length', y='Memory Usage (MB)', hue='Model', marker='o')
# plt.title("Memory Usage by Sentence Length")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # === Per-model analysis ===
# models = data['Model'].unique()
# for model in models:
#     model_data = data[data['Model'] == model]
#
#     print(f"\n🔍 Analysis for model: {model}")
#     print("-" * 60)
#
#     print("Average accuracy:", round(model_data['Accurancy'].mean(), 2))
#     print("Average processing time (s):", round(model_data['Processing Time (s)'].mean(), 2))
#     print("Average memory usage (MB):", round(model_data['Memory Usage (MB)'].mean(), 2))
#
#     # Correlation check
#     corr = model_data[['Sentence Length', 'Accurancy', 'Processing Time (s)', 'Memory Usage (MB)']].corr()
#     print("\nCorrelation matrix:")
#     print(corr)
#
#     # Plots for current model
#     fig, axs = plt.subplots(1, 3, figsize=(18, 4))
#
#     sns.lineplot(ax=axs[0], data=model_data, x='Sentence Length', y='Accurancy', marker='o')
#     axs[0].set_title(f"{model}: Accuracy")
#     axs[0].grid(True)
#
#     sns.lineplot(ax=axs[1], data=model_data, x='Sentence Length', y='Processing Time (s)', marker='o')
#     axs[1].set_title(f"{model}: Processing Time")
#     axs[1].grid(True)
#
#     sns.lineplot(ax=axs[2], data=model_data, x='Sentence Length', y='Memory Usage (MB)', marker='o')
#     axs[2].set_title(f"{model}: Memory Usage")
#     axs[2].grid(True)
#
#     plt.suptitle(f"Model Analysis: {model}", fontsize=16)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()

# === Final ranking summary ===
summary = data.groupby('Model').agg({
    'Accurancy': 'mean',
    'Processing Time (s)': 'mean',
    'Memory Usage (MB)': 'mean'
}).sort_values(by='Accurancy', ascending=False).reset_index()

print("\n=== Final Summary Across All Models ===")
print(summary)
