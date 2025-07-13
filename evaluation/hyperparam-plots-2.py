import pandas as pd
import matplotlib.pyplot as plt

# Load the relevant dataset
data1 = pd.read_csv("out/hyperparam-search-1.csv")
data2 = pd.read_csv("out/hyperparam-search-2.csv")
data3 = pd.read_csv("out/hyperparam-search-3.csv")

# Extract subsets of data for each plot for all papers
data_sets = []
for data in [data1, data2, data3]:
    data_sets.append({
        'breadth': data.iloc[:4],
        'depth': data.iloc[4:9],
        'diversity': data.iloc[9:13]
    })

# Set up the figure and axis for the three plots
fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=False)

# Aggregate data across papers
agg_data = {}
for metric in ['breadth', 'depth', 'diversity']:
    df = pd.concat([d[metric] for d in data_sets])
    agg_data[metric] = df.groupby(metric).agg({
        'quality_score': 'mean',
        'avg_relevance_score': 'mean',
        'std_relevance_score': 'mean'
    }).reset_index()

# Plot 1: Breadth
ax = axes[0]
ax.set_title("Breadth vs Scores", fontsize=20)
ax.plot(agg_data['breadth']['breadth'], agg_data['breadth']['quality_score'],
        label='Quality Score', color='blue', marker='o')
ax.errorbar(agg_data['breadth']['breadth'], agg_data['breadth']['avg_relevance_score'],
            yerr=agg_data['breadth']['std_relevance_score'], label='Avg Relevance Score',
            color='orange', marker='s', linestyle='--')
ax.set_xlabel('Breadth', fontsize=16)
ax.set_ylabel('Score', fontsize=16)
ax.set_ylim(0, 10)
ax.legend(loc='lower right', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
ax.tick_params(labelsize=16)

# Plot 2: Depth 
ax = axes[1]
ax.set_title("Depth vs Scores", fontsize=20)
ax.plot(agg_data['depth']['depth'], agg_data['depth']['quality_score'],
        label='Quality Score', color='blue', marker='o')
ax.errorbar(agg_data['depth']['depth'], agg_data['depth']['avg_relevance_score'],
            yerr=agg_data['depth']['std_relevance_score'], label='Avg Relevance Score',
            color='orange', marker='s', linestyle='--')
ax.set_xlabel('Depth', fontsize=16)
ax.set_ylabel('Score', fontsize=16)
ax.set_ylim(0, 10)
ax.legend(loc='lower right', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
ax.tick_params(labelsize=16)

# Plot 3: Diversity
ax = axes[2]
ax.set_title("Diversity vs Scores", fontsize=20)
ax.plot(agg_data['diversity']['diversity'], agg_data['diversity']['quality_score'],
        label='Quality Score', color='blue', marker='o')
ax.errorbar(agg_data['diversity']['diversity'], agg_data['diversity']['avg_relevance_score'],
            yerr=agg_data['diversity']['std_relevance_score'], label='Avg Relevance Score',
            color='orange', marker='s', linestyle='--')
ax.set_xlabel('Diversity', fontsize=16)
ax.set_ylabel('Score', fontsize=16)
ax.set_ylim(0, 10)
ax.legend(loc='lower right', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
ax.tick_params(labelsize=16)

# Adjust layout for better visualization
plt.tight_layout()
plt.savefig('out/plots/hyperparam-search-plot.pdf', bbox_inches='tight', dpi=300)
plt.show()
