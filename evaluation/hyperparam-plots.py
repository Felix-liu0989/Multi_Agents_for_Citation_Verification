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

# Set up the figure and axis for the nine plots
fig, axes = plt.subplots(3, 3, figsize=(18, 18), sharey=False)

paper_titles = ['Paper 1', 'Paper 2', 'Paper 3']

for row, (data_dict, paper) in enumerate(zip(data_sets, paper_titles)):
    # Plot 1: Breadth
    ax = axes[row, 0]
    ax.set_title(f"{paper} - Breadth vs Scores", fontsize=20)
    ax.plot(data_dict['breadth']['breadth'], data_dict['breadth']['quality_score'],
            label='Quality Score', color='blue', marker='o')
    ax.errorbar(data_dict['breadth']['breadth'], data_dict['breadth']['avg_relevance_score'],
                yerr=data_dict['breadth']['std_relevance_score'], label='Avg Relevance Score',
                color='orange', marker='s', linestyle='--')
    ax.set_xlabel('Breadth', fontsize=16)
    ax.set_ylabel('Score', fontsize=16)
    ax.set_ylim(0, 10)
    ax.legend(loc='lower right', fontsize=14)

    # Plot 2: Depth
    ax = axes[row, 1]
    ax.set_title(f"{paper} - Depth vs Scores", fontsize=20)
    ax.plot(data_dict['depth']['depth'], data_dict['depth']['quality_score'],
            label='Quality Score', color='blue', marker='o')
    ax.errorbar(data_dict['depth']['depth'], data_dict['depth']['avg_relevance_score'],
                yerr=data_dict['depth']['std_relevance_score'], label='Avg Relevance Score',
                color='orange', marker='s', linestyle='--')
    ax.set_xlabel('Depth', fontsize=16)
    ax.set_ylabel('Score', fontsize=16)
    ax.set_ylim(0, 10)
    ax.legend(loc='lower right', fontsize=14)

    # Plot 3: Diversity
    ax = axes[row, 2]
    ax.set_title(f"{paper} - Diversity vs Scores", fontsize=20)
    ax.plot(data_dict['diversity']['diversity'], data_dict['diversity']['quality_score'],
            label='Quality Score', color='blue', marker='o')
    ax.errorbar(data_dict['diversity']['diversity'], data_dict['diversity']['avg_relevance_score'],
                yerr=data_dict['diversity']['std_relevance_score'], label='Avg Relevance Score',
                color='orange', marker='s', linestyle='--')
    ax.set_xlabel('Diversity', fontsize=16)
    ax.set_ylabel('Score', fontsize=16)
    ax.set_ylim(0, 10)
    ax.legend(loc='lower right', fontsize=14)

# Adjust layout for better visualization
plt.tight_layout()
plt.savefig('out/plots/hyperparam-search-plot.pdf', bbox_inches='tight', dpi=300)
plt.show()
