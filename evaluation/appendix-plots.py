import random
import pandas as pd
import matplotlib.pyplot as plt
import os

#**********************************
# monthly arxiv submissions plot
#**********************************

df = pd.read_csv("out/arxiv_monthly_submissions.csv")
df = df[:-1]

df['month'] = pd.to_datetime(df['month'])
df = df.sort_values(by='month')

max_month = df['month'][df['submissions'].idxmax()]
max_value = df['submissions'].max()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['month'], df['submissions'] / 1000, color='blue', linewidth=2, label='Submissions')

ax.scatter(
        max_month, max_value / 1000,
        color='red', s=100, zorder=5, label=f'Max: {max_value/1000:.1f}k'
)
ax.text(
        max_month, max_value / 1000 + 1,
        f'Max: {max_value/1000:.1f}k ({max_month.strftime("%m/%Y")})',
        fontsize=14, ha='right', color='black'
)

plt.ylim(0, (df['submissions'].max() / 1000) + 3)
xlim = ax.get_xlim()
plt.xlim(xlim[0], xlim[1]-400)

ax.grid(visible=True, which='major', linestyle='--', alpha=0.5)

ax.set_title("Monthly Submissions Over Time", fontsize=20)
ax.set_xlabel("Month", fontsize=16)
ax.set_ylabel("Submissions (in thousands)", fontsize=16)

ax.fill_between(
    df['month'],
    df['submissions'] / 1000,
    color='blue',
    alpha=0.3
)

plt.xticks(rotation=45, fontsize = 14)
plt.yticks(fontsize=14)
ax.legend(fontsize=14)

plt.tight_layout()

output_dir = "./out/plots"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "monthly_submissions_line_plot.pdf")
plt.savefig(output_path)
plt.show()

#**********************************
# monthly arxiv submissions plot
#**********************************
df = pd.read_csv("out/arxiv_submissions_by_category_history.csv", encoding="utf-16")
df = df.sort_values(by="Count", ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))

cmap = plt.colormaps.get_cmap('Paired')
bar_colors = [cmap(i / len(df)) for i in range(len(df))]

bars = ax.bar(
    df['Archive'],
    df['Count'],
    color=bar_colors
)
ax.set_title("Cumulative Entries by Category", fontsize=20)
ax.set_xlabel("Category", fontsize=16)
ax.set_ylabel("Entries (in thousands)", fontsize=16)


plt.xticks(rotation=45, fontsize=14)
ax.set_yticks(ax.get_yticks())
ax.set_yticklabels([f"{int(tick / 1000)}" for tick in ax.get_yticks()])
plt.yticks(fontsize=14)
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + (height * 0.01),
        f"{height / 1000:.1f}k",
        ha='center',
        va='bottom',
        fontsize=14
    )


ax.grid(visible=True, which='major', linestyle='--', alpha=0.5)
plt.tight_layout()
output_path = os.path.join(output_dir, "cumulative_entries_by_category_histogram.pdf")
plt.savefig(output_path)

plt.show()

