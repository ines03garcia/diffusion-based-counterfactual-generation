"""
 Given the radiologist assessment Excel file, process the data to generate visualizations and statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ast

import matplotlib.pyplot as plt
from src.config import METADATA_ROOT, DATA_ROOT


def clear_output_directory(output_path):
    """Delete all files in the output directory."""
    if output_path.exists():
        for item in output_path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                clear_output_directory(item)
        print(f"Cleared output directory: {output_path}")


def load_data(excel_file):
    """Load data from Excel file containing radiologist ratings."""
    df_table1 = pd.read_excel(excel_file, sheet_name="table1")
    df_table2 = pd.read_excel(excel_file, sheet_name="table2")
    return df_table1, df_table2


def load_radiologist_dataset():
    """Load radiologist dataset to get breast density and BI-RADS information."""
    csv_file = METADATA_ROOT / Path("radiologist_dataset.csv")
    df_radio = pd.read_csv(csv_file)
    
    # Create mapping from "{radiologist_id}_{image_id}" to breast_density and breast_birads
    density_map = {}
    birads_map = {}
    for _, row in df_radio.iterrows():
        # Remove .png extension from image_id
        image_id = row['image_id'].replace('.png', '')
        key = f"{row['radiologist_id']}_{image_id}"
        density_map[key] = row['breast_density']
        birads_map[key] = row['breast_birads']

    return density_map, birads_map


def organize_table_data(df_table, density_map, birads_map):
    """Organize table data into structured DataFrame with image IDs, ratings, breast density, and BI-RADS."""
    image_ids = df_table.columns
    ratings = df_table.iloc[0:].values
    
    # Create organized DataFrame
    organized = pd.DataFrame(ratings.T, columns=[f"Rater_{i+1}" for i in range(ratings.shape[0])])
    organized.insert(0, "image_id", image_ids)
    organized = organized.dropna(subset=["image_id"])
    
    # Map breast density and BI-RADS
    organized["breast_density"] = organized["image_id"].map(density_map)
    organized["breast_birads"] = organized["image_id"].map(birads_map)
    
    return organized


def calculate_stats(df):
    """Calculate mean, standard deviation, and median ratings for each image."""
    rating_cols = [col for col in df.columns if col.startswith("Rater_")]
    df["mean_rating"] = df[rating_cols].mean(axis=1)
    df["std_rating"] = df[rating_cols].std(axis=1)
    df["median_rating"] = df[rating_cols].median(axis=1)
    return df


def plot_rating_distributions(table1_organized, table2_organized, per_image_path):
    """Create histograms showing distribution of ratings for both tables."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Table 1 distribution
    rating_cols_t1 = [col for col in table1_organized.columns if col.startswith("Rater_")]
    all_ratings_t1 = table1_organized[rating_cols_t1].values.flatten()
    all_ratings_t1 = all_ratings_t1[~np.isnan(all_ratings_t1)]

    axes[0].hist(all_ratings_t1, bins=np.arange(0.5, 6.5, 1), edgecolor='black', alpha=0.7)
    axes[0].set_xlabel("Rating")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Ratings - Whole Image (Table 1)")
    axes[0].set_xticks(range(1, 6))

    # Table 2 distribution
    rating_cols_t2 = [col for col in table2_organized.columns if col.startswith("Rater_")]
    all_ratings_t2 = table2_organized[rating_cols_t2].values.flatten()
    all_ratings_t2 = all_ratings_t2[~np.isnan(all_ratings_t2)]

    axes[1].hist(all_ratings_t2, bins=np.arange(0.5, 6.5, 1), edgecolor='black', alpha=0.7)
    axes[1].set_xlabel("Rating")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Ratings - Bounding Box (Table 2)")
    axes[1].set_xticks(range(1, 6))

    plt.tight_layout()
    plt.savefig(per_image_path / "rating_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return all_ratings_t1, all_ratings_t2


def plot_mean_ratings_distributions(table1_organized, table2_organized, per_image_path):
    """Create histograms showing distribution of mean ratings across images."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Table 1 histogram
    mean_ratings_t1 = table1_organized["mean_rating"].values
    axes[0].hist(mean_ratings_t1, bins=np.arange(0.5, 6.5, 1), edgecolor='black', alpha=0.7)
    axes[0].set_xlabel("Rating")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Mean Ratings - Full Image (Table 1)")
    axes[0].set_xticks(range(1, 6))
    axes[0].set_xlim(0.5, 5.5)

    # Table 2 histogram
    mean_ratings_t2 = table2_organized["mean_rating"].values
    axes[1].hist(mean_ratings_t2, bins=np.arange(0.5, 6.5, 1), edgecolor='black', alpha=0.7)
    axes[1].set_xlabel("Rating")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Mean Ratings - Bounding Box (Table 2)")
    axes[1].set_xticks(range(1, 6))
    axes[1].set_xlim(0.5, 5.5)

    plt.tight_layout()
    plt.savefig(per_image_path / "mean_ratings_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_mean_ratings_by_density(table1_organized, table2_organized, per_density_path):
    """Create grouped bar charts showing distribution of mean ratings grouped by breast density."""
    densities = ['DENSITY A', 'DENSITY B', 'DENSITY C', 'DENSITY D']
    colors = ['#6FA8DC', '#93C47D', '#FFE599', '#E06666']  # Semi-pastel Blue, Green, Yellow, Red
    density_colors = dict(zip(densities, colors))
    ratings = range(1, 6)
    
    # Table 1 - grouped bar chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Calculate percentage for each rating and density combination
    rating_density_percentages_t1 = {}
    for rating in ratings:
        rating_density_percentages_t1[rating] = {}
        for density in densities:
            density_data = table1_organized[table1_organized['breast_density'] == density]['mean_rating']
            if len(density_data) > 0:
                # Count how many times this rating appears (within bin range)
                count = np.sum((density_data >= rating - 0.5) & (density_data < rating + 0.5))
                percentage = (count / len(density_data)) * 100
                rating_density_percentages_t1[rating][density] = percentage
            else:
                rating_density_percentages_t1[rating][density] = 0
    
    # Create grouped bars
    num_densities = len(densities)
    bar_width = 0.2
    x_pos = np.arange(len(ratings))
    
    for i, density in enumerate(densities):
        percentages = [rating_density_percentages_t1[rating][density] for rating in ratings]
        offset = (i - num_densities / 2) * bar_width + bar_width / 2
        ax.bar(x_pos + offset, percentages, bar_width, 
               color=density_colors[density], edgecolor='white', linewidth=1, label=density)
    
    ax.set_xlabel("Rating", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Mean Ratings Distribution by Breast Density - Full Image (Table 1)", fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ratings)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(per_density_path / "mean_ratings_by_density_table1.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Table 2 - grouped bar chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Calculate percentage for each rating and density combination
    rating_density_percentages_t2 = {}
    for rating in ratings:
        rating_density_percentages_t2[rating] = {}
        for density in densities:
            density_data = table2_organized[table2_organized['breast_density'] == density]['mean_rating']
            if len(density_data) > 0:
                # Count how many times this rating appears (within bin range)
                count = np.sum((density_data >= rating - 0.5) & (density_data < rating + 0.5))
                percentage = (count / len(density_data)) * 100
                rating_density_percentages_t2[rating][density] = percentage
            else:
                rating_density_percentages_t2[rating][density] = 0
    
    # Create grouped bars
    for i, density in enumerate(densities):
        percentages = [rating_density_percentages_t2[rating][density] for rating in ratings]
        offset = (i - num_densities / 2) * bar_width + bar_width / 2
        ax.bar(x_pos + offset, percentages, bar_width, 
               color=density_colors[density], edgecolor='white', linewidth=1, label=density)
    
    ax.set_xlabel("Rating", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Mean Ratings Distribution by Breast Density - Bounding Box (Table 2)", fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ratings)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(per_density_path / "mean_ratings_by_density_table2.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_mean_ratings_by_birads(table1_organized, table2_organized, per_birads_path):
    """Create grouped bar charts showing distribution of mean ratings grouped by BI-RADS category."""
    # Get unique BI-RADS categories from the data and sort them
    birads_t1 = sorted(table1_organized['breast_birads'].dropna().unique())
    birads_t2 = sorted(table2_organized['breast_birads'].dropna().unique())
    all_birads = sorted(set(birads_t1) | set(birads_t2))
    
    # Define colors for BI-RADS categories (green to red spectrum for increasing risk)
    color_palette = ['#93C47D', '#FFE599', '#F6B26B', '#E06666', '#CC0000']
    # Create color mapping based on available BI-RADS
    birads_colors = {birads: color_palette[i % len(color_palette)] for i, birads in enumerate(all_birads)}
    
    ratings = range(1, 6)
    
    # Table 1 - grouped bar chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Calculate percentage for each rating and BI-RADS combination
    rating_birads_percentages_t1 = {}
    for rating in ratings:
        rating_birads_percentages_t1[rating] = {}
        for birads in all_birads:
            birads_data = table1_organized[table1_organized['breast_birads'] == birads]['mean_rating']
            if len(birads_data) > 0:
                # Count how many times this rating appears (within bin range)
                count = np.sum((birads_data >= rating - 0.5) & (birads_data < rating + 0.5))
                percentage = (count / len(birads_data)) * 100
                rating_birads_percentages_t1[rating][birads] = percentage
            else:
                rating_birads_percentages_t1[rating][birads] = 0
    
    # Create grouped bars
    num_birads = len(all_birads)
    bar_width = 0.8 / num_birads if num_birads > 0 else 0.2
    x_pos = np.arange(len(ratings))
    
    for i, birads in enumerate(all_birads):
        percentages = [rating_birads_percentages_t1[rating][birads] for rating in ratings]
        offset = (i - num_birads / 2) * bar_width + bar_width / 2
        ax.bar(x_pos + offset, percentages, bar_width, 
               color=birads_colors[birads], edgecolor='white', linewidth=1, label=birads)
    
    ax.set_xlabel("Rating", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Mean Ratings Distribution by BI-RADS Category - Full Image (Table 1)", fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ratings)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(per_birads_path / "mean_ratings_by_birads_table1.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Table 2 - grouped bar chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Calculate percentage for each rating and BI-RADS combination
    rating_birads_percentages_t2 = {}
    for rating in ratings:
        rating_birads_percentages_t2[rating] = {}
        for birads in all_birads:
            birads_data = table2_organized[table2_organized['breast_birads'] == birads]['mean_rating']
            if len(birads_data) > 0:
                # Count how many times this rating appears (within bin range)
                count = np.sum((birads_data >= rating - 0.5) & (birads_data < rating + 0.5))
                percentage = (count / len(birads_data)) * 100
                rating_birads_percentages_t2[rating][birads] = percentage
            else:
                rating_birads_percentages_t2[rating][birads] = 0
    
    # Create grouped bars
    for i, birads in enumerate(all_birads):
        percentages = [rating_birads_percentages_t2[rating][birads] for rating in ratings]
        offset = (i - num_birads / 2) * bar_width + bar_width / 2
        ax.bar(x_pos + offset, percentages, bar_width, 
               color=birads_colors[birads], edgecolor='white', linewidth=1, label=birads)
    
    ax.set_xlabel("Rating", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Mean Ratings Distribution by BI-RADS Category - Bounding Box (Table 2)", fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ratings)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(per_birads_path / "mean_ratings_by_birads_table2.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_summary_statistics(table1_organized, table2_organized, all_ratings_t1, all_ratings_t2, output_path):
    """Create bar chart with summary statistics for both tables."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    summary_data = {
        'Metric': ['Full Image\nMean', 'Full Image\nMedian', 'Full Image\nStd Dev',
                   'Bounding Box\nMean', 'Bounding Box\nMedian', 'Bounding Box\nStd Dev'],
        'Value': [
            table1_organized["mean_rating"].mean(),
            table1_organized["median_rating"].median(),
            all_ratings_t1.std(),
            table2_organized["mean_rating"].mean(),
            table2_organized["median_rating"].median(),
            all_ratings_t2.std()
        ]
    }

    colors = ['#1f77b4'] * 3 + ['#ff7f0e'] * 3
    bars = ax.bar(summary_data['Metric'], summary_data['Value'], color=colors, alpha=0.7)
    ax.set_ylabel('Value')
    ax.set_title('Summary Statistics of Radiologist Ratings')
    ax.set_ylim(0, 6)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path / "summary_statistics.png", dpi=300, bbox_inches='tight')
    plt.close()


def calculate_rater_statistics(table1_organized, table2_organized):
    """Calculate statistics for each rater across all images."""
    rating_cols_t1 = [col for col in table1_organized.columns if col.startswith("Rater_")]
    rating_cols_t2 = [col for col in table2_organized.columns if col.startswith("Rater_")]
    
    # Statistics for Table 1
    rater_stats_t1 = pd.DataFrame({
        'Rater': rating_cols_t1,
        'Mean': [table1_organized[col].mean() for col in rating_cols_t1],
        'Median': [table1_organized[col].median() for col in rating_cols_t1],
        'Std': [table1_organized[col].std() for col in rating_cols_t1],
        'Count': [table1_organized[col].count() for col in rating_cols_t1]
    })
    
    # Statistics for Table 2
    rater_stats_t2 = pd.DataFrame({
        'Rater': rating_cols_t2,
        'Mean': [table2_organized[col].mean() for col in rating_cols_t2],
        'Median': [table2_organized[col].median() for col in rating_cols_t2],
        'Std': [table2_organized[col].std() for col in rating_cols_t2],
        'Count': [table2_organized[col].count() for col in rating_cols_t2]
    })
    
    return rater_stats_t1, rater_stats_t2


def plot_rater_statistics(table1_organized, table2_organized, rater_stats_t1, rater_stats_t2, per_rater_path):
    """Create visualizations showing statistics per rater."""
    # Plot 1: Mean ratings per rater
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    rater_labels_t1 = [f"R{i+1}" for i in range(len(rater_stats_t1))]
    axes[0].bar(rater_labels_t1, rater_stats_t1['Mean'], yerr=rater_stats_t1['Std'], 
                capsize=5, alpha=0.7, color='#1f77b4')
    axes[0].set_xlabel("Rater")
    axes[0].set_ylabel("Mean Rating")
    axes[0].set_title("Mean Rating per Rater - Full Image (Table 1)")
    axes[0].set_ylim(0, 6)
    axes[0].axhline(y=rater_stats_t1['Mean'].mean(), color='r', linestyle='--', label='Overall Mean')
    axes[0].legend()
    
    rater_labels_t2 = [f"R{i+1}" for i in range(len(rater_stats_t2))]
    axes[1].bar(rater_labels_t2, rater_stats_t2['Mean'], yerr=rater_stats_t2['Std'], 
                capsize=5, alpha=0.7, color='#ff7f0e')
    axes[1].set_xlabel("Rater")
    axes[1].set_ylabel("Mean Rating")
    axes[1].set_title("Mean Rating per Rater - Bounding Box (Table 2)")
    axes[1].set_ylim(0, 6)
    axes[1].axhline(y=rater_stats_t2['Mean'].mean(), color='r', linestyle='--', label='Overall Mean')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(per_rater_path / "mean_ratings_per_rater.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Box plots showing distribution of ratings per rater
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    rating_cols_t1 = [col for col in table1_organized.columns if col.startswith("Rater_")]
    data_t1 = [table1_organized[col].dropna().values for col in rating_cols_t1]
    axes[0].boxplot(data_t1, tick_labels=rater_labels_t1)
    axes[0].set_xlabel("Rater")
    axes[0].set_ylabel("Rating")
    axes[0].set_title("Rating Distribution per Rater - Full Image (Table 1)")
    axes[0].set_ylim(0, 6)
    axes[0].grid(axis='y', alpha=0.3)
    
    rating_cols_t2 = [col for col in table2_organized.columns if col.startswith("Rater_")]
    data_t2 = [table2_organized[col].dropna().values for col in rating_cols_t2]
    axes[1].boxplot(data_t2, tick_labels=rater_labels_t2)
    axes[1].set_xlabel("Rater")
    axes[1].set_ylabel("Rating")
    axes[1].set_title("Rating Distribution per Rater - Bounding Box (Table 2)")
    axes[1].set_ylim(0, 6)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(per_rater_path / "rating_distribution_per_rater.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_organized_data(table1_organized, table2_organized, rater_stats_t1, rater_stats_t2, per_image_path, per_rater_path):
    """Save organized data to CSV files."""
    table1_organized.to_csv(per_image_path / "table1_organized.csv", index=False, float_format='%.3f')
    table2_organized.to_csv(per_image_path / "table2_organized.csv", index=False, float_format='%.3f')
    rater_stats_t1.to_csv(per_rater_path / "rater_statistics_table1.csv", index=False, float_format='%.3f')
    rater_stats_t2.to_csv(per_rater_path / "rater_statistics_table2.csv", index=False, float_format='%.3f')


def print_summary_statistics(table1_organized, table2_organized, output_path):
    """Print summary statistics to console."""
    print(f"Visualizations saved to: {output_path}")
    print(f"\nTable 1 (Full Image) Summary:")
    print(f"  Mean Rating: {table1_organized['mean_rating'].mean():.3f}")
    print(f"  Median Rating: {table1_organized['median_rating'].median():.3f}")
    print(f"  Number of Images: {len(table1_organized)}")

    print(f"\nTable 2 (Bounding Box) Summary:")
    print(f"  Mean Rating: {table2_organized['mean_rating'].mean():.3f}")
    print(f"  Median Rating: {table2_organized['median_rating'].median():.3f}")
    print(f"  Number of Images: {len(table2_organized)}")



def plot_boxplot_by_density(
    df,
    save_path=None,
    title="Rating distribution by breastjjj density",
    figsize=(10, 6),
    dpi=300,
    font_sizes=None,
):
    """Create boxplot showing mean ratings per image grouped by breast density."""
    import numpy as np
    import matplotlib.pyplot as plt

    # Default font sizes (override by passing font_sizes dict)
    fs = {
        "title": 18,
        "axis": 16,
        "ticks": 14,
        "stats": 12,  # not plotted, just for potential future annotations
    }
    if font_sizes:
        fs.update(font_sizes)

    densities = ["DENSITY A", "DENSITY B", "DENSITY C", "DENSITY D"]
    colors = ["#6FA8DC", "#93C47D", "#FFE599", "#E06666"]  # Blue, Green, Yellow, Red

    # Collect mean ratings for each density
    data_by_density = []
    labels = []
    stats_info = []

    for density in densities:
        density_df = df[df["breast_density"] == density]
        if len(density_df) > 0:
            mean_ratings = density_df["mean_rating"].dropna().values
            if len(mean_ratings) > 0:
                data_by_density.append(mean_ratings)
                labels.append(density.replace("DENSITY ", ""))
                stats_info.append(
                    {
                        "density": density,
                        "n_images": len(mean_ratings),
                        "mean": mean_ratings.mean(),
                        "median": np.median(mean_ratings),
                        "min": mean_ratings.min(),
                        "max": mean_ratings.max(),
                    }
                )

    # Print statistics
    print(f"\n{title}:")
    for stat in stats_info:
        print(
            f"  {stat['density']}: n_images={stat['n_images']}, "
            f"mean={stat['mean']:.3f}, median={stat['median']:.3f}, "
            f"range=[{stat['min']:.2f}, {stat['max']:.2f}]"
        )

    # Create boxplot
    fig, ax = plt.subplots(figsize=figsize)

    positions = np.arange(len(data_by_density)) * 0.8 + 1  

    bp = ax.boxplot(
        data_by_density,
        positions=positions,
        tick_labels=labels,
        showfliers=True,
        patch_artist=True,
        widths=0.35
    )

    ax.set_xticks(positions)

    # Color the boxes + make lines a bit thicker
    for patch, color in zip(bp["boxes"], colors[: len(data_by_density)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(1.5)

    for k in ["whiskers", "caps", "medians"]:
        for artist in bp[k]:
            artist.set_linewidth(1.5)

    # Bigger text
    ax.set_xlabel("Breast Density", fontsize=fs["axis"])
    ax.set_ylabel("Mean Rating per Image", fontsize=fs["axis"])
    ax.set_title(title, fontsize=fs["title"], pad=12)

    # Bigger tick labels
    ax.tick_params(axis="both", which="major", labelsize=fs["ticks"])

    ax.set_ylim(0.5, 5.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_annotation_area_vs_rating(
    df,
    save_path=None,
    title="Annotation Area vs Mean Rating",
    figsize=(10, 6),
    dpi=300,
    font_sizes=None,
):
    """Create scatter plot showing relationship between annotation area and mean rating."""
    from scipy import stats
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import ast
    from pathlib import Path

    # Default font sizes
    fs = {
        "title": 18,
        "axis": 16,
        "ticks": 14,
        "legend": 14,
    }
    if font_sizes:
        fs.update(font_sizes)

    # Load radiologist dataset
    csv_file = METADATA_ROOT / Path("radiologist_dataset.csv")
    df_radio = pd.read_csv(csv_file)

    # Calculate annotation area for each image
    annotation_areas = {}
    for _, row in df_radio.iterrows():
        image_id = row['image_id'].replace('.png', '')
        key = f"{row['radiologist_id']}_{image_id}"

        xmin_list = ast.literal_eval(row['resized_xmin'])
        ymin_list = ast.literal_eval(row['resized_ymin'])
        xmax_list = ast.literal_eval(row['resized_xmax'])
        ymax_list = ast.literal_eval(row['resized_ymax'])

        area = 0
        for i in range(len(xmin_list)):
            area += (xmax_list[i] - xmin_list[i]) * (ymax_list[i] - ymin_list[i])

        if area <= 45000:
            annotation_areas[key] = area

    # Map + rescale
    df['annotation_area'] = df['image_id'].map(annotation_areas)
    df['annotation_area'] = df['annotation_area'] / 10000.0

    plot_df = df[['annotation_area', 'mean_rating']].dropna()

    if len(plot_df) == 0:
        print("No data available for annotation area vs rating plot")
        return

    # Regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        plot_df['annotation_area'],
        plot_df['mean_rating']
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter
    ax.scatter(
        plot_df['annotation_area'],
        plot_df['mean_rating'],
        alpha=0.6,
        s=60,
        edgecolors='black',
        linewidth=0.7
    )

    # Regression line
    x_line = np.array([
        plot_df['annotation_area'].min(),
        plot_df['annotation_area'].max()
    ])
    y_line = slope * x_line + intercept

    ax.plot(
        x_line,
        y_line,
        'r--',
        linewidth=2,
        label=f'y = {slope:.4f}x + {intercept:.2f}\n'
              f'R² = {r_value**2:.3f}, p = {p_value:.4f}'
    )

    # Bigger text
    ax.set_xlabel("Annotation Area (x10⁴ pixels²)", fontsize=fs["axis"])
    ax.set_ylabel("Mean Rating per Image", fontsize=fs["axis"])
    ax.set_title(title, fontsize=fs["title"], pad=12)

    # Bigger tick labels
    ax.tick_params(axis='both', which='major', labelsize=fs["ticks"], width=1.2)

    # Bigger legend
    ax.legend(loc='best', fontsize=fs["legend"])

    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Print stats
    print(f"\n{title}:")
    print(f"  Number of images: {len(plot_df)}")
    print(f"  Correlation coefficient (R): {r_value:.3f}")
    print(f"  R-squared: {r_value**2:.3f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Slope: {slope:.6f}")
    print(f"  Intercept: {intercept:.3f}")

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)



def main():
    """Main function to orchestrate the radiologist assessment analysis."""
    # Set up paths
    excel_file = METADATA_ROOT / Path("Radiologist Assessment of Counterfactuals A-E.xlsx")
    output_path = DATA_ROOT / Path("results/radiologist_assessment")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Clear output directory before starting
    clear_output_directory(output_path)
    
    # Create subfolders
    per_image_path = output_path / "per_image"
    per_rater_path = output_path / "per_rater"
    per_density_path = output_path / "per_density"
    per_birads_path = output_path / "per_birads"
    per_image_path.mkdir(parents=True, exist_ok=True)
    per_rater_path.mkdir(parents=True, exist_ok=True)
    per_density_path.mkdir(parents=True, exist_ok=True)
    per_birads_path.mkdir(parents=True, exist_ok=True)

    # Load the data
    df_table1, df_table2 = load_data(excel_file)
    density_map, birads_map = load_radiologist_dataset()

    # Organize tables
    table1_organized = organize_table_data(df_table1, density_map, birads_map)
    table2_organized = organize_table_data(df_table2, density_map, birads_map)

    # Calculate statistics per image
    table1_organized = calculate_stats(table1_organized)
    table2_organized = calculate_stats(table2_organized)

    # Calculate statistics per rater
    rater_stats_t1, rater_stats_t2 = calculate_rater_statistics(table1_organized, table2_organized)

    # Create visualizations
    all_ratings_t1, all_ratings_t2 = plot_rating_distributions(table1_organized, table2_organized, per_image_path)
    plot_mean_ratings_distributions(table1_organized, table2_organized, per_image_path)
    plot_mean_ratings_by_density(table1_organized, table2_organized, per_density_path)
    plot_mean_ratings_by_birads(table1_organized, table2_organized, per_birads_path)
    plot_summary_statistics(table1_organized, table2_organized, all_ratings_t1, all_ratings_t2, output_path)
    plot_rater_statistics(table1_organized, table2_organized, rater_stats_t1, rater_stats_t2, per_rater_path)

    # Save organized data
    save_organized_data(table1_organized, table2_organized, rater_stats_t1, rater_stats_t2, per_image_path, per_rater_path)

    # Print summary statistics
    print_summary_statistics(table1_organized, table2_organized, output_path)

    plot_boxplot_by_density(table1_organized, save_path=output_path / "boxplot_by_density_table1.png", title="Rating Distribution by Breast Density")
    plot_annotation_area_vs_rating(table1_organized, save_path=output_path / "annotation_area_vs_rating_table1.png", title="Annotation Area vs Mean Rating")

if __name__ == "__main__":
    main()