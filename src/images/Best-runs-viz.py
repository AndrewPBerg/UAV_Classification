import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re

# Load the CSV file
csv_path = "Best Results Export Aug 1 2025.csv"
df = pd.read_csv(csv_path)

def extract_model_type_from_name(name):
    """Extract model type from the experiment name"""
    if pd.isna(name):
        return "unknown"
    
    # Extract the model type from names like "custom_cnn-3augs-31-class-kfold"
    # or "efficientnet_b0-none-full-3augs-31-class-kfold"
    name_lower = str(name).lower()
    
    if 'custom_cnn' in name_lower:
        return 'Custom CNN (Ours)'
    elif 'efficientnet_b0' in name_lower:
        return 'EfficientNet'
    elif 'mobilenet_v3_large' in name_lower:
        return 'MobileNet'
    elif 'resnet18' in name_lower:
        return 'ResNet'
    elif 'ast' in name_lower:
        return 'Audio Spectrogram Transformer (AST)'
    elif 'efficientnet_b7' in name_lower:
        return 'EfficientNet'
    elif 'mobilenet_v3_small' in name_lower:
        return 'MobileNet'
    else:
        # Try to extract the first part before the first dash as model type
        model_type = name.split('-')[0] if '-' in name else name
        return model_type

# Extract model types and create the DataFrame directly
model_data = []
for idx, row in df.iterrows():
    model_type = extract_model_type_from_name(row['Name'])
    experiment_name = row['Name']
    accuracy = row['final_average_val_acc']
    std_accuracy = row['final_std_val_acc']
    
    model_data.append({
        'model_type': model_type,
        'experiment_name': experiment_name,
        'accuracy': accuracy,
        'std_accuracy': std_accuracy,
        'label': f"{model_type}"
    })

# Create DataFrame from the extracted data
model_df = pd.DataFrame(model_data)

# Convert accuracy and std_accuracy to percentages
model_df['accuracy'] = model_df['accuracy'] * 100
model_df['std_accuracy'] = model_df['std_accuracy'] * 100

# Sort by accuracy from least to highest
model_df = model_df.sort_values('accuracy', ascending=True).reset_index(drop=True)

# Set up the plotting style - Apple-inspired clean minimal design
# Following principles: simplicity, generous white space, minimal color palette,
# elegant typography, and focus on content over decoration
sns.set_style("white")  # Clean white background, no grid
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SF Pro Display', 'Helvetica', 'Arial', 'DejaVu Sans']

# Create the main plot with more generous spacing
fig, ax = plt.subplots(1, 1, figsize=(12.1, 9))

# Apple-inspired minimal color palette - grays with one accent color
accent_color = '#007AFF'  # Apple's signature blue
light_gray = '#8E8E93'    # Apple's secondary text color
medium_gray = '#6D6D70'   # Apple's tertiary text color
dark_gray = '#1D1D1F'     # Apple's primary text color

# Use shapes and gradients - different markers for each model type with elegant gradients
unique_models = model_df['model_type'].unique()
n_models = len(unique_models)

# Define distinct marker shapes for different model types
markers = ['o', 's', '^', 'D', 'v', '*', 'p']  # circle, square, triangle, diamond, triangle_down, star, pentagon
marker_map = dict(zip(unique_models, markers[:len(unique_models)]))

# Create gradient colors - subtle grays to accent blue with transparency gradients
colors = []
alphas = []
for i in range(n_models):
    if i == n_models - 1:  # Highlight the best performing model
        colors.append(accent_color)
        alphas.append(1.0)
    elif i == n_models - 2:  # Second best gets partial accent color
        colors.append('#5AC8FA')  # Lighter blue
        alphas.append(0.9)
    else:
        # Create gradient from dark to light gray
        gray_intensity = 0.2 + (i / max(1, n_models - 3)) * 0.5
        colors.append(plt.cm.gray(gray_intensity))
        alphas.append(0.7 + (i / max(1, n_models - 1)) * 0.2)

color_map = dict(zip(unique_models, colors))
alpha_map = dict(zip(unique_models, alphas))

# Create the line plot with error bars
x_pos = range(len(model_df))

# Subtle connecting line
ax.plot(x_pos, model_df['accuracy'], 
        color=light_gray, linewidth=1.5, alpha=0.6, zorder=1)

# Plot each point with different shapes and gradient styling
for i, (x, row) in enumerate(zip(x_pos, model_df.itertuples())):
    model_type = row.model_type
    accuracy = row.accuracy
    std_accuracy = row.std_accuracy
    
    # Determine if this is the best performing model
    is_best = i == len(model_df) - 1  # Last in sorted order = highest accuracy
    is_second_best = i == len(model_df) - 2
    
    # Size based on performance ranking
    marker_size = 10
    if is_best:
        marker_size = 16
    elif is_second_best:
        marker_size = 14
    elif i >= len(model_df) - 3:  # Third best
        marker_size = 12
    
    ax.errorbar(x, accuracy, 
                yerr=std_accuracy,
                fmt=marker_map[model_type],  # Different marker shapes
                capsize=0,  # Remove caps for cleaner look
                capthick=0,
                markersize=marker_size,
                markerfacecolor=color_map[model_type],
                markeredgecolor='white',  # Subtle white edge
                markeredgewidth=1.5,
                color=color_map[model_type],
                elinewidth=1.5,  # Thinner error bars
                alpha=alpha_map[model_type],  # Gradient alpha values
                zorder=2)

# Customize the plot with Apple's design principles
# ax.set_ylabel('K-Fold Cross-Validation Accuracy (%)', fontsize=27, fontfamily='SF Pro Display', fontweight='300', color=dark_gray)

# Clean, minimal tick styling
ax.tick_params(axis='y', labelsize=23, colors=medium_gray, length=0)
ax.tick_params(axis='x', labelsize=20, colors=medium_gray, length=0)

# Remove all spines except left for a clean minimal look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_color('#3A3A3C')  # Darker gray for better visibility
ax.spines['left'].set_linewidth(0.8)

# Add subtle horizontal reference lines at key points - updated range 93-100
y_ticks = [93, 94, 95, 96, 97, 98, 99, 100]  # Include both start (93) and end (100) range numbers
ax.set_yticks(y_ticks)
for y in y_ticks:
    ax.axhline(y=y, color='#F2F2F7', linewidth=0.5, zorder=0)

# Remove x-axis labels for models as requested - clean minimal x-axis
ax.set_xticks(x_pos)
ax.set_xticklabels([])  # Empty labels to remove model names from x-axis
ax.set_ylim(93, 100)  # Updated range from 93 to 100

# Add all accuracy labels to the right of their respective shapes
for i, (x, acc_val, std_val) in enumerate(zip(x_pos, model_df['accuracy'], model_df['std_accuracy'])):
    # Determine label styling based on performance ranking
    is_best = i == len(model_df) - 1
    is_second_best = i == len(model_df) - 2
    
    if is_best:
        label_color = accent_color
        font_weight = '600'
        font_size = 27  # Increased by additional 50% from 20
    elif is_second_best:
        label_color = '#5AC8FA'  # Light blue
        font_weight = '500'
        font_size = 27  # Increased by additional 50% from 18
    else:
        label_color = medium_gray
        font_weight = '400'
        font_size = 26  # Increased by additional 50% from 17
    
    # Position labels to the right of markers with small offset, slightly lower to avoid clipping
    x_offset = 0.15  # Small horizontal offset to the right
    y_offset = -0.1  # Small downward offset to avoid line clipping
    ax.text(x + x_offset, acc_val + y_offset,  # Positioned slightly below marker center
             f'{acc_val:.1f}%', 
             ha='left', va='center',  # Left-aligned, vertically centered
             fontweight=font_weight, fontsize=font_size, color=label_color)

# Minimal legend with different shapes and gradients
legend_elements = []
for i, model in enumerate(unique_models):
    # Find corresponding position in sorted data for this model
    model_positions = [j for j, row_model in enumerate(model_df['model_type']) if row_model == model]
    if model_positions:
        position = model_positions[0]  # Take first occurrence
        is_best = position == len(model_df) - 1
        is_second_best = position == len(model_df) - 2
        
        # Match the marker size with the plot
        marker_size = 10
        if is_best:
            marker_size = 12
        elif is_second_best:
            marker_size = 11
        
        legend_elements.append(plt.Line2D([0], [0], 
                                         marker=marker_map[model], 
                                         color='w', 
                                         markerfacecolor=color_map[model], 
                                         markersize=marker_size,
                                         markeredgecolor='white', 
                                         markeredgewidth=1.5,
                                         alpha=alpha_map[model]))

legend_elements.reverse()

# Position legend elegantly without frame
ax.legend(legend_elements, unique_models, 
          loc='upper left', 
          frameon=False,  # Remove legend frame for cleaner look
          fontsize=20,  # Increased by 50% from 13
          handlelength=1.5, 
          handletextpad=0.8, 
          labelspacing=0.7,
          labelcolor=medium_gray)

# Apply generous margins and spacing (Apple design loves white space)
plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.15)

# Save with high quality for crisp, clean output
output_path = "/Users/applefella/Documents/UAV_Classification/UAV_Classification/src/images/best_results_model_performance_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
            edgecolor='none', pad_inches=0.1)  # Extra padding for clean borders

print(f"Visualization saved to: {output_path}")

# Display extracted data
print(f"\nExtracted Model Data:")
print(model_df[['model_type', 'experiment_name', 'accuracy', 'std_accuracy']].round(2))

# Print performance ranking
print(f"\nModel Performance Ranking (by accuracy):")
ranking = model_df.sort_values('accuracy', ascending=False)[['model_type', 'accuracy', 'std_accuracy']].round(2)
for i, (_, row) in enumerate(ranking.iterrows(), 1):
    print(f"{i}. {row['model_type']}: {row['accuracy']:.1f}% ± {row['std_accuracy']:.1f}%")

plt.show()
