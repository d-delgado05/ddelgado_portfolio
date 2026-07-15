"""
Customer Segmentation Analysis for Mall Customers Dataset
This script performs comprehensive customer segmentation using K-means clustering
and creates customer personas with strategic recommendations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load data directly from Downloads folder
import os
file_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'Mall_Customers.csv')
segment_df = pd.read_csv(file_path)

print("="*80)
print("CUSTOMER SEGMENTATION ANALYSIS")
print("="*80)

# Descriptive Statistics
print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS")
print("="*60)
print(segment_df.describe())

print("\nMissing Values:")
print(segment_df.isnull().sum())

def explore_data(segment_df):
    """Explore and understand the dataset"""
    print("\n" + "="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    print("\nFirst few rows:")
    print(segment_df.head())
    
    print("\nDataset Info:")
    print(segment_df.info())
    
    print("\nColumn Names:")
    print(segment_df.columns.tolist())
    
    # Visualize distributions
    _plot_distributions(segment_df)
    
def _plot_distributions(segment_df):
    """Plot distributions of key variables"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Age distribution
    axes[0, 0].hist(segment_df['Age'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Age Distribution')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    
    # Annual Income distribution
    income_col = [col for col in segment_df.columns if 'Income' in col or 'income' in col][0]
    axes[0, 1].hist(segment_df[income_col], bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_title('Annual Income Distribution')
    axes[0, 1].set_xlabel('Annual Income (k$)')
    axes[0, 1].set_ylabel('Frequency')
    
    # Spending Score distribution
    spending_col = [col for col in segment_df.columns if 'Spending' in col or 'spending' in col][0]
    axes[1, 0].hist(segment_df[spending_col], bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_title('Spending Score Distribution')
    axes[1, 0].set_xlabel('Spending Score (1-100)')
    axes[1, 0].set_ylabel('Frequency')
    
    # Gender distribution
    gender_col = 'Gender' if 'Gender' in segment_df.columns else 'Genre'
    if gender_col in segment_df.columns:
        gender_counts = segment_df[gender_col].value_counts()
        axes[1, 1].bar(gender_counts.index, gender_counts.values, alpha=0.7, color='purple')
        axes[1, 1].set_title('Gender Distribution')
        axes[1, 1].set_xlabel('Gender')
        axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('data_distributions.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Distribution plots saved as 'data_distributions.png'")
    plt.close()

def prepare_data(segment_df):
    """Prepare data for clustering"""
    print("\n" + "="*60)
    print("DATA PREPARATION")
    print("="*60)
    
    # Identify feature columns
    feature_cols = []
    for col in segment_df.columns:
        if col not in ['CustomerID', 'Gender', 'Genre'] and segment_df[col].dtype in [np.int64, np.float64]:
            feature_cols.append(col)
    
    print(f"\nFeatures selected for clustering: {feature_cols}")
    
    # Handle categorical variables (Gender/Genre)
    gender_col = 'Gender' if 'Gender' in segment_df.columns else 'Genre'
    if gender_col in segment_df.columns:
        segment_df['Gender_Encoded'] = segment_df[gender_col].map({'Male': 0, 'Female': 1})
        feature_cols.append('Gender_Encoded')
    
    # Extract features
    X = segment_df[feature_cols].values
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)
    print(f"[OK] Data scaled. Shape: {scaled_data.shape}")
    
    return feature_cols, scaled_data, scaler

def find_optimal_clusters(scaled_data, max_k=10):
    """Find optimal number of clusters using Elbow Method"""
    print("\n" + "="*60)
    print("FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("="*60)
    
    inertias = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
    print("[OK] Elbow plot saved as 'elbow_method.png'")
    plt.close()
    
    # Calculate rate of change to suggest optimal k
    rate_of_change = np.diff(inertias)
    rate_of_change_2 = np.diff(rate_of_change)
    
    # Find elbow (where rate of change starts to level off)
    # Typically between 4-6 clusters for customer segmentation
    optimal_k = 5  # Default, but we'll let user decide or use 5-6
    
    print(f"\nSuggested number of clusters: 5-6 (based on typical customer segmentation)")
    print(f"Inertia values: {[f'{i:.2f}' for i in inertias]}")
    
    return optimal_k

def perform_clustering(segment_df, scaled_data, n_clusters=5):
    """Perform K-means clustering"""
    print("\n" + "="*60)
    print(f"PERFORMING K-MEANS CLUSTERING (k={n_clusters})")
    print("="*60)
    
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans_model.fit_predict(scaled_data)
    
    segment_df['Cluster'] = clusters
    
    print(f"\n[OK] Clustering completed!")
    print(f"\nCluster distribution:")
    print(segment_df['Cluster'].value_counts().sort_index())
    
    return clusters, kmeans_model

def analyze_segments(segment_df):
    """Analyze each customer segment"""
    print("\n" + "="*60)
    print("SEGMENT ANALYSIS")
    print("="*60)
    
    # Get feature columns
    feature_cols = [col for col in segment_df.columns 
                   if col not in ['CustomerID', 'Gender', 'Genre', 'Cluster', 'Gender_Encoded'] 
                   and segment_df[col].dtype in [np.int64, np.float64]]
    
    segment_profiles = {}
    gender_col = 'Gender' if 'Gender' in segment_df.columns else 'Genre'
    
    for cluster_id in sorted(segment_df['Cluster'].unique()):
        cluster_data = segment_df[segment_df['Cluster'] == cluster_id]
        
        profile = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(segment_df) * 100
        }
        
        # Calculate means for each feature
        for col in feature_cols:
            profile[col] = {
                'mean': cluster_data[col].mean(),
                'std': cluster_data[col].std(),
                'min': cluster_data[col].min(),
                'max': cluster_data[col].max()
            }
        
        # Gender distribution if available
        if gender_col in segment_df.columns:
            profile['gender_dist'] = cluster_data[gender_col].value_counts().to_dict()
        
        segment_profiles[cluster_id] = profile
        
        print(f"\n--- Cluster {cluster_id} ---")
        print(f"Size: {profile['size']} customers ({profile['percentage']:.1f}%)")
        for col in feature_cols:
            mean_val = profile[col]['mean']
            print(f"  {col}: {mean_val:.2f} (std: {profile[col]['std']:.2f})")
        if 'gender_dist' in profile:
            print(f"  Gender: {profile['gender_dist']}")
    
    return segment_profiles

def create_personas(segment_df, segment_profiles):
    """Create customer personas based on segments"""
    print("\n" + "="*60)
    print("CUSTOMER PERSONAS")
    print("="*60)
    
    personas = []
    
    # Get feature columns for easier access
    age_col = [col for col in segment_df.columns if 'Age' in col][0]
    income_col = [col for col in segment_df.columns if 'Income' in col or 'income' in col][0]
    spending_col = [col for col in segment_df.columns if 'Spending' in col or 'spending' in col][0]
    
    for cluster_id in sorted(segment_profiles.keys()):
        profile = segment_profiles[cluster_id]
        
        # Calculate characteristics
        avg_age = profile[age_col]['mean']
        avg_income = profile[income_col]['mean']
        avg_spending = profile[spending_col]['mean']
        
        # Determine persona characteristics
        age_group = "Young" if avg_age < 30 else "Middle-aged" if avg_age < 50 else "Mature"
        income_level = "High" if avg_income > 70 else "Medium" if avg_income > 40 else "Low"
        spending_level = "High" if avg_spending > 60 else "Medium" if avg_spending > 40 else "Low"
        
        # Create persona name and description
        persona_name = f"Segment {cluster_id + 1}"
        persona_desc = f"{age_group} customers with {income_level.lower()} income and {spending_level.lower()} spending"
        
        persona = {
            'cluster_id': cluster_id,
            'name': persona_name,
            'description': persona_desc,
            'size': profile['size'],
            'percentage': profile['percentage'],
            'avg_age': avg_age,
            'avg_income': avg_income,
            'avg_spending': avg_spending,
            'age_group': age_group,
            'income_level': income_level,
            'spending_level': spending_level
        }
        
        personas.append(persona)
        
        print(f"\n{persona_name}: {persona_desc}")
        print(f"  Size: {profile['size']} customers ({profile['percentage']:.1f}%)")
        print(f"  Average Age: {avg_age:.1f} years")
        print(f"  Average Annual Income: ${avg_income:.0f}k")
        print(f"  Average Spending Score: {avg_spending:.1f}/100")
    
    return personas

def generate_strategies(personas):
    """Generate marketing strategies for each persona"""
    print("\n" + "="*60)
    print("MARKETING STRATEGIES BY SEGMENT")
    print("="*60)
    
    strategies = {}
    
    for persona in personas:
        cluster_id = persona['cluster_id']
        income = persona['avg_income']
        spending = persona['avg_spending']
        age = persona['avg_age']
        
        strategy = []
        
        # High income, high spending - Premium/VIP
        if income > 70 and spending > 60:
            strategy.append("VIP/Premium Program: Offer exclusive access, early sales, personal shopping")
            strategy.append("Luxury Products: Focus on high-end brands and premium experiences")
            strategy.append("Loyalty Rewards: High-value rewards and cashback programs")
            strategy.append("Personalized Service: Dedicated account managers and concierge services")
        
        # High income, low spending - Potential
        elif income > 70 and spending < 40:
            strategy.append("Engagement Campaigns: Increase visit frequency with targeted promotions")
            strategy.append("Product Discovery: Showcase new arrivals and exclusive collections")
            strategy.append("Event Marketing: Invite to special events and product launches")
            strategy.append("Cross-sell Opportunities: Introduce complementary high-value products")
        
        # Medium income, high spending - Value Seekers
        elif 40 <= income <= 70 and spending > 60:
            strategy.append("Value Promotions: Discounts, BOGO offers, and bundle deals")
            strategy.append("Loyalty Program: Points-based rewards system")
            strategy.append("Seasonal Sales: Highlight sales events and clearance items")
            strategy.append("Budget-Friendly Options: Emphasize quality at affordable prices")
        
        # Medium income, medium spending - Balanced
        elif 40 <= income <= 70 and 40 <= spending <= 60:
            strategy.append("Moderate Promotions: Regular discounts and seasonal offers")
            strategy.append("Product Mix: Balance of quality and price")
            strategy.append("Engagement: Email newsletters and social media campaigns")
            strategy.append("Loyalty Benefits: Standard rewards program")
        
        # Low income, high spending - Budget Conscious
        elif income < 40 and spending > 60:
            strategy.append("Affordable Options: Focus on budget-friendly products")
            strategy.append("Payment Plans: Installment options and layaway programs")
            strategy.append("Clearance Sales: Regular clearance and discount events")
            strategy.append("Value Messaging: Emphasize deals and savings")
        
        # Low income, low spending - Price Sensitive
        elif income < 40 and spending < 40:
            strategy.append("Deep Discounts: Aggressive pricing and clearance sales")
            strategy.append("Essential Products: Focus on necessities and basics")
            strategy.append("Budget Brands: Promote affordable brand options")
            strategy.append("Seasonal Promotions: Major sales events (Black Friday, etc.)")
        
        # Age-based adjustments
        if age < 30:
            strategy.append("Digital Marketing: Social media, influencer partnerships, mobile apps")
            strategy.append("Trendy Products: Latest fashion and tech trends")
        elif age >= 50:
            strategy.append("Traditional Channels: Email, direct mail, in-store promotions")
            strategy.append("Comfort & Quality: Focus on comfort, durability, and classic styles")
        
        strategies[cluster_id] = strategy
        
        print(f"\n--- {persona['name']} Strategy ---")
        for i, s in enumerate(strategy, 1):
            print(f"  {i}. {s}")
    
    return strategies

def visualize_segments(segment_df):
    """Create visualizations of customer segments"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Get column names
    age_col = [col for col in segment_df.columns if 'Age' in col][0]
    income_col = [col for col in segment_df.columns if 'Income' in col or 'income' in col][0]
    spending_col = [col for col in segment_df.columns if 'Spending' in col or 'spending' in col][0]
    
    # 1. 2D Scatter Plot: Income vs Spending Score
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(segment_df[income_col], segment_df[spending_col], 
                        c=segment_df['Cluster'], cmap='viridis', 
                        s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    plt.xlabel('Annual Income (k$)', fontsize=12)
    plt.ylabel('Spending Score (1-100)', fontsize=12)
    plt.title('Customer Segments: Income vs Spending Score', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.savefig('segments_income_spending.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: segments_income_spending.png")
    plt.close()
    
    # 2. 2D Scatter Plot: Age vs Spending Score
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(segment_df[age_col], segment_df[spending_col], 
                        c=segment_df['Cluster'], cmap='plasma', 
                        s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Spending Score (1-100)', fontsize=12)
    plt.title('Customer Segments: Age vs Spending Score', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.savefig('segments_age_spending.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: segments_age_spending.png")
    plt.close()
    
    # 3. 3D visualization (2D projection)
    try:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(segment_df[age_col], segment_df[income_col], segment_df[spending_col],
                           c=segment_df['Cluster'], cmap='viridis', s=100, alpha=0.6)
        ax.set_xlabel('Age', fontsize=11)
        ax.set_ylabel('Annual Income (k$)', fontsize=11)
        ax.set_zlabel('Spending Score (1-100)', fontsize=11)
        ax.set_title('3D Customer Segments', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster', ax=ax)
        plt.savefig('segments_3d.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: segments_3d.png")
        plt.close()
    except ImportError:
        print("[WARNING] 3D plotting not available, skipping 3D visualization")
    
    # 4. Cluster comparison bar charts
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    cluster_means = segment_df.groupby('Cluster').agg({
        age_col: 'mean',
        income_col: 'mean',
        spending_col: 'mean'
    })
    
    cluster_means[age_col].plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
    axes[0].set_title('Average Age by Cluster', fontweight='bold')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Age')
    axes[0].tick_params(axis='x', rotation=0)
    
    cluster_means[income_col].plot(kind='bar', ax=axes[1], color='lightgreen', edgecolor='black')
    axes[1].set_title('Average Annual Income by Cluster', fontweight='bold')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Annual Income (k$)')
    axes[1].tick_params(axis='x', rotation=0)
    
    cluster_means[spending_col].plot(kind='bar', ax=axes[2], color='salmon', edgecolor='black')
    axes[2].set_title('Average Spending Score by Cluster', fontweight='bold')
    axes[2].set_xlabel('Cluster')
    axes[2].set_ylabel('Spending Score')
    axes[2].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig('cluster_comparison.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: cluster_comparison.png")
    plt.close()
    
    # 5. Cluster size distribution
    plt.figure(figsize=(10, 6))
    cluster_counts = segment_df['Cluster'].value_counts().sort_index()
    colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_counts)))
    bars = plt.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='black')
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Number of Customers', fontsize=12)
    plt.title('Customer Distribution Across Segments', fontsize=14, fontweight='bold')
    plt.xticks(cluster_counts.index)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(segment_df)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('cluster_distribution.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: cluster_distribution.png")
    plt.close()

def generate_report(segment_df, personas, strategies, segment_profiles, n_clusters):
    """Generate a comprehensive report"""
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)
    
    report = []
    report.append("="*80)
    report.append("CUSTOMER SEGMENTATION ANALYSIS REPORT")
    report.append("="*80)
    report.append("")
    report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Customers Analyzed: {len(segment_df)}")
    report.append(f"Number of Segments: {n_clusters}")
    report.append("")
    
    # Executive Summary
    report.append("="*80)
    report.append("EXECUTIVE SUMMARY")
    report.append("="*80)
    report.append("")
    report.append("This analysis identifies distinct customer segments based on demographic")
    report.append("and behavioral characteristics. Each segment represents a unique customer")
    report.append("persona with specific needs, preferences, and spending patterns.")
    report.append("")
    
    # Detailed Personas
    report.append("="*80)
    report.append("CUSTOMER PERSONAS")
    report.append("="*80)
    report.append("")
    
    for persona in personas:
        report.append(f"\n{persona['name']}: {persona['description']}")
        report.append("-" * 80)
        report.append(f"  Segment Size: {persona['size']} customers ({persona['percentage']:.1f}% of total)")
        report.append(f"  Average Age: {persona['avg_age']:.1f} years")
        report.append(f"  Average Annual Income: ${persona['avg_income']:.0f},000")
        report.append(f"  Average Spending Score: {persona['avg_spending']:.1f}/100")
        report.append(f"  Characteristics: {persona['age_group']}, {persona['income_level']} income, {persona['spending_level']} spending")
        report.append("")
    
    # Strategies
    report.append("="*80)
    report.append("MARKETING STRATEGIES BY SEGMENT")
    report.append("="*80)
    report.append("")
    
    for persona in personas:
        cluster_id = persona['cluster_id']
        report.append(f"\n{persona['name']} - Strategic Recommendations:")
        report.append("-" * 80)
        for i, strategy in enumerate(strategies[cluster_id], 1):
            report.append(f"  {i}. {strategy}")
        report.append("")
    
    # Key Insights
    report.append("="*80)
    report.append("KEY INSIGHTS")
    report.append("="*80)
    report.append("")
    
    # Find largest segment
    largest_segment = max(personas, key=lambda x: x['size'])
    report.append(f"• Largest Segment: {largest_segment['name']} with {largest_segment['size']} customers")
    
    # Find highest spending segment
    highest_spending = max(personas, key=lambda x: x['avg_spending'])
    report.append(f"• Highest Spending Segment: {highest_spending['name']} with avg score {highest_spending['avg_spending']:.1f}")
    
    # Find highest income segment
    highest_income = max(personas, key=lambda x: x['avg_income'])
    report.append(f"• Highest Income Segment: {highest_income['name']} with avg income ${highest_income['avg_income']:.0f}k")
    
    report.append("")
    report.append("="*80)
    report.append("RECOMMENDATIONS")
    report.append("="*80)
    report.append("")
    report.append("1. Personalize marketing campaigns based on segment characteristics")
    report.append("2. Develop segment-specific product offerings and promotions")
    report.append("3. Allocate marketing budget proportionally to segment value")
    report.append("4. Monitor segment evolution over time")
    report.append("5. Test and refine strategies based on segment response")
    report.append("")
    
    # Save report
    report_text = "\n".join(report)
    with open('customer_segmentation_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("[OK] Report saved as 'customer_segmentation_report.txt'")
    print("\n" + report_text)
    
    return report_text

# Run the complete analysis
if __name__ == "__main__":
    # Explore data
    explore_data(segment_df)
    
    # Prepare data
    feature_cols, scaled_data, scaler = prepare_data(segment_df)
    
    # Find optimal clusters
    optimal_k = find_optimal_clusters(scaled_data)
    
    # Perform clustering
    n_clusters = 5
    clusters, kmeans_model = perform_clustering(segment_df, scaled_data, n_clusters)
    
    # Analyze segments
    segment_profiles = analyze_segments(segment_df)
    
    # Create personas
    personas = create_personas(segment_df, segment_profiles)
    
    # Generate strategies
    strategies = generate_strategies(personas)
    
    # Visualize
    visualize_segments(segment_df)
    
    # Generate report
    generate_report(segment_df, personas, strategies, segment_profiles, n_clusters)
    
    # Save results
    output_file = 'customers_with_segments.csv'
    segment_df.to_csv(output_file, index=False)
    print(f"\n[OK] Segmented data saved as '{output_file}'")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("  - customer_segmentation_report.txt (Detailed report)")
    print("  - customers_with_segments.csv (Data with cluster assignments)")
    print("  - data_distributions.png (Data exploration)")
    print("  - elbow_method.png (Optimal cluster selection)")
    print("  - segments_income_spending.png (Segment visualization)")
    print("  - segments_age_spending.png (Segment visualization)")
    print("  - segments_3d.png (3D segment visualization)")
    print("  - cluster_comparison.png (Cluster comparisons)")
    print("  - cluster_distribution.png (Segment sizes)")

