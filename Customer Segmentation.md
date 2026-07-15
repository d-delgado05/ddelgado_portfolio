# Customer Segmentation Analysis - Mall Customers Dataset

## Overview
This analysis performs comprehensive customer segmentation on the mall_customers dataset using K-means clustering to identify distinct customer personas and develop targeted marketing strategies.

## Analysis Results

### Customer Segments Identified (5 Personas)

1. **Segment 1: Budget-Conscious Middle-Aged** (10% of customers)
   - Age: 46 years | Income: $27k | Spending: 18/100
   - Strategy: Deep discounts, essential products, budget brands

2. **Segment 2: Value-Seeking Young Adults** (27% of customers) - **LARGEST SEGMENT**
   - Age: 25 years | Income: $41k | Spending: 62/100
   - Strategy: Value promotions, loyalty programs, digital marketing, trendy products

3. **Segment 3: Premium High-Value Customers** (20% of customers) - **HIGHEST SPENDING**
   - Age: 33 years | Income: $86k | Spending: 82/100
   - Strategy: VIP programs, luxury products, personalized service

4. **Segment 4: High-Income Low-Engagement** (19.5% of customers) - **HIGHEST INCOME**
   - Age: 40 years | Income: $86k | Spending: 19/100
   - Strategy: Engagement campaigns, product discovery, event marketing

5. **Segment 5: Balanced Mature Customers** (23.5% of customers)
   - Age: 56 years | Income: $54k | Spending: 49/100
   - Strategy: Moderate promotions, traditional channels, comfort & quality focus

## Key Questions Answered

### 1. What type of customers exist?
Five distinct customer segments were identified based on age, income, and spending behavior:
- Budget-conscious middle-aged customers
- Value-seeking young adults
- Premium high-value customers
- High-income but low-engagement customers
- Balanced mature customers

### 2. How do they behave?
- **Segment 2** (Young, medium income, high spending): Largest group, active spenders despite moderate income
- **Segment 3** (Middle-aged, high income, high spending): Highest spending segment, ideal for premium offerings
- **Segment 4** (Middle-aged, high income, low spending): High potential but currently low engagement
- **Segment 1** (Middle-aged, low income, low spending): Price-sensitive, requires aggressive pricing
- **Segment 5** (Mature, medium income, medium spending): Balanced behavior, traditional preferences

### 3. What strategies should target each segment?
Each segment has specific strategic recommendations detailed in the report, including:
- Pricing strategies (premium vs. discount)
- Marketing channels (digital vs. traditional)
- Product focus (luxury vs. value)
- Engagement tactics (VIP programs vs. sales events)

## Generated Files

1. **customer_segmentation_report.txt** - Comprehensive text report with all findings
2. **customers_with_segments.csv** - Original data with cluster assignments
3. **data_distributions.png** - Data exploration visualizations
4. **elbow_method.png** - Optimal cluster selection analysis
5. **segments_income_spending.png** - Income vs. Spending Score visualization
6. **segments_age_spending.png** - Age vs. Spending Score visualization
7. **segments_3d.png** - 3D visualization of all segments
8. **cluster_comparison.png** - Bar charts comparing cluster characteristics
9. **cluster_distribution.png** - Segment size distribution

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the analysis
python customer_segmentation_analysis.py
```

## Methodology

- **Clustering Algorithm**: K-means clustering (k=5)
- **Features Used**: Age, Annual Income, Spending Score
- **Preprocessing**: StandardScaler for feature normalization
- **Validation**: Elbow method for optimal cluster selection

## Next Steps

1. Review the detailed report and visualizations
2. Implement segment-specific marketing campaigns
3. Monitor segment evolution over time
4. Test and refine strategies based on customer response
5. Consider RFM analysis for additional insights if transaction data becomes available

