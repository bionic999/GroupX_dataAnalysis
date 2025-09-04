import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

class SouthAfricaAnalyzer:
    """
    Comprehensive analysis class for South Africa employment and poverty data.
    """
    
    def __init__(self, data_path="../data/processed/south_africa_clean.csv"):
        self.data_path = data_path
        self.data = None
        self.complete_data = None  
        
      
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_data(self):
        """Load the cleaned data."""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully: {self.data.shape}")
            
            self.complete_data = self.data.dropna(subset=['employment_rate', 'poverty_rate'])
            print(f"Complete records for correlation analysis: {len(self.complete_data)}")
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def numerical_analysis_with_numpy(self):
        """Perform numerical analysis using NumPy operations."""
        print("\n=== NUMERICAL ANALYSIS WITH NUMPY ===")
        
        # Convert to numpy arrays for mathematical operations
        employment_rates = self.data['employment_rate'].values
        poverty_rates = self.complete_data['poverty_rate'].values
        
        print("1. Array Shape Analysis:")
        print(f"   Employment data shape: {employment_rates.shape}")
        print(f"   Poverty data shape: {poverty_rates.shape}")
        
        
        print("\n2. Array Reshaping Operations:")
        
        emp_2d = employment_rates.reshape(-1, 1)
        print(f"   Employment reshaped to 2D: {emp_2d.shape}")
        
        # Create a matrix of yearly differences
        emp_diff = np.diff(employment_rates)
        print(f"   Employment year-over-year changes: {emp_diff.shape}")
        
        print("\n3. Mathematical Operations:")
        
        # Employment statistics
        emp_mean = np.mean(employment_rates)
        emp_std = np.std(employment_rates)
        emp_median = np.median(employment_rates)
        emp_min, emp_max = np.min(employment_rates), np.max(employment_rates)
        
        print(f"   Employment Rate Statistics:")
        print(f"     Mean: {emp_mean:.2f}%")
        print(f"     Std Dev: {emp_std:.2f}%")
        print(f"     Median: {emp_median:.2f}%")
        print(f"     Range: {emp_min:.2f}% - {emp_max:.2f}%")
        
        # Poverty statistics  
        pov_mean = np.mean(poverty_rates)
        pov_std = np.std(poverty_rates)
        pov_median = np.median(poverty_rates)
        pov_min, pov_max = np.min(poverty_rates), np.max(poverty_rates)
        
        print(f"   Poverty Rate Statistics:")
        print(f"     Mean: {pov_mean:.2f}%")
        print(f"     Std Dev: {pov_std:.2f}%")
        print(f"     Median: {pov_median:.2f}%")
        print(f"     Range: {pov_min:.2f}% - {pov_max:.2f}%")
        
        print("\n4. Advanced Array Operations:")
        
        # Percentile calculations
        emp_percentiles = np.percentile(employment_rates, [25, 50, 75])
        pov_percentiles = np.percentile(poverty_rates, [25, 50, 75])
        
        print(f"   Employment Percentiles (25th, 50th, 75th): {emp_percentiles}")
        print(f"   Poverty Percentiles (25th, 50th, 75th): {pov_percentiles}")
        
        
        window_size = 3
        if len(employment_rates) >= window_size:
            emp_moving_avg = np.convolve(employment_rates, np.ones(window_size)/window_size, mode='valid')
            print(f"   3-year moving average (last 5 values): {emp_moving_avg[-5:]}")
        
        # Identify trends
        emp_trend = np.polyfit(range(len(employment_rates)), employment_rates, 1)
        print(f"   Employment trend (slope per year): {emp_trend[0]:.3f}%")
        
        return {
            'employment_stats': {'mean': emp_mean, 'std': emp_std, 'min': emp_min, 'max': emp_max},
            'poverty_stats': {'mean': pov_mean, 'std': pov_std, 'min': pov_min, 'max': pov_max},
            'trend': emp_trend[0]
        }
    
    def correlation_analysis(self):
        """Perform correlation analysis between employment and poverty rates."""
        print("\n=== CORRELATION ANALYSIS ===")
        
        if len(self.complete_data) < 3:
            print("Insufficient data points for correlation analysis")
            return None
            
        employment = self.complete_data['employment_rate'].values
        poverty = self.complete_data['poverty_rate'].values
        
       
        pearson_corr, pearson_p = pearsonr(employment, poverty)
        
        spearman_corr, spearman_p = spearmanr(employment, poverty)
        
        print(f"Pearson Correlation: {pearson_corr:.3f} (p-value: {pearson_p:.3f})")
        print(f"Spearman Correlation: {spearman_corr:.3f} (p-value: {spearman_p:.3f})")
        
       
        if abs(pearson_corr) > 0.7:
            strength = "strong"
        elif abs(pearson_corr) > 0.5:
            strength = "moderate"
        elif abs(pearson_corr) > 0.3:
            strength = "weak"
        else:
            strength = "very weak"
            
        direction = "negative" if pearson_corr < 0 else "positive"
        
        print(f"Interpretation: {strength} {direction} correlation between employment and poverty rates")
        
        if pearson_p < 0.05:
            print("The correlation is statistically significant (p < 0.05)")
        else:
            print("The correlation is not statistically significant (p ≥ 0.05)")
            
        return {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'p_values': {'pearson': pearson_p, 'spearman': spearman_p}
        }
    
    def create_visualizations(self):
        """Create comprehensive visualizations as separate charts."""
        print("\n=== CREATING VISUALIZATIONS ===")
        
        output_paths = []
        
        # 1. Employment Time Series Chart
        plt.figure(figsize=(12, 8))
        plt.plot(self.data['year'], self.data['employment_rate'], 
                marker='o', linewidth=3, markersize=8, color='#2E86AB', label='Employment Rate')
        plt.title('South Africa Employment Rate Over Time (1995-2023)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Employment Rate (%)', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=12)
        
        #trend line
        years = self.data['year'].values
        employment = self.data['employment_rate'].values
        z = np.polyfit(years, employment, 1)
        p = np.poly1d(z)
        plt.plot(years, p(years), "r--", alpha=0.8, linewidth=2, label=f'Trend (slope: {z[0]:.3f}%/year)')
        plt.legend(fontsize=12)
        
        # Highlight significant events
        plt.axvline(x=1997, color='orange', linestyle=':', alpha=0.7, label='Asian Financial Crisis')
        plt.axvline(x=2008, color='red', linestyle=':', alpha=0.7, label='Global Financial Crisis')
        plt.axvline(x=2020, color='purple', linestyle=':', alpha=0.7, label='COVID-19 Pandemic')
        
        plt.tight_layout()
        employment_ts_path = "../outputs/visualizations/employment_timeseries.png"
        plt.savefig(employment_ts_path, dpi=300, bbox_inches='tight')
        output_paths.append(employment_ts_path)
        plt.show()
        
        # 2. Poverty Time Series Chart
        poverty_data = self.data.dropna(subset=['poverty_rate'])
        if len(poverty_data) > 0:
            plt.figure(figsize=(12, 8))
            plt.plot(poverty_data['year'], poverty_data['poverty_rate'], 
                    marker='s', linewidth=3, markersize=8, color='#A23B72', label='Poverty Rate')
            plt.title('South Africa Poverty Rate Over Time', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Poverty Rate (%)', fontsize=12)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.legend(fontsize=12)
            
            #trend line for poverty
            pov_years = poverty_data['year'].values
            pov_rates = poverty_data['poverty_rate'].values
            if len(pov_years) >= 3:
                z_pov = np.polyfit(pov_years, pov_rates, 1)
                p_pov = np.poly1d(z_pov)
                plt.plot(pov_years, p_pov(pov_years), "r--", alpha=0.8, linewidth=2, 
                        label=f'Trend (slope: {z_pov[0]:.3f}%/year)')
                plt.legend(fontsize=12)
            
            plt.tight_layout()
            poverty_ts_path = "../outputs/visualizations/poverty_timeseries.png"
            plt.savefig(poverty_ts_path, dpi=300, bbox_inches='tight')
            output_paths.append(poverty_ts_path)
            plt.show()
        
        # 3. Employment Histogram with Statistical Analysis
        plt.figure(figsize=(12, 8))
        n, bins, patches = plt.hist(self.data['employment_rate'], bins=15, alpha=0.7, 
                                   color='#4ECDC4', edgecolor='black', linewidth=1.2)
        plt.title('Distribution of Employment Rates (1995-2023)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Employment Rate (%)', fontsize=12)
        plt.ylabel('Frequency (Number of Years)', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add statistical lines
        mean_emp = self.data['employment_rate'].mean()
        median_emp = self.data['employment_rate'].median()
        plt.axvline(mean_emp, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_emp:.1f}%')
        plt.axvline(median_emp, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_emp:.1f}%')
        
        mu, sigma = stats.norm.fit(self.data['employment_rate'])
        x = np.linspace(self.data['employment_rate'].min(), self.data['employment_rate'].max(), 100)
        y = stats.norm.pdf(x, mu, sigma)
        
        y_scaled = y * len(self.data) * (bins[1] - bins[0])
        plt.plot(x, y_scaled, 'r-', linewidth=2, alpha=0.8, label='Normal Distribution Fit')
        
        plt.legend(fontsize=11)
        plt.tight_layout()
        employment_hist_path = "../outputs/visualizations/employment_histogram.png"
        plt.savefig(employment_hist_path, dpi=300, bbox_inches='tight')
        output_paths.append(employment_hist_path)
        plt.show()
        
        # 4. Poverty Histogram
        if len(poverty_data) > 0:
            plt.figure(figsize=(12, 8))
            plt.hist(poverty_data['poverty_rate'], bins=10, alpha=0.7, 
                    color='#F38BA8', edgecolor='black', linewidth=1.2)
            plt.title('Distribution of Poverty Rates', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Poverty Rate (%)', fontsize=12)
            plt.ylabel('Frequency (Number of Years)', fontsize=12)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add statistical lines
            mean_pov = poverty_data['poverty_rate'].mean()
            median_pov = poverty_data['poverty_rate'].median()
            plt.axvline(mean_pov, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_pov:.1f}%')
            plt.axvline(median_pov, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_pov:.1f}%')
            
            plt.legend(fontsize=11)
            plt.tight_layout()
            poverty_hist_path = "../outputs/visualizations/poverty_histogram.png"
            plt.savefig(poverty_hist_path, dpi=300, bbox_inches='tight')
            output_paths.append(poverty_hist_path)
            plt.show()
        
        # 5. Box Plot Analysis
        plt.figure(figsize=(14, 8))
        
        # Create decade groups for comparison
        self.data['decade'] = (self.data['year'] // 10) * 10
        data_for_box = []
        labels_for_box = []
        
        for decade in sorted(self.data['decade'].unique()):
            decade_data = self.data[self.data['decade'] == decade]['employment_rate']
            if len(decade_data) > 0:
                data_for_box.append(decade_data)
                labels_for_box.append(f"{decade}s")
        
        if data_for_box:
            box_plot = plt.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
            
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            plt.title('Employment Rate Distribution by Decade', fontsize=16, fontweight='bold', pad=20)
            plt.ylabel('Employment Rate (%)', fontsize=12)
            plt.xlabel('Decade', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            boxplot_path = "../outputs/visualizations/employment_boxplot.png"
            plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
            output_paths.append(boxplot_path)
            plt.show()
        
        # 6. Year-over-Year Change Analysis
        # Filter data to start from 2005
        change_data = self.data[self.data['year'] >= 2005].copy()
        employment_change = change_data['employment_rate'].diff()
        years = change_data['year'].values

        plt.figure(figsize=(16, 8))
        colors = ['red' if x < 0 else 'green' for x in employment_change[1:]]

        bars = plt.bar(years[1:], employment_change[1:], alpha=0.8, color=colors)
        plt.title('Year-over-Year Employment Rate Changes (2005 onwards)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Change in Employment Rate (%)', fontsize=12)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        plt.grid(True, alpha=0.3)

        
        for i, (year, change) in enumerate(zip(years[1:], employment_change[1:])):
            if abs(change) > 1.5:  # Only label significant changes
                plt.text(year, change + (0.1 if change > 0 else -0.1), f'{change:.1f}%',
                         ha='center', va='bottom' if change > 0 else 'top', fontsize=9)

        plt.xticks(years[1:], rotation=45, ha='right')  # Show all years on x-axis
        plt.tight_layout()
        change_path = "../outputs/visualizations/employment_changes.png"
        plt.savefig(change_path, dpi=300, bbox_inches='tight')
        output_paths.append(change_path)
        plt.show()
        
        # 7. Correlation Scatter Plot
        if len(self.complete_data) > 0:
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(self.complete_data['employment_rate'], 
                                 self.complete_data['poverty_rate'], 
                                 alpha=0.8, s=120, c=self.complete_data['year'], 
                                 cmap='viridis', edgecolors='black', linewidth=1)
            plt.colorbar(scatter, label='Year')
            
            
            if len(self.complete_data) >= 3:
                z = np.polyfit(self.complete_data['employment_rate'], 
                              self.complete_data['poverty_rate'], 1)
                p = np.poly1d(z)
                plt.plot(self.complete_data['employment_rate'], 
                        p(self.complete_data['employment_rate']), 
                        "r--", alpha=0.8, linewidth=3, 
                        label=f'Trend Line (slope: {z[0]:.2f})')
                
                # Calculate and display correlation
                correlation = self.complete_data['employment_rate'].corr(self.complete_data['poverty_rate'])
                plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=plt.gca().transAxes, fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.title('Employment vs Poverty Rate Relationship', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Employment Rate (%)', fontsize=12)
            plt.ylabel('Poverty Rate (%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11)
            
            plt.tight_layout()
            correlation_path = "../outputs/visualizations/employment_poverty_correlation.png"
            plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
            output_paths.append(correlation_path)
            plt.show()
        
        # 8. Summary Statistics Comparison
        plt.figure(figsize=(14, 8))
        
        metrics = ['Mean', 'Median', 'Min', 'Max', 'Std Dev']
        emp_values = [
            self.data['employment_rate'].mean(),
            self.data['employment_rate'].median(),
            self.data['employment_rate'].min(),
            self.data['employment_rate'].max(),
            self.data['employment_rate'].std()
        ]
        
        if len(poverty_data) > 0:
            pov_values = [
                poverty_data['poverty_rate'].mean(),
                poverty_data['poverty_rate'].median(),
                poverty_data['poverty_rate'].min(),
                poverty_data['poverty_rate'].max(),
                poverty_data['poverty_rate'].std()
            ]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            plt.bar(x - width/2, emp_values, width, label='Employment Rate', 
                   alpha=0.8, color='#4ECDC4')
            plt.bar(x + width/2, pov_values, width, label='Poverty Rate', 
                   alpha=0.8, color='#F38BA8')
            
            
            for i, (emp_val, pov_val) in enumerate(zip(emp_values, pov_values)):
                plt.text(i - width/2, emp_val + max(emp_values)*0.01, f'{emp_val:.1f}%', 
                        ha='center', va='bottom', fontsize=10)
                plt.text(i + width/2, pov_val + max(pov_values)*0.01, f'{pov_val:.1f}%', 
                        ha='center', va='bottom', fontsize=10)
        else:
            plt.bar(metrics, emp_values, alpha=0.8, color='#4ECDC4')
            for i, val in enumerate(emp_values):
                plt.text(i, val + max(emp_values)*0.01, f'{val:.1f}%', 
                        ha='center', va='bottom', fontsize=10)
        
        plt.title('Statistical Summary Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Rate (%)', fontsize=12)
        plt.xlabel('Statistics', fontsize=12)
        plt.xticks(range(len(metrics)), metrics)
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        summary_stats_path = "../outputs/visualizations/summary_statistics.png"
        plt.savefig(summary_stats_path, dpi=300, bbox_inches='tight')
        output_paths.append(summary_stats_path)
        plt.show()
        
        # 9. Combined Overview Chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        
        ax1.plot(self.data['year'], self.data['employment_rate'], 
                marker='o', linewidth=2, markersize=6, color='#2E86AB')
        ax1.set_title('Employment Rate Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Employment Rate (%)')
        ax1.grid(True, alpha=0.3)
        
        
        if len(poverty_data) > 0:
            ax2.plot(poverty_data['year'], poverty_data['poverty_rate'], 
                    marker='s', linewidth=2, markersize=6, color='#A23B72')
            ax2.set_title('Poverty Rate Over Time', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Poverty Rate (%)')
            ax2.grid(True, alpha=0.3)
        
        
        ax3.hist(self.data['employment_rate'], bins=12, alpha=0.7, 
                color='#4ECDC4', edgecolor='black')
        ax3.set_title('Employment Rate Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Employment Rate (%)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3, axis='y')
        
        
        employment_change = self.data['employment_rate'].diff()
        colors = ['red' if x < 0 else 'green' for x in employment_change[1:]]
        ax4.bar(self.data['year'][1:], employment_change[1:], alpha=0.8, color=colors)
        ax4.set_title('Year-over-Year Changes', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Change (%)')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        overview_path = "../outputs/visualizations/south_africa_overview.png"
        plt.savefig(overview_path, dpi=300, bbox_inches='tight')
        output_paths.append(overview_path)
        plt.show()
        
        # 10. Side-by-Side Bar Chart for Employment vs Poverty by Year
        if len(self.complete_data) > 0:
            plt.figure(figsize=(16, 10))
            
            # Get data for years where both employment and poverty data exist which is from 20005
            years = self.complete_data['year'].values
            employment_rates = self.complete_data['employment_rate'].values
            poverty_rates = self.complete_data['poverty_rate'].values
            
           
            bar_width = 0.35
            x_pos = np.arange(len(years))
            
            
            bars1 = plt.bar(x_pos - bar_width/2, employment_rates, bar_width, 
                           label='Employment Rate', color='#4ECDC4', alpha=0.8, 
                           edgecolor='black', linewidth=1)
            bars2 = plt.bar(x_pos + bar_width/2, poverty_rates, bar_width, 
                           label='Poverty Rate', color='#F38BA8', alpha=0.8, 
                           edgecolor='black', linewidth=1)
            
            
            plt.title('Employment vs Poverty Rates by Year - Side-by-Side Comparison', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Rate (%)', fontsize=12)
            plt.xticks(x_pos, years, rotation=45, ha='right')
            plt.legend(fontsize=12, loc='upper right')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars (for better readability)
            def add_value_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
            
            
            max_rate = max(max(employment_rates), max(poverty_rates))
            plt.ylim(0, max_rate * 1.1)
            
            plt.tight_layout()
            sideby_path = "../outputs/visualizations/employment_poverty_sidebyside.png"
            plt.savefig(sideby_path, dpi=300, bbox_inches='tight')
            output_paths.append(sideby_path)
            plt.show()
        
        # 11. Alternative version with all employment years and available poverty data
        plt.figure(figsize=(18, 10))
        
        # Get all employment data
        all_years = self.data['year'].values
        all_employment = self.data['employment_rate'].values
        
        # Create poverty array with NaN for missing years
        all_poverty = np.full(len(all_years), np.nan)
        poverty_data = self.data.dropna(subset=['poverty_rate'])
        for i, year in enumerate(all_years):
            poverty_row = poverty_data[poverty_data['year'] == year]
            if not poverty_row.empty:
                all_poverty[i] = poverty_row['poverty_rate'].iloc[0]
        
        
        bar_width = 0.35
        x_pos = np.arange(len(all_years))
        
        # Create bars for employment
        bars1 = plt.bar(x_pos - bar_width/2, all_employment, bar_width, 
                       label='Employment Rate', color='#4ECDC4', alpha=0.8, 
                       edgecolor='black', linewidth=1)
        
        # Create bars for poverty
        poverty_mask = ~np.isnan(all_poverty)
        bars2 = plt.bar(x_pos[poverty_mask] + bar_width/2, all_poverty[poverty_mask], bar_width, 
                       label='Poverty Rate', color='#F38BA8', alpha=0.8, 
                       edgecolor='black', linewidth=1)
        
        
        plt.title('Employment vs Poverty Rates by Year - Complete Time Series', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Rate (%)', fontsize=12)
        plt.xticks(x_pos[::2], all_years[::2], rotation=45, ha='right')  # Show every 2nd year to avoid crowding
        plt.legend(fontsize=12, loc='upper right')
        plt.grid(True, alpha=0.3, axis='y')
        
       
        plt.figtext(0.02, 0.02, 'Note: Poverty rate data not available for all years', 
                   fontsize=10, style='italic', alpha=0.7)
        
        
        max_rate = max(max(all_employment), np.nanmax(all_poverty))
        plt.ylim(0, max_rate * 1.1)
        
        plt.tight_layout()
        complete_sideby_path = "../outputs/visualizations/employment_poverty_complete_sidebyside.png"
        plt.savefig(complete_sideby_path, dpi=300, bbox_inches='tight')
        output_paths.append(complete_sideby_path)
        plt.show()

        # 12. Combined Employment & Poverty Line Chart (2005 onwards)
        combined_data = self.complete_data[self.complete_data['year'] >= 2005]

        if len(combined_data) > 0:
            plt.figure(figsize=(14, 8))
            plt.plot(combined_data['year'], combined_data['employment_rate'],
                     marker='o', linewidth=3, markersize=8, color='#2E86AB', label='Employment Rate')
            plt.plot(combined_data['year'], combined_data['poverty_rate'],
                     marker='s', linewidth=3, markersize=8, color='#A23B72', label='Poverty Rate')

            plt.title('Employment vs Poverty Rates (2005 onwards)', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Rate (%)', fontsize=12)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.legend(fontsize=12)

            
            plt.axvline(x=2008, color='red', linestyle=':', alpha=0.7, label='Global Financial Crisis')
            plt.axvline(x=2020, color='purple', linestyle=':', alpha=0.7, label='COVID-19 Pandemic')

            plt.tight_layout()
            combined_line_path = "../outputs/visualizations/employment_poverty_linechart_2005.png"
            plt.savefig(combined_line_path, dpi=300, bbox_inches='tight')
            output_paths.append(combined_line_path)
            plt.show()
        
        print(f"\n=== All visualizations saved successfully! ===")
        for i, path in enumerate(output_paths, 1):
            print(f"{i}. {path.split('/')[-1]}")
        
        return output_paths
    
    def trend_analysis(self):
        """Analyze trends and patterns in the data."""
        print("\n=== TREND ANALYSIS ===")
        
        # Employment trend analysis
        years = self.data['year'].values
        employment = self.data['employment_rate'].values
        
        # Linear trend
        emp_trend = np.polyfit(years, employment, 1)
        emp_trend_func = np.poly1d(emp_trend)
        
        print(f"Employment Trend Analysis:")
        print(f"  Linear trend: {emp_trend[0]:.3f}% per year")
        
        if emp_trend[0] > 0:
            print(f"  ↗ Employment rate has been increasing over time")
        else:
            print(f"  ↘ Employment rate has been declining over time")
        
        # Calculate R-squared for trend fit
        predicted = emp_trend_func(years)
        ss_res = np.sum((employment - predicted) ** 2)
        ss_tot = np.sum((employment - np.mean(employment)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"  Trend fit (R²): {r_squared:.3f}")
        
        # Identify periods of significant change
        employment_change = np.diff(employment)
        significant_changes = np.where(np.abs(employment_change) > 2 * np.std(employment_change))[0]
        
        print(f"\nPeriods of Significant Change:")
        for idx in significant_changes:
            year_from = years[idx]
            year_to = years[idx + 1]
            change = employment_change[idx]
            direction = "increased" if change > 0 else "decreased"
            print(f"  {year_from}-{year_to}: Employment rate {direction} by {abs(change):.1f}%")
        
        # Volatility analysis
        volatility = np.std(employment_change)
        print(f"\nVolatility Analysis:")
        print(f"  Average year-over-year change: {np.mean(employment_change):.2f}%")
        print(f"  Standard deviation of changes: {volatility:.2f}%")
        
        return {
            'trend_slope': emp_trend[0],
            'r_squared': r_squared,
            'volatility': volatility,
            'significant_changes': len(significant_changes)
        }
    
    def generate_insights(self):
        """Generate key insights from the analysis."""
        print("\n=== KEY INSIGHTS AND FINDINGS ===")
        
        # Data coverage insights
        print("1. DATA COVERAGE:")
        print(f"   • Employment data: {self.data['year'].min()}-{self.data['year'].max()} ({len(self.data)} years)")
        print(f"   • Poverty data: Available for {len(self.complete_data)} years")
        
        # Employment insights
        current_emp = self.data['employment_rate'].iloc[-1]
        max_emp = self.data['employment_rate'].max()
        min_emp = self.data['employment_rate'].min()
        max_emp_year = self.data.loc[self.data['employment_rate'].idxmax(), 'year']
        min_emp_year = self.data.loc[self.data['employment_rate'].idxmin(), 'year']
        
        print(f"\n2. EMPLOYMENT INSIGHTS:")
        print(f"   • Current employment rate (2023): {current_emp:.1f}%")
        print(f"   • Highest employment rate: {max_emp:.1f}% in {max_emp_year}")
        print(f"   • Lowest employment rate: {min_emp:.1f}% in {min_emp_year}")
        
        # Poverty insights
        if len(self.complete_data) > 0:
            latest_poverty = self.complete_data['poverty_rate'].iloc[-1]
            print(f"\n3. POVERTY INSIGHTS:")
            print(f"   • Latest poverty rate: {latest_poverty:.1f}%")
            print(f"   • Over half of South Africans live below the national poverty line")
        
        # Correlation insights
        if len(self.complete_data) >= 3:
            correlation = self.complete_data['employment_rate'].corr(self.complete_data['poverty_rate'])
            print(f"\n4. EMPLOYMENT-POVERTY RELATIONSHIP:")
            print(f"   • Correlation coefficient: {correlation:.3f}")
            if correlation < -0.5:
                print(f"   • Strong negative relationship: Higher employment tends to coincide with lower poverty")
            elif correlation > 0.5:
                print(f"   • Strong positive relationship: Higher employment coincides with higher poverty")
            else:
                print(f"   • Weak relationship between employment and poverty rates")
        
        # Economic events context
        print(f"\n5. HISTORICAL CONTEXT:")
        print(f"   • 1997 dip: Likely related to Asian Financial Crisis impact")
        print(f"   • 2008-2009 decline: Global Financial Crisis impact visible")
        print(f"   • 2020 onward: COVID-19 pandemic impact on employment")
        
        print(f"\n6. POLICY IMPLICATIONS:")
        print(f"   • Employment creation is crucial for poverty reduction")
        print(f"   • Economic shocks have lasting effects on employment")
        print(f"   • Structural unemployment remains a significant challenge")
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("=== STARTING COMPREHENSIVE ANALYSIS ===")
        
       
        if not self.load_data():
            return False
        
        # numerical analysis
        numerical_results = self.numerical_analysis_with_numpy()
        
        # Correlation analysis
        correlation_results = self.correlation_analysis()
        
        # Create visualizations
        viz_paths = self.create_visualizations()
        
        # Trend analysis
        trend_results = self.trend_analysis()
        
        # Generate insights
        self.generate_insights()
        
        print("\n=== ANALYSIS COMPLETED SUCCESSFULLY ===")
        
        return {
            'numerical': numerical_results,
            'correlation': correlation_results,
            'trends': trend_results,
            'visualization_paths': viz_paths
        }

def main():
    """Main function to run the complete analysis."""
    analyzer = SouthAfricaAnalyzer()
    results = analyzer.run_complete_analysis()
    
    if results:
        print("\nAnalysis completed successfully!")
        print("check outputs/visualizations folder for generated charts.")
    else:
        print("Analysis failed!")

if __name__ == "__main__":
    main()
