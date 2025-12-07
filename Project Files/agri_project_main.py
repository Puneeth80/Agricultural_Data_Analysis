import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from abc import ABC, abstractmethod

# Definition for Data Loader Interface
# Allows us to easily swap CSV for SQL or JSON later without breaking the app
class IDataLoader(ABC):
    @abstractmethod
    def load_data(self, filepath: str) -> pd.DataFrame:
        pass

class CSVLoader(IDataLoader):
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Loads a CSV file and verifies it exists."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Error: The file '{filepath}' was not found.")
        
        print(f"Successfully loaded data from: {filepath}")
        return pd.read_csv(filepath)

class DataProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def clean_data(self):
        """Handle missing values using mean imputation to preserve data density."""
        if 'rainfall_mm' in self.df.columns and self.df['rainfall_mm'].isnull().any():
            print("Processing: Filling missing rainfall data with mean values...")
            self.df['rainfall_mm'] = self.df['rainfall_mm'].fillna(self.df['rainfall_mm'].mean())
        return self.df

    def feature_engineering(self):
        """Create new metrics to normalize yield against farm size."""
        if 'crop_yield' in self.df.columns and 'farm_size_hectares' in self.df.columns:
            print("Processing: Calculating Yield per Hectare (Efficiency Metric)...")
            self.df['Yield_per_Hectare'] = self.df['crop_yield'] / self.df['farm_size_hectares']
        return self.df

class DataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def print_summary(self):
        """Output key statistical correlations to the console."""
        print("\n--- Statistical Summary ---")
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        if 'rainfall_mm' in corr and 'crop_yield' in corr:
            print(f"Correlation (Rain vs Yield): {corr.loc['rainfall_mm', 'crop_yield']:.4f}")
        
        if 'farm_size_hectares' in corr and 'crop_yield' in corr:
            print(f"Correlation (Size vs Yield): {corr.loc['farm_size_hectares', 'crop_yield']:.4f}")

class DataVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        sns.set_theme(style="whitegrid")
        
        # Ensure output directory exists to keep the project clean
        self.output_dir = "Results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_heatmap(self, filename="correlation_heatmap.png"):
        """Generates a readable heatmap with no cropped labels."""
        # 1. Make the figure larger (Width=12, Height=10)
        plt.figure(figsize=(12, 10))
        
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        # 2. Draw the heatmap
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", square=True)
        
        plt.title('Correlation Matrix', fontsize=16)
        
        # 3. Rotate the bottom labels 45 degrees so they fit
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0) # Keep side labels straight
        
        # 4. The Magic Fix: Automatically adjust margins to fit everything
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved visualization: {save_path}")

    # def save_pairplot(self, filename="pairplot_analysis.png"):
    #     """Generates a pairplot for a deep-dive into feature distributions."""
    #     print("Generating pairplot (this may take a moment)...")
    #     hue_col = 'soil_quality_index' if 'soil_quality_index' in self.df.columns else None
        
    #     plot = sns.pairplot(self.df, hue=hue_col, palette='viridis', diag_kind='kde')
        
    #     save_path = os.path.join(self.output_dir, filename)
    #     plot.savefig(save_path)
    #     plt.close()
    #     print(f"Saved visualization: {save_path}")
    def save_key_insight(self, filename="key_insight_size_vs_yield.png"):
        """Plots ONLY the most important relationship we found."""
        plt.figure(figsize=(10, 6))
        
        # We only plot the "Thief" (Farm Size) vs the Outcome (Yield)
        sns.regplot(x='farm_size_hectares', y='crop_yield', data=self.df, 
                    color='darkred', line_kws={'color': 'red'})
        
        plt.title('The Key Driver: Farm Size vs. Crop Yield', fontsize=16)
        plt.xlabel('Farm Size (Hectares)')
        plt.ylabel('Total Yield')
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"   🎯 Saved Key Insight graph to result folder")

    def save_impact_analysis(self, filename="rainfall_impact.png"):
        """Compares raw yield vs efficiency side-by-side."""
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Absolute Yield
        plt.subplot(1, 2, 1)
        sns.regplot(x='rainfall_mm', y='crop_yield', data=self.df, color='green', scatter_kws={'alpha':0.5})
        plt.title('Rainfall vs. Total Yield')

        # Subplot 2: Efficiency (Yield/Hectare)
        plt.subplot(1, 2, 2)
        if 'Yield_per_Hectare' in self.df.columns:
            sns.regplot(x='rainfall_mm', y='Yield_per_Hectare', data=self.df, color='blue', scatter_kws={'alpha':0.5})
            plt.title('Rainfall vs. Efficiency')
            
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved visualization: {save_path}")

class AgriAnalyticsApp:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = CSVLoader()
    
    def execute(self):
        print("Starting Agricultural Data Analysis...")
        
        # 1. Load Data
        try:
            raw_data = self.loader.load_data(self.file_path)
        except Exception as e:
            print(e)
            return

        # 2. Process Data
        processor = DataProcessor(raw_data)
        processor.clean_data()
        clean_df = processor.feature_engineering()

        # 3. Analyze Data
        analyzer = DataAnalyzer(clean_df)
        analyzer.print_summary()

        # 4. Visualize Results
        print("\nGenerating visualizations...")
        visualizer = DataVisualizer(clean_df)
        visualizer.save_heatmap()
        visualizer.save_key_insight()
        visualizer.save_impact_analysis()

        print(f"\nAnalysis complete. Results saved in '{visualizer.output_dir}' folder.")

# Entry point for the script
def main():
    target_file = "crop_yield_data.csv"
    
    # Logic to locate the file in common directories (Downloads, local, etc.)
    user_home = os.path.expanduser("~")
    search_paths = [
        os.path.join(user_home, "Downloads/Crops_EDA_Project/Agricultural Data Analysis (Python) /Data Files", target_file),
        target_file,
        os.path.join("Data Files", target_file),
        os.path.join("..", "Data Files", target_file)
    ]
    
    valid_path = None
    print("Locating dataset...")
    for path in search_paths:
        if os.path.exists(path):
            valid_path = path
            print(f"Dataset found at: {valid_path}")
            break
    
    if valid_path:
        app = AgriAnalyticsApp(valid_path)
        app.execute()
    else:
        print("\nError: Could not find 'crop_yield_data.csv'. Please check the file path.")

if __name__ == "__main__":
    main()