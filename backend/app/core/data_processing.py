"""
Data processing utilities for DataWhiz
"""
import pandas as pd
import numpy as np
import os

def load_data_file(filepath):
    """Load data file based on file extension"""
    try:
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            return pd.read_excel(filepath)
        elif filepath.endswith('.json'):
            return pd.read_json(filepath)
        else:
            return None
    except Exception as e:
        print(f"Error loading file {filepath}: {str(e)}")
        return None

def merge_dataframes(dataframes):
    """Simple side-by-side merge - preserve all columns even with same names"""
    if not dataframes:
        return pd.DataFrame()
    
    if len(dataframes) == 1:
        return dataframes[0]
    
    # Simple side-by-side merge - preserve all columns
    try:
        # Start with first file
        result_df = dataframes[0].copy()
        
        # Add all columns from other files, renaming duplicates
        for i in range(1, len(dataframes)):
            df = dataframes[i]
            for col in df.columns:
                # If column already exists, rename it
                if col in result_df.columns:
                    new_col_name = f"{col}_file{i+1}"
                    result_df[new_col_name] = df[col]
                else:
                    result_df[col] = df[col]
        
        return result_df
    except Exception as e:
        print(f"Warning: Could not merge dataframes: {str(e)}")
        return dataframes[0]

def analyze_uploaded_file(filepath):
    """Analyze uploaded file and return basic info"""
    try:
        # Determine file type and load accordingly
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            raise ValueError('Unsupported file format')
        
        # Basic data analysis
        info = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'sample_data': df.head(5).to_dict('records'),
            'summary_stats': df.describe().to_dict()
        }
        
        return info
        
    except Exception as e:
        raise Exception(f"Error analyzing file: {str(e)}")
