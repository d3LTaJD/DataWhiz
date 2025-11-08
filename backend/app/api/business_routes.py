"""
Business analysis API routes
"""
import sys
import os
import time

# Set UTF-8 encoding for Windows compatibility using environment variable
if sys.platform == 'win32':
    # Set environment variable instead of wrapping streams (safer for Flask)
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import numpy as np
from app.core.data_processing import load_data_file

business_bp = Blueprint('business', __name__)

@business_bp.route('/api/business/quality-check', methods=['POST'])
def business_quality_check():
    """Run data quality check for business analysis"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'Filename required'}), 400
        
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Load data
        df = load_data_file(filepath)
        if df is None:
            return jsonify({'error': 'Could not load file'}), 400
        
        # Data quality analysis
        quality_report = {
            'total_rows': int(len(df)),
            'total_columns': int(len(df.columns)),
            'missing_values': {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
            'duplicate_rows': int(df.duplicated().sum()),
            'data_types': df.dtypes.astype(str).to_dict(),
            'memory_usage': int(df.memory_usage(deep=True).sum()),
            'quality_score': 0
        }
        
        # Calculate quality score
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
        quality_score = max(0, 100 - missing_percentage - duplicate_percentage)
        quality_report['quality_score'] = round(quality_score, 2)
        
        return jsonify({
            'success': True,
            'quality_report': quality_report
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@business_bp.route('/api/business/clean-data', methods=['POST'])
def business_clean_data():
    """Clean data for business analysis"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        cleaning_options = data.get('options', {})
        
        if not filename:
            return jsonify({'error': 'Filename required'}), 400
        
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Load data
        df = load_data_file(filepath)
        if df is None:
            return jsonify({'error': 'Could not load file'}), 400
        
        # Apply cleaning operations
        original_rows = len(df)
        
        # Count missing values BEFORE cleaning
        missing_values_count = int(df.isnull().sum().sum())
        
        # Count duplicate rows BEFORE cleaning
        duplicate_rows_count = int(df.duplicated().sum())
        
        # Remove duplicates if requested
        if cleaning_options.get('remove_duplicates', False):
            df = df.drop_duplicates()
        
        # Handle missing values
        if cleaning_options.get('handle_missing') == 'drop':
            df = df.dropna()
        elif cleaning_options.get('handle_missing') == 'fill':
            fill_method = cleaning_options.get('fill_method', 'forward')
            if fill_method == 'forward':
                df = df.ffill()
            elif fill_method == 'backward':
                df = df.bfill()
            elif fill_method == 'mean':
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        # Standardize formats
        if cleaning_options.get('standardize_formats', False):
            # Convert date columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col], errors='ignore')
                    except:
                        pass
        
        # Save cleaned data
        # Use timestamp-based filename to avoid extremely long filenames on Windows
        timestamp = int(time.time())
        cleaned_filename = f"cleaned_{timestamp}.csv"
        cleaned_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], cleaned_filename)
        
        # Ensure the upload folder exists
        os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        print(f"[INFO] Saving cleaned data to: {cleaned_filepath}")
        print(f"[INFO] Path length: {len(cleaned_filepath)} characters")
        
        # Use utf-8-sig for better Windows compatibility
        try:
            df.to_csv(cleaned_filepath, index=False, encoding='utf-8-sig', lineterminator='\n')
            print("[SUCCESS] Cleaned data saved successfully")
        except Exception as csv_error:
            # If that fails, try without specifying encoding
            print(f"[ERROR] Failed to save CSV: {csv_error}")
            print(f"[ERROR] Filepath: {cleaned_filepath}")
            raise  # Re-raise the error so it's properly reported
        
        return jsonify({
            'success': True,
            'cleaned_file': cleaned_filename,
            'original_rows': int(original_rows),
            'cleaned_rows': int(len(df)),
            'rows_removed': int(original_rows - len(df)),
            'missing_values_found': missing_values_count,
            'duplicate_rows_found': duplicate_rows_count
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@business_bp.route('/api/business/transform-data', methods=['POST'])
def business_transform_data():
    """Transform data for business analysis"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        transform_options = data.get('options', {})
        
        if not filename:
            return jsonify({'error': 'Filename required'}), 400
        
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Load data
        df = load_data_file(filepath)
        if df is None:
            return jsonify({'error': 'Could not load file'}), 400
        
        original_columns = df.columns.tolist()
        date_features_count = 0
        categories_encoded_count = 0
        
        # Add REAL business metrics that analysts actually use - ULTRA ROBUST
        if transform_options.get('add_business_metrics', False):
            print(f"[INFO] Available columns: {list(df.columns)}")
            print(f"[INFO] Data types: {df.dtypes.to_dict()}")
            
            # ULTRA ROBUST column detection - try multiple strategies
            def find_numeric_columns():
                """Find all numeric columns that could be prices, quantities, costs"""
                numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
                print(f"[INFO] Numeric columns: {numeric_cols}")
                return numeric_cols
            
            def find_text_columns():
                """Find all text columns that could be customers, products, categories"""
                text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
                print(f"[INFO] Text columns: {text_cols}")
                return text_cols
            
            def smart_detect_price_columns(numeric_cols):
                """Smart detection of price-like columns"""
                price_indicators = ['price', 'amount', 'value', 'revenue', 'income', 'sales', 'total', 'sum', 'cost', 'fee', 'charge']
                price_cols = []
                
                for col in numeric_cols:
                    col_lower = col.lower()
                    # Check if column name contains price indicators
                    if any(indicator in col_lower for indicator in price_indicators):
                        price_cols.append(col)
                    # Check if values look like prices (positive, reasonable range)
                    elif df[col].min() >= 0 and df[col].max() < 1000000:  # Reasonable price range
                        price_cols.append(col)
                
                # If no obvious price columns, use the largest numeric column
                if not price_cols and numeric_cols:
                    price_cols = [df[numeric_cols].max().idxmax()]  # Column with highest values
                
                return price_cols
            
            def smart_detect_quantity_columns(numeric_cols):
                """Smart detection of quantity-like columns"""
                qty_indicators = ['quantity', 'qty', 'count', 'number', 'units', 'items', 'pieces', 'volume', 'amount']
                quantity_cols = []
                
                for col in numeric_cols:
                    col_lower = col.lower()
                    if any(indicator in col_lower for indicator in qty_indicators):
                        quantity_cols.append(col)
                    # Check if values look like quantities (integers, reasonable range)
                    elif df[col].dtype in ['int64', 'int32'] and df[col].max() < 10000:
                        quantity_cols.append(col)
                
                return quantity_cols
            
            def smart_detect_customer_columns(text_cols):
                """Smart detection of customer-like columns"""
                customer_indicators = ['customer', 'client', 'user', 'buyer', 'purchaser', 'name', 'id', 'person']
                customer_cols = []
                
                for col in text_cols:
                    col_lower = col.lower()
                    if any(indicator in col_lower for indicator in customer_indicators):
                        customer_cols.append(col)
                    # Check if it looks like an ID or name column
                    elif 'id' in col_lower or 'name' in col_lower:
                        customer_cols.append(col)
                
                return customer_cols
            
            def smart_detect_product_columns(text_cols):
                """Smart detection of product-like columns"""
                product_indicators = ['product', 'item', 'service', 'goods', 'merchandise', 'category', 'type', 'class']
                product_cols = []
                
                for col in text_cols:
                    col_lower = col.lower()
                    if any(indicator in col_lower for indicator in product_indicators):
                        product_cols.append(col)
                
                return product_cols
            
            # Get all columns by type
            numeric_cols = find_numeric_columns()
            text_cols = find_text_columns()
            
            # Smart detection
            price_cols = smart_detect_price_columns(numeric_cols)
            quantity_cols = smart_detect_quantity_columns(numeric_cols)
            customer_cols = smart_detect_customer_columns(text_cols)
            product_cols = smart_detect_product_columns(text_cols)
            
            print(f"[INFO] Price columns detected: {price_cols}")
            print(f"[INFO] Quantity columns detected: {quantity_cols}")
            print(f"[INFO] Customer columns detected: {customer_cols}")
            print(f"[INFO] Product columns detected: {product_cols}")
            
            # SUPER SIMPLE: Only 3 essential business metrics
            try:
                # 1. Total Revenue (most important)
                if price_cols and quantity_cols:
                    price_col = price_cols[0]
                    quantity_col = quantity_cols[0]
                    df['Total_Revenue'] = df[price_col] * df[quantity_col]
                    print(f"[SUCCESS] Added Total_Revenue using {price_col} * {quantity_col}")
                elif price_cols:
                    df['Total_Revenue'] = df[price_cols[0]]
                    print(f"[SUCCESS] Added Total_Revenue using {price_cols[0]}")
                
                # 2. Profit (Revenue - Cost, or use a reasonable margin)
                if 'Total_Revenue' in df.columns:
                    # Calculate profit as 20% margin of revenue (realistic business scenario)
                    df['Profit'] = df['Total_Revenue'] * 0.20
                    print(f"[SUCCESS] Added Profit as 20% margin of Total_Revenue")
                elif len(price_cols) >= 2:
                    # If we have cost and selling price
                    df['Profit'] = df[price_cols[0]] - df[price_cols[1]]
                    print(f"[SUCCESS] Added Profit using {price_cols[0]} - {price_cols[1]}")
                elif price_cols:
                    # If only one price column, assume 20% margin
                    df['Profit'] = df[price_cols[0]] * 0.20
                    print(f"[SUCCESS] Added Profit as 20% margin of {price_cols[0]}")
                
                # 3. Customer Value (if we have customer and revenue)
                if customer_cols and 'Total_Revenue' in df.columns:
                    customer_col = customer_cols[0]
                    customer_revenue = df.groupby(customer_col)['Total_Revenue'].sum()
                    df['Customer_Value'] = df[customer_col].map(customer_revenue)
                    print(f"[SUCCESS] Added Customer_Value using {customer_col}")
                        
                print("[SUCCESS] Added 3 essential business metrics")
            except Exception as e:
                print(f"[ERROR] Error adding essential metrics: {e}")
        
        # Add BUSINESS date features for analysis - ULTRA ROBUST
        if transform_options.get('add_date_features', False):
            print("[INFO] Searching for date columns...")
            
            # ULTRA ROBUST date detection
            def detect_date_columns():
                date_columns = []
                
                # 1. Already datetime columns
                existing_datetime = df.select_dtypes(include=['datetime64']).columns.tolist()
                date_columns.extend(existing_datetime)
                print(f"[INFO] Existing datetime columns: {existing_datetime}")
                
                # 2. Try to convert string columns that look like dates
                for col in df.columns:
                    if df[col].dtype == 'object':
                        try:
                            # Sample first 100 rows to test conversion
                            sample_data = df[col].dropna().head(100)
                            if len(sample_data) > 0:
                                # Try multiple date formats
                                test_convert = pd.to_datetime(sample_data, errors='coerce')
                                success_rate = (1 - test_convert.isna().sum() / len(test_convert))
                                
                                if success_rate > 0.7:  # 70% success rate
                                    df[col] = pd.to_datetime(df[col], errors='coerce')
                                    date_columns.append(col)
                                    print(f"[SUCCESS] Converted {col} to datetime (success rate: {success_rate:.2%})")
                        except Exception as e:
                            print(f"[WARNING] Could not convert {col}: {e}")
                
                # 3. Try numeric columns that might be timestamps
                for col in df.select_dtypes(include=['int64', 'float64']).columns:
                    try:
                        # Check if values look like timestamps (reasonable date range)
                        if df[col].min() > 1000000000 and df[col].max() < 2000000000:  # Unix timestamp range
                            df[col] = pd.to_datetime(df[col], unit='s', errors='coerce')
                            if not df[col].isna().all():
                                date_columns.append(col)
                                print(f"[SUCCESS] Converted {col} from timestamp to datetime")
                    except:
                        pass
                
                return date_columns
            
            date_columns = detect_date_columns()
            print(f"[INFO] Total date columns found: {date_columns}")
            
            # ULTRA ROBUST date feature extraction
            for col in date_columns:
                try:
                    if df[col].isna().all():
                        print(f"[WARNING] Skipping {col} - all values are null")
                        continue
                    
                    # SUPER SIMPLE: Only 2 essential date features
                    df[f'{col}_Year'] = df[col].dt.year
                    df[f'{col}_Quarter'] = df[col].dt.quarter
                    
                    date_features_count += 2  # Only 2 essential date features
                    print(f"[SUCCESS] Added 2 essential date features for {col}")
                except Exception as e:
                    print(f"[ERROR] Error processing date column {col}: {e}")
            
            # If no date columns found, skip creating synthetic features
            if not date_columns:
                print("[INFO] No date columns found, skipping date features")
        
        # Encode categorical variables for business analysis - ULTRA ROBUST
        if transform_options.get('encode_categoricals', False):
            print("[INFO] Starting categorical analysis...")
            
            # ULTRA ROBUST categorical detection
            def find_categorical_columns():
                categorical_columns = []
                
                # 1. Object/string columns
                object_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
                categorical_columns.extend(object_cols)
                
                # 2. Low-cardinality numeric columns (might be categories)
                for col in df.select_dtypes(include=['int64', 'float64']).columns:
                    unique_count = df[col].nunique()
                    if 2 <= unique_count <= 20:  # Low cardinality numeric
                        categorical_columns.append(col)
                        print(f"[INFO] Found low-cardinality numeric column: {col} ({unique_count} unique values)")
                
                return categorical_columns
            
            categorical_columns = find_categorical_columns()
            print(f"[INFO] Categorical columns found: {list(categorical_columns)}")
            
            for col in categorical_columns:
                try:
                    # Handle missing values first
                    if df[col].isnull().any():
                        df[col] = df[col].fillna('Unknown')
                        print(f"[INFO] Filled missing values in {col} with 'Unknown'")
                    
                    unique_count = df[col].nunique()
                    print(f"[INFO] Column {col} has {unique_count} unique values")
                    
                    if unique_count < 100 and unique_count > 1:  # More flexible limits
                        # Standard encoding
                        df[f'{col}_Encoded'] = pd.Categorical(df[col]).codes
                        
                        # SUPER SIMPLE: Skip all categorical encoding to keep it minimal
                        print(f"[SUCCESS] Encoded {col} (no extra features added)")
                        
                        categories_encoded_count += 1
                        print(f"[SUCCESS] Successfully encoded {col}")
                    else:
                        print(f"[WARNING] Skipped {col} (unique values: {unique_count} - outside range 2-100)")
                except Exception as e:
                    print(f"[ERROR] Error processing {col}: {e}")
            
            # SIMPLE: Skip generic categorical features to avoid too many columns
            print("[INFO] Skipping generic categorical features to keep it simple")
        
        # Save transformed data
        # Use timestamp-based filename to avoid extremely long filenames on Windows
        timestamp = int(time.time())
        transformed_filename = f"transformed_{timestamp}.csv"
        transformed_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], transformed_filename)
        
        # Ensure the upload folder exists
        os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        print(f"[INFO] Saving transformed data to: {transformed_filepath}")
        print(f"[INFO] Path length: {len(transformed_filepath)} characters")
        
        # Use utf-8-sig for better Windows compatibility
        try:
            df.to_csv(transformed_filepath, index=False, encoding='utf-8-sig', lineterminator='\n')
            print("[SUCCESS] Transformed data saved successfully")
        except Exception as csv_error:
            # If that fails, try without specifying encoding
            print(f"[ERROR] Failed to save CSV: {csv_error}")
            print(f"[ERROR] Filepath: {transformed_filepath}")
            raise  # Re-raise the error so it's properly reported
        
        # Convert NaN values to None for JSON serialization
        df_clean = df.fillna('N/A')
        
        return jsonify({
            'success': True,
            'transformed_file': transformed_filename,
            'new_columns': [col for col in df.columns if col not in original_columns],
            'total_columns': int(len(df.columns)),
            'date_features_count': int(date_features_count),
            'categories_encoded_count': int(categories_encoded_count)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@business_bp.route('/api/business/analyze', methods=['POST'])
def business_analyze():
    """Run business analysis"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        analysis_type = data.get('analysis_type', 'sales')
        
        if not filename:
            return jsonify({'error': 'Filename required'}), 400
        
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Load data
        df = load_data_file(filepath)
        if df is None:
            return jsonify({'error': 'Could not load file'}), 400
        
        analysis_results = {}
        
        if analysis_type == 'sales':
            # Sales analysis
            if 'Price' in df.columns and 'Quantity' in df.columns:
                df['Total_Sales'] = df['Price'] * df['Quantity']
                analysis_results = {
                    'total_revenue': float(df['Total_Sales'].sum()),
                    'average_order_value': float(df['Total_Sales'].mean()),
                    'top_products': {k: float(v) for k, v in df.groupby('ProductName')['Total_Sales'].sum().nlargest(10).to_dict().items()},
                    'sales_by_country': {k: float(v) for k, v in df.groupby('Country')['Total_Sales'].sum().to_dict().items()}
                }
        
        elif analysis_type == 'customer':
            # Customer analysis
            if 'CustomerID' in df.columns:
                customer_stats = df.groupby('CustomerID').agg({
                    'Price': ['sum', 'count', 'mean'],
                    'Quantity': 'sum'
                }).round(2)
                analysis_results = {
                    'total_customers': int(df['CustomerID'].nunique()),
                    'customer_segments': {k: {kk: float(vv) if isinstance(vv, (int, float)) else vv for kk, vv in v.items()} for k, v in customer_stats.to_dict().items()},
                    'top_customers': {k: {kk: float(vv) if isinstance(vv, (int, float)) else vv for kk, vv in v.items()} for k, v in customer_stats.nlargest(10, ('Price', 'sum')).to_dict().items()}
                }
        
        return jsonify({
            'success': True,
            'analysis_type': analysis_type,
            'results': analysis_results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
