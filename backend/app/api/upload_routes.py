"""
Upload and file management API routes
"""
from flask import Blueprint, request, jsonify, current_app, send_file
from urllib.parse import unquote
import os
import pandas as pd
from werkzeug.utils import secure_filename
from app.core.data_processing import load_data_file, merge_dataframes, analyze_uploaded_file

upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle multiple file uploads and merge them"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Validate file count
        if len(files) > 10:
            return jsonify({'error': 'Maximum 10 files allowed'}), 400
        
        if len(files) < 1:
            return jsonify({'error': 'Minimum 1 file required'}), 400
        
        uploaded_files = []
        dataframes = []
        
        # Process each file
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Load the data
                df = load_data_file(filepath)
                if df is not None:
                    dataframes.append(df)
                    uploaded_files.append({
                        'filename': filename,
                        'rows': len(df),
                        'columns': len(df.columns)
                    })
        
        if not dataframes:
            return jsonify({'error': 'No valid data files found'}), 400
        
        # Merge all dataframes
        merged_data = merge_dataframes(dataframes)
        
        # Save merged data
        merged_filename = f"merged_data_{int(pd.Timestamp.now().timestamp())}.csv"
        merged_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], merged_filename)
        
        # Ensure the upload folder exists
        os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        print(f"[INFO] Saving merged data to: {merged_filepath}")
        print(f"[INFO] Path length: {len(merged_filepath)} characters")
        
        # Use utf-8-sig for better Windows compatibility
        try:
            merged_data.to_csv(merged_filepath, index=False, encoding='utf-8-sig', lineterminator='\n')
            print("[SUCCESS] Merged data saved successfully")
        except Exception as csv_error:
            # If that fails, try without specifying encoding
            print(f"[ERROR] Failed to save CSV: {csv_error}")
            print(f"[ERROR] Filepath: {merged_filepath}")
            raise  # Re-raise the error so it's properly reported
        
        # Get simple preview data (first 50 rows) - just convert to basic format
        preview_data = []
        for i in range(min(50, len(merged_data))):
            row = {}
            for col in merged_data.columns:
                value = merged_data.iloc[i][col]
                # Convert NaN to None for JSON, keep everything else as-is
                if pd.isna(value):
                    row[col] = None
                else:
                    row[col] = str(value)  # Convert to string to avoid JSON issues
            preview_data.append(row)
        
        return jsonify({
            'success': True,
            'uploaded_files': uploaded_files,
            'merged_file': {
                'filename': merged_filename,
                'filepath': merged_filepath,
                'total_rows': len(merged_data),
                'total_columns': len(merged_data.columns),
                'column_names': merged_data.columns.tolist()
            },
            'preview_data': preview_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@upload_bp.route('/api/preview/<filename>', methods=['GET'])
def get_preview(filename):
    """Get preview of merged data"""
    try:
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Load data
        df = load_data_file(filepath)
        if df is None:
            return jsonify({'error': 'Could not load file'}), 400
        
        # Get preview data (first 50 rows)
        preview_data = []
        for i in range(min(50, len(df))):
            row = {}
            for col in df.columns:
                value = df.iloc[i][col]
                if pd.isna(value):
                    row[col] = None
                else:
                    row[col] = str(value)
            preview_data.append(row)
        
        return jsonify({
            'success': True,
            'preview_data': preview_data,
            'total_rows': len(df),
            'columns': df.columns.tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@upload_bp.route('/api/data/<filename>', methods=['GET'])
def get_data(filename):
    """Get data from uploaded file"""
    try:
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Load data
        df = load_data_file(filepath)
        if df is None:
            return jsonify({'error': 'Could not load file'}), 400
        
        # Convert to JSON - handle NaN values properly
        # Replace NaN values with None for JSON serialization
        df_clean = df.fillna('N/A')
        data = df_clean.to_dict('records')
        
        return jsonify({
            'success': True,
            'data': data,
            'columns': df.columns.tolist(),
            'shape': df.shape
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@upload_bp.route('/api/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """Download a file from the uploads folder"""
    try:
        # URL decode the filename in case it contains encoded characters
        filename = unquote(filename)
        
        # Remove any path traversal attempts
        filename = os.path.basename(filename)
        
        # Build the full file path
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        
        # Additional security: ensure the file is within the uploads directory
        filepath = os.path.normpath(os.path.abspath(filepath))
        uploads_dir = os.path.normpath(os.path.abspath(current_app.config['UPLOAD_FOLDER']))
        
        if not filepath.startswith(uploads_dir):
            return jsonify({'error': 'Invalid file path - security violation'}), 400
        
        if not os.path.exists(filepath):
            return jsonify({'error': f'File not found: {filename}'}), 404
        
        if not os.path.isfile(filepath):
            return jsonify({'error': f'Not a file: {filename}'}), 400
        
        # Determine the mimetype based on file extension
        if filename.endswith('.csv'):
            mimetype = 'text/csv'
        elif filename.endswith('.xlsx'):
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        elif filename.endswith('.xls'):
            mimetype = 'application/vnd.ms-excel'
        else:
            mimetype = 'application/octet-stream'
        
        # Use the filename as-is for download
        download_filename = filename
        
        # Try to use download_name (Flask 2.0+), fallback to attachment_filename (older Flask)
        try:
            return send_file(
                filepath,
                mimetype=mimetype,
                as_attachment=True,
                download_name=download_filename
            )
        except TypeError:
            # Fallback for older Flask versions (Flask < 2.0)
            try:
                return send_file(
                    filepath,
                    mimetype=mimetype,
                    as_attachment=True,
                    attachment_filename=download_filename
                )
            except Exception:
                # Last resort: send without specifying download name
                return send_file(
                    filepath,
                    mimetype=mimetype,
                    as_attachment=True
                )
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        file_name = filename if 'filename' in locals() else 'unknown'
        print(f"Download error for file '{file_name}': {error_details}")
        return jsonify({
            'error': str(e),
            'message': f'Failed to download file: {file_name}'
        }), 500
