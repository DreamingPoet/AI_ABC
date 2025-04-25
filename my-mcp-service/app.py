import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import hashlib
from datetime import datetime

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置文件
UPLOAD_FOLDER = 'E:\AIProjects\AITest\MCPFileTest'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'json'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_info(filepath):
    stat = os.stat(filepath)
    return {
        'size': stat.st_size,
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'is_dir': os.path.isdir(filepath)
    }

@app.route('/api/files', methods=['GET'])
def list_files():
    path = request.args.get('path', '')
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], path)
    
    if not os.path.exists(full_path):
        return jsonify({'error': 'Path not found'}), 404
    
    if not os.path.isdir(full_path):
        return jsonify({'error': 'Not a directory'}), 400
    
    files = []
    for item in os.listdir(full_path):
        item_path = os.path.join(full_path, item)
        file_info = get_file_info(item_path)
        file_info['name'] = item
        files.append(file_info)
    
    return jsonify({'path': path, 'files': files})

@app.route('/api/files/content', methods=['GET'])
def read_file():
    path = request.args.get('path', '')
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], path)
    
    if not os.path.exists(full_path):
        return jsonify({'error': 'File not found'}), 404
    
    if os.path.isdir(full_path):
        return jsonify({'error': 'Cannot read content of a directory'}), 400
    
    try:
        with open(full_path, 'r') as f:
            content = f.read()
        return jsonify({
            'path': path,
            'content': content,
            'info': get_file_info(full_path)
        })
    except UnicodeDecodeError:
        return jsonify({'error': 'File is not text-based'}), 400

@app.route('/api/files', methods=['POST'])
def create_file():
    path = request.form.get('path', '')
    content = request.form.get('content', '')
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], path)
    
    if os.path.exists(full_path):
        return jsonify({'error': 'File already exists'}), 400
    
    try:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)
        return jsonify({'message': 'File created successfully', 'path': path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/files/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'info': get_file_info(save_path)
        })
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/files', methods=['DELETE'])
def delete_file():
    path = request.args.get('path', '')
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], path)
    
    if not os.path.exists(full_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        if os.path.isdir(full_path):
            os.rmdir(full_path)
        else:
            os.remove(full_path)
        return jsonify({'message': 'Deleted successfully', 'path': path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/files/move', methods=['POST'])
def move_file():
    source = request.form.get('source', '')
    target = request.form.get('target', '')
    
    full_source = os.path.join(app.config['UPLOAD_FOLDER'], source)
    full_target = os.path.join(app.config['UPLOAD_FOLDER'], target)
    
    if not os.path.exists(full_source):
        return jsonify({'error': 'Source file not found'}), 404
    
    try:
        os.rename(full_source, full_target)
        return jsonify({
            'message': 'File moved successfully',
            'source': source,
            'target': target
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/files/download', methods=['GET'])
def download_file():
    path = request.args.get('path', '')
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], path)
    
    if not os.path.exists(full_path):
        return jsonify({'error': 'File not found'}), 404
    
    if os.path.isdir(full_path):
        return jsonify({'error': 'Cannot download a directory'}), 400
    
    return send_file(full_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)