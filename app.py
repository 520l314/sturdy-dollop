import os
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import io
import scipy.io as sio
import pandas as pd
import spectral
import unmixing   # 我们的 GMOGH 算法

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

ALLOWED_EXTENSIONS = {'mat', 'hdr', 'csv', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 存储上传的数据和端元库
uploaded_data = {}
uploaded_library = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_data', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件类型'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        ext = filename.rsplit('.', 1)[1].lower()
        if ext == 'mat':
            mat = sio.loadmat(filepath)
            data_var = None
            for var in ['data', 'Y', 'X', 'HSI', 'hyper', 'img', 'hsi']:
                if var in mat:
                    data_var = var
                    break
            if data_var is None:
                for k, v in mat.items():
                    if not k.startswith('__') and isinstance(v, np.ndarray) and v.ndim >= 2:
                        data_var = k
                        break
            if data_var is None:
                return jsonify({'error': '未找到高光谱数据变量'}), 400
            data = mat[data_var]
            # 确保形状为 (rows, cols, bands)
            if data.ndim == 2:
                # 可能是 (pixels, bands) 或 (bands, pixels)
                if data.shape[0] < data.shape[1]:
                    data = data.T
                data = data[np.newaxis, :, :]
            elif data.ndim != 3:
                return jsonify({'error': '数据维度不是2或3'}), 400
            rows, cols, bands = data.shape
            data_flat = data.reshape((rows*cols, bands))
            uploaded_data['data'] = data_flat.tolist()
            uploaded_data['original_shape'] = (rows, cols)
            uploaded_data['bands'] = bands

        elif ext == 'hdr':
            base = os.path.splitext(filepath)[0]
            img = spectral.open_image(base)
            data = img.load()
            rows, cols, bands = data.shape
            data_flat = data.reshape((rows*cols, bands))
            uploaded_data['data'] = data_flat.tolist()
            uploaded_data['original_shape'] = (rows, cols)
            uploaded_data['bands'] = bands

        elif ext in ['csv', 'txt']:
            data = pd.read_csv(filepath, header=None).values
            if data.ndim != 2:
                return jsonify({'error': 'CSV数据应为二维'}), 400
            rows, cols = 1, data.shape[0]
            bands = data.shape[1]
            data_flat = data.reshape((rows*cols, bands))
            uploaded_data['data'] = data_flat.tolist()
            uploaded_data['original_shape'] = (rows, cols)
            uploaded_data['bands'] = bands

        return jsonify({
            'message': '数据上传成功',
            'shape': uploaded_data['original_shape'],
            'bands': uploaded_data['bands']
        })
    except Exception as e:
        return jsonify({'error': f'文件处理失败: {str(e)}'}), 500

@app.route('/upload_library', methods=['POST'])
def upload_library():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件类型'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        ext = filename.rsplit('.', 1)[1].lower()
        if ext == 'mat':
            mat = sio.loadmat(filepath)
            lib_var = None
            for var in ['library', 'A', 'L', 'endmembers', 'endmember_library']:
                if var in mat:
                    lib_var = var
                    break
            if lib_var is None:
                for k, v in mat.items():
                    if not k.startswith('__') and isinstance(v, np.ndarray) and v.ndim == 2:
                        lib_var = k
                        break
            if lib_var is None:
                return jsonify({'error': '未找到端元库变量'}), 400
            lib = mat[lib_var]
        elif ext in ['csv', 'txt']:
            lib = pd.read_csv(filepath, header=None).values
        else:
            return jsonify({'error': '不支持的文件格式'}), 400

        # 确保形状为 (bands, M)
        if 'bands' in uploaded_data:
            bands = uploaded_data['bands']
            if lib.shape[0] == bands:
                pass
            elif lib.shape[1] == bands:
                lib = lib.T
            else:
                return jsonify({'error': f'端元库波段数({lib.shape[0]})与数据波段数({bands})不匹配'}), 400

        uploaded_library['library'] = lib.tolist()
        uploaded_library['shape'] = lib.shape
        return jsonify({'message': '端元库上传成功', 'shape': lib.shape})
    except Exception as e:
        return jsonify({'error': f'端元库处理失败: {str(e)}'}), 500

@app.route('/unmix', methods=['POST'])
def unmix():
    if 'data' not in uploaded_data:
        return jsonify({'error': '请先上传高光谱数据'}), 400
    if 'library' not in uploaded_library:
        return jsonify({'error': '请先上传端元库'}), 400

    # 获取前端参数
    target_k = int(request.form.get('target_k', 5))
    beta = float(request.form.get('beta', 0.9))
    max_iter = int(request.form.get('max_iter', 3000))
    lr = float(request.form.get('lr', 0.05))
    pop_size = int(request.form.get('pop_size', 30))

    data = np.array(uploaded_data['data'])          # (n_pixels, n_bands)
    library = np.array(uploaded_library['library']) # (bands, M)
    rows, cols = uploaded_data['original_shape']

    try:
        selected_idx, abundance = unmixing.gmogh_unmixing(
            Y=data,
            library=library,
            target_k=target_k,
            beta=beta,
            max_iters=max_iter,
            lr=lr,
            pop_size=pop_size,
            verbose=False
        )
    except Exception as e:
        return jsonify({'error': f'解混失败: {str(e)}'}), 500

    n_endmembers = abundance.shape[0]
    # 丰度图重塑为 (K, H, W)
    abundance_img = abundance.reshape(n_endmembers, rows, cols)

    # 显示丰度图（最多显示前 4 个）
    n_display = min(n_endmembers, 4)
    fig, axes = plt.subplots(1, n_display, figsize=(4*n_display, 3))
    if n_display == 1:
        axes = [axes]
    for i in range(n_display):
        im = axes[i].imshow(abundance_img[i], cmap='jet', vmin=0, vmax=1)
        axes[i].set_title(f'端元 {selected_idx[i]+1}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # 存储丰度用于下载（转置为 pixels × K）
    uploaded_data['abundance'] = abundance.T.tolist()
    uploaded_data['selected_indices'] = selected_idx.tolist()

    return jsonify({
        'status': 'success',
        'image': img_base64,
        'n_endmembers': n_endmembers,
        'selected_indices': selected_idx.tolist()
    })

@app.route('/download_abundance', methods=['GET'])
def download_abundance():
    if 'abundance' not in uploaded_data:
        return jsonify({'error': '没有丰度结果'}), 400
    abundance = np.array(uploaded_data['abundance'])  # (pixels, K)
    df = pd.DataFrame(abundance)
    output = io.StringIO()
    df.to_csv(output, index=False, header=False)
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='abundance.csv'
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)