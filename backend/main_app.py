import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from skimage.morphology import skeletonize
from skimage.measure import regionprops # 'cell' analizi için
from matplotlib.colors import hsv_to_rgb # Renk üretimi ve 'angles' analizi için
from scipy import stats as scipy_stats # 'branchLength' ve 'angles' analizleri için
from scipy.spatial import ConvexHull # 'angles' analizi için
import os

# --- Global Yapılandırma: Dosya Yolları ---
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR) # Proje kök dizini (safak/)

# 'public' klasörünün doğru yolunu tanımlayın
# Bu yol, proje kök dizininin (PROJECT_ROOT) altındaki 'public' klasörüdür.
STATIC_FOLDER_PATH = os.path.join(PROJECT_ROOT, 'public')

# Flask uygulamasını doğru static_folder yoluyla başlatın
app = Flask(__name__, static_folder=STATIC_FOLDER_PATH, static_url_path='/')
CORS(app) # CORS ayarınız varsa devam etsin

# Statik dosya servisi (index.html ve diğer frontend varlıkları için)
@app.route('/', methods=['GET'])
def serve_index():
    # app.static_folder artık PROJECT_ROOT/public'i işaret ediyor
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>', methods=['GET'])
def serve_static(path):
    # Bu route, /images/original/002.jpg gibi istekleri
    # PROJECT_ROOT/public/images/original/002.jpg gibi dosyalara yönlendirir.
    try:
        return send_from_directory(app.static_folder, path)
    except FileNotFoundError: # Veya werkzeug.exceptions.NotFound
        # Eğer SPA (Single Page Application) kullanıyorsanız ve path bir dosya değilse,
        # React Router'ın devralması için index.html'e fallback yapabilirsiniz.
        # Eğer doğrudan dosya bulunamadıysa 404 dönmesi daha doğru olabilir.
        # Bu kısmı projenizin ihtiyacına göre düzenleyin.
        # Eğer path /api/ ile başlamıyorsa ve dosya değilse index.html'e yönlendirme
        if not path.startswith('api/') and '.' not in path: # Basit bir kontrol
             app.logger.warn(f"Static file {path} not found, serving index.html as fallback for SPA.")
             return send_from_directory(app.static_folder, 'index.html')
        app.logger.error(f"Static file {path} not found.")
        return jsonify({"error": "File not found"}), 404


# --- Dosya Yolları (Bu kısımlarınız zaten doğru görünüyordu) ---
# Mask klasör yollarını güncelle
ORIGINAL_IMAGES_PATH = os.path.join(PROJECT_ROOT, 'public', 'images', 'original') # safak/public/images/original
CELL_MASKS_PATH     = os.path.join(PROJECT_ROOT, 'public', 'images', 'masks', 'cell') # safak/public/images/masks/cell
BRANCH_MASKS_PATH   = os.path.join(PROJECT_ROOT, 'public', 'images', 'masks', 'branch') # safak/public/images/masks/branch

# modeller backend içinde
CELL_MODEL_PATH     = os.path.join(BACKEND_DIR, 'best_cell_model.hdf5') # safak/backend/best_cell_model.hdf5
BEST_MODEL_PATH     = os.path.join(BACKEND_DIR, 'best_model.hdf5') # safak/backend/best_model.hdf5
BRANCH_MODEL_PATH   = os.path.join(BACKEND_DIR, 'best_branch_model.hdf5') # safak/backend/best_branch_model.hdf5

# --- Yardımcı Fonksiyonlar: Keras Modelleri için Metrikler ---
def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection + 1e-5) / (union + 1e-5)

def iou_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1e-5) / (union + 1e-5)

# --- Yardımcı Fonksiyonlar: Model Yükleme ---
def load_keras_model(model_path):
    """Özel metriklerle bir Keras modelini yükler."""
    with tf.keras.utils.custom_object_scope({'dice_coef': dice_coef, 'iou_score': iou_score}):
        model = tf.keras.models.load_model(model_path)
    return model

# --- Yardımcı Fonksiyonlar: Görüntü Kodlama/Kod Çözme ---
def decode_base64_image(base64_string):
    """Base64 string'ini OpenCV görüntüsüne dönüştürür."""
    image_data = base64_string.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Görüntü çözülemedi. Base64 string'ini ve görüntü formatını kontrol edin.")
    return img

def encode_image_to_base64(image_array):
    """OpenCV görüntüsünü base64 string'ine dönüştürür."""
    # Görüntünün BGR formatında olduğundan emin olalım (OpenCV varsayılanı)
    if image_array.ndim == 3 and image_array.shape[2] == 3: # Renkli görüntü ise
        pass # Zaten BGR olabilir
    elif image_array.ndim == 2 : # Gri tonlamalı ise
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR) # Encode için BGR'ye çevir
    
    # Görüntü tipini uint8 yapalım, imencode bunu bekler
    if image_array.dtype != np.uint8:
        if np.max(image_array) <= 1.0 and np.min(image_array) >=0.0 : # Normalizeli float ise
             image_array = (image_array * 255).astype(np.uint8)
        else: # Başka bir aralıkta float ise veya farklı bir tip ise
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)


    _, buffer = cv2.imencode('.png', image_array)
    img_b64 = base64.b64encode(buffer).decode()
    return f'data:image/png;base64,{img_b64}'

# --- Yardımcı Fonksiyon: Temel İstatistikler ---
def calculate_basic_stats(arr_like):
    """Bir dizi için temel istatistikleri (ortalama, min, max, std) hesaplar."""
    arr = list(arr_like) 
    if not arr: 
        return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
    return {
        'mean': float(np.mean(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'std': float(np.std(arr))
    }

# --- Yardımcı Fonksiyonlar: Dal Segmentasyon Çekirdeği ---
def _get_branch_segmentation_skeleton(orig_img_gray_resized_norm, mask_resized_binary, branch_model, best_model, threshold_value):
    """Dalları segmente etmek ve iskelet görüntüsünü (256x256) döndürmek için çekirdek mantık."""
    inp_tensor = orig_img_gray_resized_norm[None, ..., None] 

    pred_branch = branch_model.predict(inp_tensor)[0, ..., 0] 
    pred_best = best_model.predict(inp_tensor)[0, ..., 0]

    bin_branch = (pred_branch > 0.5).astype(np.uint8)
    bin_best = (pred_best > 0.5).astype(np.uint8)
    combined = cv2.bitwise_and(bin_branch, bin_best)
    
    if mask_resized_binary.shape != combined.shape:
         mask_resized_binary = cv2.resize(mask_resized_binary, combined.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    
    filtered = combined * mask_resized_binary

    _, bw_thresh = cv2.threshold((filtered * 255).astype(np.uint8), threshold_value, 255, cv2.THRESH_BINARY)
    skeleton_img = skeletonize(bw_thresh // 255).astype(np.uint8)
    return skeleton_img

def _trace_branches_from_skeleton_img(skeleton_img, filter_by_endpoints=True):
    """Bir iskelet görüntüsünden DFS kullanarak dalları izler ve isteğe bağlı olarak uç noktalara göre filtreler.
       Dal listesini (her dal (x,y) koordinat listesidir) ve uç nokta koordinat listesini döndürür.
    """
    endpoints = []
    h, w = skeleton_img.shape
    for r in range(h): 
        for c in range(w): 
            if skeleton_img[r, c] == 0:
                continue
            nh_count = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and skeleton_img[nr, nc]:
                        nh_count += 1
            if nh_count == 1: 
                endpoints.append((c, r)) 

    visited_pixels = set()
    traced_branches = []

    def _dfs_trace(start_x, start_y):
        stack = [(start_x, start_y)]
        current_branch_path = []
        while stack:
            px, py = stack.pop()
            if (px, py) not in visited_pixels:
                visited_pixels.add((px, py))
                current_branch_path.append((px, py))
                for dx_offset in [-1, 0, 1]:
                    for dy_offset in [-1, 0, 1]:
                        if dx_offset == 0 and dy_offset == 0:
                            continue
                        next_x, next_y = px + dx_offset, py + dy_offset
                        if 0 <= next_x < w and 0 <= next_y < h and \
                           skeleton_img[next_y, next_x] and (next_x, next_y) not in visited_pixels:
                            stack.append((next_x, next_y))
        return current_branch_path

    for r_idx in range(h): 
        for c_idx in range(w): 
            if skeleton_img[r_idx, c_idx] and (c_idx, r_idx) not in visited_pixels:
                path = _dfs_trace(c_idx, r_idx)
                if len(path) > 1: 
                    traced_branches.append(path)
    
    if filter_by_endpoints:
        final_branches_to_return = [b for b in traced_branches if any(pixel_coord in endpoints for pixel_coord in b)]
    else:
        final_branches_to_return = traced_branches
        
    return final_branches_to_return, endpoints

# --- Yardımcı Fonksiyon: Renk Üretimi ---
def generate_unique_bgr_colors(num_colors):
    """Benzersiz BGR renklerinden oluşan bir liste üretir."""
    colors_list = []
    if num_colors <= 0:
        return colors_list
    hues = np.linspace(0, 1, num_colors, endpoint=False)
    for h_val in hues:
        rgb_color = hsv_to_rgb([h_val, 1.0, 1.0]) 
        bgr_color = (int(rgb_color[2] * 255), int(rgb_color[1] * 255), int(rgb_color[0] * 255))
        colors_list.append(bgr_color)
    return colors_list

# --- Ana İşlem Rotası ---
@app.route('/process-image', methods=['POST'])
def process_image_route():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Girdi verisi sağlanmadı"}), 400
        
        analysis_type = data.get('analysisType')
        image_name = data.get('imageName')  # Örn: "002.jpg"
        threshold_param = int(data.get('threshold', 128))

        if not analysis_type:
            return jsonify({"error": "analysisType parametresi eksik"}), 400
        if not image_name:
            return jsonify({"error": "imageName parametresi eksik"}), 400

        # Debug log ekleyelim
        app.logger.debug(f"Received request - analysisType: {analysis_type}, imageName: {image_name}, threshold: {threshold_param}")

        # Orijinal görüntüyü yükle
        original_img_path = os.path.join(ORIGINAL_IMAGES_PATH, image_name)
        if not os.path.exists(original_img_path):
            return jsonify({"error": f"Görüntü bulunamadı: {original_img_path}"}), 404
        
        original_img = cv2.imread(original_img_path)
        if original_img is None:
            return jsonify({"error": "Görüntü yüklenemedi"}), 500

        # Maske yolunu belirleme (.png uzantılı)
        mask_filename = f"{image_name.split('.')[0]}.png"  # "002.jpg" -> "002.png"
        if analysis_type in ['cell', 'cellArea']:
            mask_path = os.path.join(CELL_MASKS_PATH, mask_filename)
        else:  # branch, branchLength, angles
            mask_path = os.path.join(BRANCH_MASKS_PATH, mask_filename)

        if not os.path.exists(mask_path):
            return jsonify({"error": f"Maske bulunamadı: {mask_path}"}), 404
            
    except ValueError as ve: 
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Görüntü çözülürken veya istek ayrıştırılırken hata: {str(e)}"}), 400

    common_best_model = None
    cell_seg_model = None
    branch_seg_model = None

    try:
        if analysis_type in ['cell', 'cellArea', 'branch', 'branchLength', 'angles']:
            common_best_model = load_keras_model(BEST_MODEL_PATH)
        if analysis_type in ['cell', 'cellArea']:
            cell_seg_model = load_keras_model(CELL_MODEL_PATH)
        if analysis_type in ['branch', 'branchLength', 'angles']:
            branch_seg_model = load_keras_model(BRANCH_MODEL_PATH)
    except Exception as e:
        return jsonify({"error": f"Model yüklenirken hata oluştu: {str(e)}"}), 500

    # --- Analiz Türü: Hücre Segmentasyonu ve İstatistikleri ---
    if analysis_type == 'cell':
        try:
            # Dinamik mask yolu oluştur
            mask_filename = f"{image_name.split('.')[0]}.png"  # örn: "040.jpg" -> "040.png"
            mask_path = os.path.join(CELL_MASKS_PATH, mask_filename)
            
            # Maske dosyasını kontrol et
            if not os.path.exists(mask_path):
                return jsonify({"error": f"Maske bulunamadı: {mask_path}"}), 404
                
            mask_img_for_cell = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask_img_for_cell is None:
                return jsonify({"error": f"Hücre maskesi yüklenemedi: {mask_path}"}), 500

            gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            resized_gray_img = cv2.resize(gray_img, (256, 256)) # uint8, [0,255]
            # DÜZELTME: Hücre modelleri için normalizasyon kaldırıldı.
            input_tensor_cell = np.expand_dims(resized_gray_img, axis=(0, -1)) # uint8, [0,255]

            pred_cell = cell_seg_model.predict(input_tensor_cell)[0]
            pred_best = common_best_model.predict(input_tensor_cell)[0]

            # Model çıktıları [0,1] aralığında olabilir, bu yüzden 255 ile çarpıyoruz.
            m1 = (pred_cell[:, :, 1] * 255).astype(np.uint8)
            m2 = (pred_best[:, :, 1] * 255).astype(np.uint8)
            combined_pred = cv2.addWeighted(m1, 0.5, m2, 0.5, 0)

            _, pred_binarized = cv2.threshold(combined_pred, threshold_param, 255, cv2.THRESH_BINARY)
            pred_mask_resized = cv2.resize(pred_binarized, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)

            if mask_img_for_cell.ndim == 3 and mask_img_for_cell.shape[2] in [3, 4]:
                mask_gray = cv2.cvtColor(mask_img_for_cell, cv2.COLOR_BGR2GRAY)
            else:
                mask_gray = mask_img_for_cell
            
            mask_gray_resized_for_cell = mask_gray # Yeniden adlandırma
            if mask_gray.shape[:2] != original_img.shape[:2]:
                 mask_gray_resized_for_cell = cv2.resize(mask_gray, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            _, gt_mask_binarized = cv2.threshold(mask_gray_resized_for_cell, threshold_param, 255, cv2.THRESH_BINARY)
            refined_segmentation = cv2.bitwise_and(pred_mask_resized, gt_mask_binarized)
            
            output_display_img = np.zeros_like(original_img)
            output_display_img[refined_segmentation > 0] = (0, 255, 0) # Yeşil

            contours, _ = cv2.findContours(refined_segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            H_orig, W_orig = refined_segmentation.shape[:2]

            perimeters_list, widths_list, heights_list, aspect_ratios_list, centroid_dists_list, feret_diams_list, eccentricities_list = [], [], [], [], [], [], []

            if contours:
                # refined_segmentation uint8 olmalı
                if refined_segmentation.dtype != np.uint8:
                    refined_segmentation_uint8 = refined_segmentation.astype(np.uint8)
                else:
                    refined_segmentation_uint8 = refined_segmentation

                num_labels, labels_map = cv2.connectedComponents(refined_segmentation_uint8)
                if num_labels > 1: 
                    props = regionprops(labels_map, intensity_image=refined_segmentation_uint8) # intensity_image eklendi
                    for p in props:
                        eccentricities_list.append(p.eccentricity)

                for cnt in contours:
                    perimeters_list.append(cv2.arcLength(cnt, True))
                    x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(cnt)
                    widths_list.append(w_bbox)
                    heights_list.append(h_bbox)
                    aspect_ratios_list.append(w_bbox / (h_bbox + 1e-6))
                    
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        dist_to_center = np.sqrt((cx - W_orig / 2) ** 2 + (cy - H_orig / 2) ** 2)
                        centroid_dists_list.append(dist_to_center)
                    
                    pts_contour = cnt.reshape(-1, 2)
                    max_d_feret = 0.0
                    if len(pts_contour) >= 2:
                        for i in range(len(pts_contour)):
                            for j in range(i + 1, len(pts_contour)):
                                dist_pts = np.linalg.norm(pts_contour[i] - pts_contour[j])
                                if dist_pts > max_d_feret: max_d_feret = dist_pts
                    feret_diams_list.append(max_d_feret)
            
            results = {
                'cellCount': len(contours),
                'cellDensity': len(contours) / (W_orig * H_orig) if (W_orig * H_orig) > 0 else 0,
                'meanPerimeter': calculate_basic_stats(perimeters_list)['mean'],
                'meanFeret': calculate_basic_stats(feret_diams_list)['mean'],
                'meanEccentricity': calculate_basic_stats(eccentricities_list)['mean'],
                'meanAspectRatio': calculate_basic_stats(aspect_ratios_list)['mean'],
                'meanCentroidDist': calculate_basic_stats(centroid_dists_list)['mean'],
                'meanBBoxWidth': calculate_basic_stats(widths_list)['mean'],
                'meanBBoxHeight': calculate_basic_stats(heights_list)['mean'],
                'histograms': {
                    'cellCount': { 
                        'labels': ['0-5', '5-10', '10-15', '15-20', '>20'],
                        'data': [len([x for x in widths_list if x < 5]), 
                                 len([x for x in widths_list if 5 <= x < 10]),
                                 len([x for x in widths_list if 10 <= x < 15]),
                                 len([x for x in widths_list if 15 <= x < 20]),
                                 len([x for x in widths_list if x >= 20])]
                    }
                }
            }
            encoded_image = encode_image_to_base64(output_display_img)
            
            # Cell visualization için kırmızı transparan overlay oluştur
            cell_visualization = original_img.copy()
            # Kırmızı maske oluştur - transparanlık kaldırıldı
            overlay = np.zeros_like(original_img)
            overlay[refined_segmentation > 0] = (0, 0, 255)  # BGR format - kırmızı
            # Maskeyi orijinal görüntüyle birleştir - addWeighted yerine doğrudan atama
            cell_visualization[refined_segmentation > 0] = (0, 0, 255)  # Tamamen kırmızı

            # Contour'ları çiz
            contours, _ = cv2.findContours(refined_segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(cell_visualization, contours, -1, (255, 255, 255), 3)  # Beyaz contour'lar
            # Base64'e çevir
            encoded_cell_vis = encode_image_to_base64(cell_visualization)
            return jsonify({
                'processedImage': encoded_image, 
                'cellVisualization': encoded_cell_vis,  # Yeni visualization
                'analysisResults': results
            })
        except Exception as e:
            app.logger.error(f"'cell' analizi sırasında hata: {e}", exc_info=True)
            return jsonify({"error": f"'cell' analizi sırasında hata: {str(e)}"}), 500

    # --- Analiz Türü: Dal Görselleştirme ---
    elif analysis_type == 'branch':
        try:
            # Dinamik mask yolu oluştur
            mask_filename = f"{image_name.split('.')[0]}.png"
            mask_path = os.path.join(BRANCH_MASKS_PATH, mask_filename)
            
            if not os.path.exists(mask_path):
                return jsonify({"error": f"Maske bulunamadı: {mask_path}"}), 404
                
            mask_for_branch = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_for_branch is None:
                return jsonify({"error": f"Dal maskesi yüklenemedi: {mask_path}"}), 500

            img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            img_gray_resized = cv2.resize(img_gray, (256, 256))
            # Dal modelleri için normalizasyon KORUNUYOR.
            img_gray_resized_norm = img_gray_resized.astype(np.float32) / 255.0 
            
            mask_resized = cv2.resize(mask_for_branch, (256, 256), interpolation=cv2.INTER_NEAREST)
            mask_resized_binary = (mask_resized > 127).astype(np.uint8)

            skeleton_256 = _get_branch_segmentation_skeleton(
                img_gray_resized_norm, mask_resized_binary, 
                branch_seg_model, common_best_model, threshold_param
            )
            branches_256, _ = _trace_branches_from_skeleton_img(skeleton_256, filter_by_endpoints=True)

            overlay_img = original_img.copy()
            scale_x = original_img.shape[1] / 256.0
            scale_y = original_img.shape[0] / 256.0
            branch_colors = generate_unique_bgr_colors(len(branches_256))
            branch_lengths_pixels = []

            for idx, branch_path_256 in enumerate(branches_256):
                color_to_draw = branch_colors[idx % len(branch_colors)] if branch_colors else (0,0,255) # Renk yoksa kırmızı
                points_orig_scale = [(int(x_coord * scale_x), int(y_coord * scale_y)) for x_coord, y_coord in branch_path_256]
                
                current_branch_length = 0.0
                if len(points_orig_scale) > 1:
                    for i in range(len(points_orig_scale) - 1):
                        cv2.line(overlay_img, points_orig_scale[i], points_orig_scale[i+1], color_to_draw, 12) 
                        current_branch_length += np.linalg.norm(np.array(points_orig_scale[i+1]) - np.array(points_orig_scale[i]))
                if current_branch_length > 0:
                    branch_lengths_pixels.append(current_branch_length)
            
            stats_branch = calculate_basic_stats(branch_lengths_pixels)
            hist_labels = [f"Dal {i+1}" for i in range(len(branch_lengths_pixels))]
            hist_data = [round(length, 2) for length in branch_lengths_pixels]

            results = {
                'totalBranches': len(branch_lengths_pixels),
                'averageLength': stats_branch['mean'],
                'minLength': stats_branch['min'],
                'maxLength': stats_branch['max'],
                'stdLength': stats_branch['std'],
                'histograms': { 
                    'branchLength': { 'labels': hist_labels, 'data': hist_data }
                }
            }
            encoded_image = encode_image_to_base64(overlay_img)
            return jsonify({'processedImage': encoded_image, 'analysisResults': results})
        except Exception as e:
            app.logger.error(f"'branch' analizi sırasında hata: {e}", exc_info=True)
            return jsonify({"error": f"'branch' analizi sırasında hata: {str(e)}"}), 500

    # --- Analiz Türü: Hücre Alan Analizi ---
    elif analysis_type == 'cellArea':
        try:
            gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            resized_gray_img = cv2.resize(gray_img, (256, 256)) # uint8, [0,255]
            # DÜZELTME: Hücre modelleri için normalizasyon kaldırıldı.
            input_tensor_cell_area = np.expand_dims(resized_gray_img, axis=(0, -1)) # uint8, [0,255]

            pred_cell = cell_seg_model.predict(input_tensor_cell_area)[0]
            pred_best = common_best_model.predict(input_tensor_cell_area)[0]

            src1 = (pred_cell[:, :, 1] * 255).astype(np.uint8)
            src2 = (pred_best[:, :, 1] * 255).astype(np.uint8)
            
            if src1.shape != src2.shape: 
                src2 = cv2.resize(src2, (src1.shape[1], src1.shape[0]))
                
            combined_preds = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)
            _, binary_thresh = cv2.threshold(combined_preds, threshold_param, 255, cv2.THRESH_BINARY)
            binary_morphed = cv2.morphologyEx(binary_thresh, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

            num_labels, _, stats_cca, centroids_cca = cv2.connectedComponentsWithStats(binary_morphed, connectivity=8)
            
            output_img_256 = cv2.resize(original_img, (256, 256))
            areas_list = []
            for i in range(1, num_labels): 
                area_cca = stats_cca[i, cv2.CC_STAT_AREA]
                areas_list.append(float(area_cca))
                
                center_pt = (int(centroids_cca[i][0]), int(centroids_cca[i][1]))
                radius_val = int(np.sqrt(area_cca / np.pi)) 
                rand_color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
                
                cv2.circle(output_img_256, center_pt, radius_val, rand_color, 1)
                cv2.putText(output_img_256, f"{area_cca}", (center_pt[0]-10, center_pt[1]+5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

            final_output_img = cv2.resize(output_img_256, (original_img.shape[1], original_img.shape[0]))
            encoded_image = encode_image_to_base64(final_output_img)
            
            area_stats = calculate_basic_stats(areas_list)
            results = {
                'totalCells': len(areas_list),
                'averageArea': area_stats['mean'],
                'minArea': area_stats['min'],
                'maxArea': area_stats['max'],
                'std': area_stats['std'],
                'histograms': {
                    'cellArea': { 
                        'labels': ['0-100', '100-200', '200-300', '300-400', '>400'],
                        'data': [len([x for x in areas_list if x < 100]),
                                 len([x for x in areas_list if 100 <= x < 200]),
                                 len([x for x in areas_list if 200 <= x < 300]),
                                 len([x for x in areas_list if 300 <= x < 400]),
                                 len([x for x in areas_list if x >= 400])]
                    }
                }
            }
            return jsonify({'processedImage': encoded_image, 'analysisResults': results})
        except Exception as e:
            app.logger.error(f"'cellArea' analizi sırasında hata: {e}", exc_info=True)
            return jsonify({"error": f"'cellArea' analizi sırasında hata: {str(e)}"}), 500

    # --- Analiz Türü: Detaylı Dal Uzunluk Analizi ---
    elif analysis_type == 'branchLength':
        try:
            # Dinamik mask yolu oluştur
            mask_filename = f"{image_name.split('.')[0]}.png"
            mask_path = os.path.join(BRANCH_MASKS_PATH, mask_filename)
            
            if not os.path.exists(mask_path):
                return jsonify({"error": f"Maske bulunamadı: {mask_path}"}), 404
                
            # Maskeyi yükle
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return jsonify({"error": f"Dal maskesi yüklenemedi: {mask_path}"}), 500

            # Gri tonlamaya çevir ve normalize et
            gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (256, 256)).astype(np.float32) / 255.0
            inp_tensor = resized[None, ..., None]

            # Tahminleri al
            pred_branch = branch_seg_model.predict(inp_tensor)[0, ..., 0]
            pred_best = common_best_model.predict(inp_tensor)[0, ..., 0]

            # Maskeyi yeniden boyutlandır ve binary hale getir
            mask_resized = cv2.resize(mask, (256, 256)) // 255

            # Tahminleri birleştir ve maskele
            combined = cv2.bitwise_and(
                (pred_branch > 0.5).astype(np.uint8),
                (pred_best > 0.5).astype(np.uint8)
            ) * mask_resized

            # Skeletonize işlemi
            _, bw = cv2.threshold((combined * 255).astype(np.uint8), threshold_param, 255, cv2.THRESH_BINARY)
            skel = skeletonize(bw // 255).astype(np.uint8)

            # Branch tespiti için DFS
            branches = []
            visited = set()
            h, w = skel.shape

            def dfs(x, y):
                stack = [(x, y)]
                branch = []
                while stack:
                    cx, cy = stack.pop()
                    if (cx, cy) not in visited:
                        visited.add((cx, cy))
                        branch.append((cx, cy))
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                nx, ny = cx + dx, cy + dy
                                if 0 <= nx < w and 0 <= ny < h and skel[ny, nx]:
                                    stack.append((nx, ny))
                return branch

            # Dalları bul
            for y in range(h):
                for x in range(w):
                    if skel[y, x] and (x, y) not in visited:
                        b = dfs(x, y)
                        if len(b) > 1:
                            branches.append(b)

            # Görselleştirme ve uzunluk hesaplama
            overlay = original_img.copy()
            lengths = []
            scale_x = original_img.shape[1] / 256
            scale_y = original_img.shape[0] / 256

            for branch in branches:
                pts = [(int(x*scale_x), int(y*scale_y)) for x, y in branch]
                
                # Branch çizgileri
                for i in range(len(pts)-1):
                    cv2.line(overlay, pts[i], pts[i+1], (0,0,255), 5)
                
                # Uzunluk hesapla
                length = sum(np.linalg.norm(np.array(pts[i])-np.array(pts[i-1])) 
                        for i in range(1, len(pts)))
                lengths.append(length)

                # Branch üzerine yazı
                center = tuple(np.mean(pts, axis=0).astype(int))
                text = f"{length:.1f}"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
                
                # Arka plan dikdörtgeni
                cv2.rectangle(overlay, 
                            (center[0] - text_width//2 - 5, center[1] - text_height//2 - 5),
                            (center[0] + text_width//2 + 5, center[1] + text_height//2 + 5),
                            (255, 255, 255), -1)
                
                # Yazı
                cv2.putText(overlay, text, 
                        (center[0] - text_width//2, center[1] + text_height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

            # İstatistikleri hesapla
            def calculate_advanced_stats(lengths):
                if not lengths:
                    return {}

                arr = np.array(lengths)
                sorted_len = np.sort(arr)[::-1]

                return {
                    'medianLength': float(np.median(arr)),
                    'varianceLength': float(np.var(arr)),
                    'longest5Mean': float(np.mean(sorted_len[:5])) if len(arr)>=5 else 0.0,
                    'shortest5Mean': float(np.mean(sorted_len[-5:])) if len(arr)>=5 else 0.0,
                    'percentile25': float(np.percentile(arr, 25)),
                    'percentile75': float(np.percentile(arr, 75)),
                    'iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
                    'lengthSum': float(np.sum(arr)),
                    'normalizedLengths': (arr / arr.max()).tolist() if arr.max() > 0 else [],
                    'lengthSkewness': float(scipy_stats.skew(arr)) if len(arr) > 1 else 0.0,
                    'lengthKurtosis': float(scipy_stats.kurtosis(arr)) if len(arr) > 1 else 0.0
                }

            # İstatistikleri hesapla
            advanced_stats = calculate_advanced_stats(lengths)
            basic_stats = {
                'totalBranches': len(lengths),
                'averageLength': float(np.mean(lengths)) if lengths else 0.0,
                'minLength': float(np.min(lengths)) if lengths else 0.0,
                'maxLength': float(np.max(lengths)) if lengths else 0.0,
                'stdLength': float(np.std(lengths)) if lengths else 0.0
            }

            # Histogram verisi
            hist_bins = np.arange(0, 5000 + 100, 100)
            hist_values, _ = np.histogram(lengths, bins=hist_bins)
            hist_labels = [f"{hist_bins[i]}-{hist_bins[i+1]}" for i in range(len(hist_bins)-1)]

            # Sonuçları döndür
            encoded_image = encode_image_to_base64(overlay)
            return jsonify({
                'processedImage': encoded_image,
                'analysisResults': {
                    **basic_stats,
                    **advanced_stats,
                    'histograms': {
                        'branchLength': {
                            'labels': hist_labels,
                            'data': hist_values.tolist()
                        }
                    }
                }
            })
        except Exception as e:
            app.logger.error(f"'branchLength' analizi sırasında hata: {e}", exc_info=True)
            return jsonify({"error": f"'branchLength' analizi sırasında hata: {str(e)}"}), 500

    # --- Analiz Türü: Dal Açıları Analizi ---
    elif analysis_type == 'angles':
        try:
            # Dinamik mask yolu oluştur
            mask_filename = f"{image_name.split('.')[0]}.png"
            mask_path = os.path.join(BRANCH_MASKS_PATH, mask_filename)
            
            if not os.path.exists(mask_path):
                return jsonify({"error": f"Maske bulunamadı: {mask_path}"}), 404
                
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return jsonify({"error": f"Dal maskesi yüklenemedi: {mask_path}"}), 500

        # Görüntü işleme
            gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            inp = cv2.resize(gray, (256, 256)).astype(np.float32) / 255.0
            inp_tensor = inp[None, ..., None]

            # Model tahminleri
            pred_branch = branch_seg_model.predict(inp_tensor)[0, ..., 0]
            pred_best = common_best_model.predict(inp_tensor)[0, ..., 0]

            # Binary işlemler
            bin_branch = (pred_branch > 0.5).astype(np.uint8)
            bin_best = (pred_best > 0.5).astype(np.uint8)
            combined = cv2.bitwise_and(bin_branch, bin_best)

            mask_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST) // 255
            filtered = combined * mask_resized

            _, bw = cv2.threshold((filtered * 255).astype(np.uint8), threshold_param, 255, cv2.THRESH_BINARY)
            skel = skeletonize(bw // 255).astype(np.uint8)

            # Uç noktaları ve dalları bul
            endpoints = []
            h, w = skel.shape
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if skel[y, x]:
                        neigh = skel[y-1:y+2, x-1:x+2]
                        if np.sum(neigh) == 2:
                            endpoints.append((x, y))

            visited = set()
            branches = []

            def dfs(x, y):
                stack = [(x, y)]
                branch = []
                while stack:
                    cx, cy = stack.pop()
                    if (cx, cy) not in visited:
                        visited.add((cx, cy))
                        branch.append((cx, cy))
                        for nx in range(cx-1, cx+2):
                            for ny in range(cy-1, cy+2):
                                if 0 <= nx < w and 0 <= ny < h and skel[ny, nx]:
                                    stack.append((nx, ny))
                return branch

            for y in range(h):
                for x in range(w):
                    if skel[y, x] and (x, y) not in visited:
                        b = dfs(x, y)
                        if len(b) > 1 and any(p in endpoints for p in b):
                            branches.append(b)

            # Görselleştirme ve açı hesaplama
            overlay = original_img.copy()
            scale_x = original_img.shape[1] / 256
            scale_y = original_img.shape[0] / 256
            angles = []

            for branch in branches:
                branch_endpoints = [p for p in branch if p in endpoints]
                for i in range(len(branch_endpoints)):
                    for j in range(i+1, len(branch_endpoints)):
                        p1, p2 = branch_endpoints[i], branch_endpoints[j]
                        p1r = (int(p1[0]*scale_x), int(p1[1]*scale_y))
                        p2r = (int(p2[0]*scale_x), int(p2[1]*scale_y))
                        dy = p2r[1] - p1r[1]
                        dx = p2r[0] - p1r[0]
                        angle = np.degrees(np.arctan2(dy, dx)) % 360
                        dist = np.hypot(dx, dy)
                        angles.append({
                            'points': [p1r, p2r],
                            'angle': float(round(angle,2)),
                            'distance': float(round(dist,2))
                        })
                        
                        color = tuple(int(c*255) for c in hsv_to_rgb([angle/360,1,1])[::-1])
                        cv2.line(overlay, p1r, p2r, color, 2)
                        text_pos = ((p1r[0]+p2r[0])//2, (p1r[1]+p2r[1])//2)
                        cv2.putText(overlay, f"{round(angle,1)} deg", text_pos,
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
                        cv2.putText(overlay, f"{round(angle,1)} deg", text_pos,
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # İstatistikleri hesapla
            values = [a['angle'] for a in angles]
            stats = {
                'average': float(np.mean(values)) if values else 0.0,
                'min': float(np.min(values)) if values else 0.0,
                'max': float(np.max(values)) if values else 0.0,
                'std': float(np.std(values)) if values else 0.0,
                'resultantVectorLength': float(np.sqrt(
                    np.sum(np.cos(np.radians(values)))**2 +
                    np.sum(np.sin(np.radians(values)))**2
                ) / len(values)) if values else 0.0,
                'angleSkewness': float(scipy_stats.skew(values)) if len(values) > 2 else 0.0,
                'angles': angles
            }

            # Histogram hesaplama
            hist_bins = np.arange(0, 391, 30)  # 0, 30, 60, ..., 360
            hist_values, _ = np.histogram(values, bins=hist_bins)
            hist_labels = [f"{hist_bins[i]}-{hist_bins[i+1]}" for i in range(len(hist_bins)-1)]
            
            stats['histograms'] = {
                'angles': {
                    'labels': hist_labels,
                    'data': hist_values.tolist()
                }
            }

            encoded_image = encode_image_to_base64(overlay)
            return jsonify({
                'processedImage': encoded_image,
                'analysisResults': stats
            })

        except Exception as e:
            app.logger.error(f"'angles' analizi sırasında hata: {str(e)}", exc_info=True)
            return jsonify({"error": f"'angles' analizi sırasında hata: {str(e)}"}), 500
            
    else:
        return jsonify({"error": f"Bilinmeyen veya uygulanmamış analiz türü: {analysis_type}"}), 400

# Yeni endpoint ekle
@app.route('/get-images', methods=['GET'])
def get_images():
    try:
        # Sadece .jpg uzantılı orijinal görüntüleri listele
        image_files = [f for f in os.listdir(ORIGINAL_IMAGES_PATH) 
                      if f.endswith('.jpg')]
        
        # Görüntüleri sayısal sıraya göre sırala (002.jpg, 003.jpg, ...)
        image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        
        return jsonify({
            'images': image_files,
            'total': len(image_files)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Flask'in kendi logger'ını kullanmak için:
    # import logging
    # app.logger.setLevel(logging.ERROR) # veya DEBUG, INFO
    app.run(host='0.0.0.0', port=5000, debug=True)