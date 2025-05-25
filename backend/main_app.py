import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from skimage.morphology import skeletonize
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import io
app = Flask(__name__)
CORS(app)



@app.route('/process-image', methods=['POST'])
def process_image():
    
    data = request.get_json()
    analysis_type = data.get('analysisType')
    threshold = int(data.get('threshold', 128))
        
        # Görüntüyü decode et
    image_data = data['image'].split(',')[1]
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if analysis_type == 'cell':

        if isinstance(img, np.ndarray):  # If OpenCV (NumPy array)
            height, width = img.shape[:2]
        else:  # If PIL Image
            width, height = img.size

        def dice_coef(y_true, y_pred):
            intersection = tf.reduce_sum(y_true * y_pred)
            union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
            return (2.0 * intersection + 1e-5) / (union + 1e-5)

        def iou_score(y_true, y_pred):
            intersection = tf.reduce_sum(y_true * y_pred)
            union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
            return (intersection + 1e-5) / (union + 1e-5)

        def load_models():
            """Load the cell and best segmentation models with custom metrics."""
            with tf.keras.utils.custom_object_scope({'dice_coef': dice_coef, 'iou_score': iou_score}):
                CELL_MODEL_PATH     = r"C:\Users\safak\Desktop\pc12_dataset\safak\backend\best_cell_model.hdf5"
                BEST_MODEL_PATH     = r"C:\Users\safak\Desktop\pc12_dataset\safak\backend\best_model.hdf5"
                cell_model = tf.keras.models.load_model(CELL_MODEL_PATH)
                best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
            return cell_model, best_model

        def process_cell_segmentation(img, mask_img, cell_model, best_model, threshold=threshold):
            """
            Predict cell segmentation, refine with mask, and return black-background output
            where only segmented cell regions are green.
            """
            # 1) Prepare model input
            gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (256, 256))
            inp     = np.expand_dims(resized, axis=(0, -1))

            # 2) Model predictions
            pred_cell = cell_model.predict(inp)[0]
            pred_best = best_model.predict(inp)[0]

            # 3) Extract channel and combine
            m1       = (pred_cell[:, :, 1] * 255).astype(np.uint8)
            m2       = (pred_best[:, :, 1] * 255).astype(np.uint8)
            combined = cv2.addWeighted(m1, 0.5, m2, 0.5, 0)

            # 4) Binarize prediction
            _, pred_bin = cv2.threshold(combined, threshold, 255, cv2.THRESH_BINARY)
            pred_mask   = cv2.resize(pred_bin, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            # 5) Process ground-truth mask
            mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY) if mask_img.ndim == 3 else mask_img
            _, gt_bin  = cv2.threshold(mask_gray, threshold, 255, cv2.THRESH_BINARY)

            # 6) Refine prediction by mask (logical AND)
            refined = cv2.bitwise_and(pred_mask, gt_bin)

            # 7) Create black background and paint refined regions green
            output = np.zeros_like(img)
            output[refined > 0] = (0, 255, 0)

            return output, refined
        MASK_IMAGE_PATH     = r"C:\Users\safak\Desktop\pc12_dataset\safak\backend\cell_mask\023.png"
        mask = cv2.imread(MASK_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
        cell_model, best_model = load_models()
        processed_img, refined_mask = process_cell_segmentation(img, mask, cell_model, best_model, threshold)

        # Konturları bul
        contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        H, W = refined_mask.shape[:2]

        perimeters = []
        widths = []
        heights = []
        aspect_ratios = []
        centroid_dists = []
        feret_diams = []
        eccentricities = []

        # scikit-image regionprops ile eksantriklik hesaplama
        num_labels, labels = cv2.connectedComponents(refined_mask)
        from skimage.measure import regionprops
        for prop in regionprops(labels):
            eccentricities.append(prop.eccentricity)

        for cnt in contours:
            # 1) Perimetre
            perimeters.append(cv2.arcLength(cnt, True))
            # 2&3) Bounding-box
            x, y, w, h = cv2.boundingRect(cnt)
            widths.append(w)
            heights.append(h)
            # 4) Aspect ratio
            aspect_ratios.append(w / (h + 1e-6))
            # 5) Centroid distance to image center
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                d = ((cx - W / 2) ** 2 + (cy - H / 2) ** 0.5)
                centroid_dists.append(d)
            # 6) Max Feret diameter
            pts = cnt.reshape(-1, 2)
            max_d = 0.0
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    dist = np.linalg.norm(pts[i] - pts[j])
                    if dist > max_d:
                        max_d = dist
            feret_diams.append(max_d)

        # İstatistikleri hesapla
 
        def stats(arr):
            return {
                'mean': float(np.mean(arr)) if arr else 0.0,
                'min': float(np.min(arr)) if arr else 0.0,
                'max': float(np.max(arr)) if arr else 0.0,
                'std': float(np.std(arr)) if arr else 0.0
            }

        results = {
            'cellCount': len(contours),
            'cellDensity': len(contours) / (W * H),
            'meanPerimeter': stats(perimeters)['mean'],
            'meanFeret': stats(feret_diams)['mean'],
            'meanEccentricity': stats(eccentricities)['mean'],
            'meanAspectRatio': stats(aspect_ratios)['mean'],
            'meanCentroidDist': stats(centroid_dists)['mean'],
            'meanBBoxWidth': stats(widths)['mean'],
            'meanBBoxHeight': stats(heights)['mean'],
            'histograms': {
            'cellCount': {
            'labels': ['0-5', '5-10', '10-15', '15-20'],
            'data': [len([x for x in widths if x < 5]), 
                    len([x for x in widths if 5 <= x < 10]),
                    len([x for x in widths if 10 <= x < 15]),
                    len([x for x in widths if x >= 15])]
                }
            }
        }

        # Görüntüyü encode et
        _, buffer = cv2.imencode('.png', processed_img)
        img_b64 = base64.b64encode(buffer).decode()

        return jsonify({
            'processedImage': f'data:image/png;base64,{img_b64}',
            'analysisResults': results
        })

    elif analysis_type == 'branch':


        
        def dice_coef(y_true, y_pred):
            intersection = tf.reduce_sum(y_true * y_pred)
            union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
            return (2.0 * intersection + 1e-5) / (union + 1e-5)

        def iou_score(y_true, y_pred):
            intersection = tf.reduce_sum(y_true * y_pred)
            union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
            return (intersection + 1e-5) / (union + 1e-5)
        def segment_and_draw_branches(
            orig_img: np.ndarray,
            mask_path: str,
            branch_model_path: str,
            best_model_path: str,
            threshold: int = 127
        ) -> tuple:
            """
            Segment branches and draw them on the original image.
            Returns the overlay image and a list of branches (pixel coordinates).
            """
            with tf.keras.utils.custom_object_scope({'dice_coef': dice_coef, 'iou_score': iou_score}):
                branch_model = tf.keras.models.load_model(branch_model_path)
                best_model = tf.keras.models.load_model(best_model_path)

            orig = orig_img.copy()
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            inp = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            inp = cv2.resize(inp, (256, 256)).astype(np.float32) / 255.0
            inp_tensor = inp[None, ..., None]

            pred_branch = branch_model.predict(inp_tensor)[0, ..., 0]
            pred_best = best_model.predict(inp_tensor)[0, ..., 0]

            bin_branch = (pred_branch > 0.5).astype(np.uint8)
            bin_best = (pred_best > 0.5).astype(np.uint8)
            combined = cv2.bitwise_and(bin_branch, bin_best)

            mask_resized = cv2.resize(mask, (256, 256)) // 255
            filtered = combined * mask_resized

            _, bw = cv2.threshold((filtered * 255).astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
            skel = skeletonize(bw // 255).astype(np.uint8)

            endpoints = []
            h, w = skel.shape
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if skel[y, x]:
                        neigh = skel[y - 1:y + 2, x - 1:x + 2]
                        if neigh.sum() == 2:
                            endpoints.append((x, y))

            visited = set()

            def dfs(x, y):
                stack = [(x, y)]
                branch = []
                while stack:
                    cx, cy = stack.pop()
                    if (cx, cy) not in visited:
                        visited.add((cx, cy))
                        branch.append((cx, cy))
                        for nx in range(cx - 1, cx + 2):
                            for ny in range(cy - 1, cy + 2):
                                if 0 <= nx < w and 0 <= ny < h and skel[ny, nx]:
                                    stack.append((nx, ny))
                return branch

            branches = []
            for y in range(h):
                for x in range(w):
                    if skel[y, x] and (x, y) not in visited:
                        b = dfs(x, y)
                        if len(b) > 1:
                            branches.append(b)

            branches = [b for b in branches if any(p in endpoints for p in b)]
            def generate_bgr_colors(n):
                from matplotlib.colors import hsv_to_rgb 
                hues = np.linspace(0, 1, n, endpoint=False)
                colors = []
                for h in hues:
                    rgb = hsv_to_rgb([h, 1, 1])       # [r,g,b] float64 0–1
                    rgb = (rgb * 255).astype(int)     # np.int64
                    # OpenCV BGR formatına çevir ve saf Python int'e dönüştür
                    bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
                    colors.append(bgr)
                return colors
            colors = generate_bgr_colors(len(branches))
            overlay = orig.copy()
            sx, sy = orig.shape[1] / 256, orig.shape[0] / 256

            for idx, branch in enumerate(branches):
                col = colors[idx]
                pts = [(int(x * sx), int(y * sy)) for x, y in branch]
                for i in range(len(pts) - 1):
                    cv2.line(overlay, pts[i], pts[i + 1], col, 12)

            return overlay, branches

        overlay, branches = segment_and_draw_branches(
            orig_img=img,
            mask_path=r'C:\Users\safak\Desktop\pc12_dataset\safak\backend\branch\023.png',
            branch_model_path=r'C:\Users\safak\Desktop\pc12_dataset\safak\backend\best_branch_model.hdf5',
            best_model_path=r'C:\Users\safak\Desktop\pc12_dataset\safak\backend\best_model.hdf5',
            threshold=threshold
        )

        lengths = []
        sx, sy = img.shape[1] / 256, img.shape[0] / 256
        for branch in branches:
            pts = [(x * sx, y * sy) for x, y in branch]
            length = sum(
                ((pts[i][0] - pts[i - 1][0]) ** 2 + (pts[i][1] - pts[i - 1][1]) ** 2) ** 0.5
                for i in range(1, len(pts))
            )
            lengths.append(length)

        total = len(lengths)
        avg_len = float(np.mean(lengths)) if lengths else 0.0
        min_len = float(np.min(lengths)) if lengths else 0.0
        max_len = float(np.max(lengths)) if lengths else 0.0
        std_len = float(np.std(lengths)) if lengths else 0.0
        
        labels = [f"Branch {i+1}" for i in range(len(lengths))]

# data: Her branch'ın uzunluğu
        data = [round(length, 2) for length in lengths]  # lengths verisinin her bir elemanını 2 ondalık basamağa yuvarla
        _, buffer = cv2.imencode('.png', overlay)
        img_b64 = base64.b64encode(buffer).decode()
        
        return jsonify({
            'processedImage': f'data:image/png;base64,{img_b64}',
            'analysisResults': {
                'totalBranches': total,
                'averageLength': avg_len,
                'minLength': min_len,
                'maxLength': max_len,
                'stdLength': std_len,
                'histograms': {
                'branchLength': {
                'labels':labels,
                'data': data
                    }
                }
            }
        })

    elif analysis_type == 'cellArea':
        # Hücre alan analizi için fonksiyonlar
        def dice_coef(y_true, y_pred):
            intersection = tf.reduce_sum(y_true * y_pred)
            union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
            return (2.0 * intersection + 1e-5) / (union + 1e-5)

        def iou_score(y_true, y_pred):
            intersection = tf.reduce_sum(y_true * y_pred)
            union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
            return (intersection + 1e-5) / (union + 1e-5)

        def load_cell_models():
            with tf.keras.utils.custom_object_scope({'dice_coef': dice_coef, 'iou_score': iou_score}):
                cell_model = tf.keras.models.load_model(r"C:\Users\safak\Desktop\pc12_dataset\safak\backend\best_cell_model.hdf5")
                best_model = tf.keras.models.load_model(r"C:\Users\safak\Desktop\pc12_dataset\safak\backend\best_model.hdf5")
            return cell_model, best_model

        def process_cell_areas(img, threshold):
            # Modelleri yükle
            cell_model, best_model = load_cell_models()
            
            # Görüntüyü ön işle
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (256, 256))
            inp = np.expand_dims(resized, axis=(0, -1))

            # Tahminleri al
            pred_cell = cell_model.predict(inp)[0]
            pred_best = best_model.predict(inp)[0]

            # Maskeleri birleştir
            src1 = (pred_cell[:, :, 1] * 255).astype(np.uint8)
            src2 = (pred_best[:, :, 1] * 255).astype(np.uint8)
            
            if src1.shape != src2.shape:
                src2 = cv2.resize(src2, (src1.shape[1], src1.shape[0]))
            
            combined = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)

            # İkili görüntü oluştur
            _, binary = cv2.threshold(combined, threshold, 255, cv2.THRESH_BINARY)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

            # Bağlı bileşen analizi
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
            
            # Görselleştirme
            output = cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2RGB)
            areas = []
            
            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                areas.append(area)
                
                center = (int(x + w/2), int(y + h/2))
                radius = int(np.sqrt(area/np.pi))
                color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
                
                cv2.circle(output, center, radius, color, 2)
                cv2.putText(output, f"{area}", (center[0]-20, center[1]+5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

            return output, areas

        # Hücre alanlarını işle
        processed_img, areas = process_cell_areas(img, threshold)


        # Sonucu base64'e çevir
        processed_img = cv2.resize(processed_img, (img.shape[1], img.shape[0]))
        _, buffer = cv2.imencode('.png', processed_img)
 
        processed_img = f'data:image/png;base64,{base64.b64encode(buffer).decode()}'

        return jsonify({
            'processedImage': processed_img,
            'analysisResults': {
                'totalCells': len(areas),
                'averageArea': np.mean(areas).item(),
                'minArea': np.min(areas).item(),
                'maxArea': np.max(areas).item(),
                'std': np.std(areas).item(),
                'histograms': {
                'cellArea': {
                'labels': ['0-100', '100-200', '200-300', '300-400'],
                'data': [len([x for x in areas if x < 100]),
                        len([x for x in areas if 100 <= x < 200]),
                        len([x for x in areas if 200 <= x < 300]),
                        len([x for x in areas if x >= 300])]
                    }
                } 
            }
        })
    
    elif analysis_type == 'branchLength':
        from scipy import stats as scipy_stats
        from matplotlib.colors import hsv_to_rgb
        import random

        def dice_coef(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            intersection = tf.reduce_sum(y_true * y_pred)
            return (2. * intersection + 1e-5) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-5)

        def iou_score(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            intersection = tf.reduce_sum(y_true * y_pred)
            union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
            return (intersection + 1e-5) / (union + 1e-5)

        def process_branch_length(img, threshold):
            # Model yolları
            BRANCH_MODEL_PATH = r"C:\Users\safak\Desktop\pc12_dataset\safak\backend\best_branch_model.hdf5"
            BEST_MODEL_PATH = r"C:\Users\safak\Desktop\pc12_dataset\safak\backend\best_model.hdf5"
            MASK_PATH = r"C:\Users\safak\Desktop\pc12_dataset\safak\backend\branch\023.png"

            # Modelleri yükle
            with tf.keras.utils.custom_object_scope({'dice_coef': dice_coef, 'iou_score': iou_score}):
                branch_model = tf.keras.models.load_model(BRANCH_MODEL_PATH)
                best_model = tf.keras.models.load_model(BEST_MODEL_PATH)

            # Segmentasyon işlemleri
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (256, 256)).astype(np.float32) / 255.0
            inp_tensor = resized[None, ..., None]

            pred_branch = branch_model.predict(inp_tensor)[0, ..., 0]
            pred_best = best_model.predict(inp_tensor)[0, ..., 0]

            # Maskeyi yükle ve filtrele
            mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
            mask_resized = cv2.resize(mask, (256, 256)) // 255

            combined = cv2.bitwise_and(
                (pred_branch > 0.5).astype(np.uint8),
                (pred_best > 0.5).astype(np.uint8)
            ) * mask_resized

            # Skeletonize
            _, bw = cv2.threshold((combined * 255).astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
            skel = skeletonize(bw // 255).astype(np.uint8)

            # Branch tespiti (DFS ile)
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

            for y in range(h):
                for x in range(w):
                    if skel[y, x] and (x, y) not in visited:
                        b = dfs(x, y)
                        if len(b) > 1:
                            branches.append(b)

            # Renk paleti oluştur
            def generate_colors(n):
                return [tuple((np.array(hsv_to_rgb([i/n, 1, 1]))*255).astype(int)) for i in range(n)]

            colors = generate_colors(len(branches)) if branches else []

            # Uzunluk hesaplama ve görselleştirme
            overlay = img.copy()
            lengths = []
            scale_x = img.shape[1] / 256
            scale_y = img.shape[0] / 256

            for idx, branch in enumerate(branches):
                # Branch'ı çiz
                
                pts = [(int(x*scale_x), int(y*scale_y)) for x, y in branch]
                
                # Branch çizgileri
                for i in range(len(pts)-1):
                    cv2.line(overlay, pts[i], pts[i+1], (0,0,255), 5)
                
                # Uzunluk hesapla
                length = sum(np.linalg.norm(np.array(pts[i])-np.array(pts[i-1])) 
                        for i in range(1, len(pts)))
                lengths.append(length)

                # Branch üzerine yazı yaz (beyaz arka planlı)
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
            def calculate_stats(lengths):
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
                    'lengthSkewness': float(scipy_stats.skew(arr)),
                    'lengthKurtosis': float(scipy_stats.kurtosis(arr))
                }

            stats = calculate_stats(lengths)
            stats.update({
                'totalBranches': len(lengths),
                'averageLength': np.mean(lengths).item() if lengths else 0.0,
                'minLength': np.min(lengths).item() if lengths else 0.0,
                'maxLength': np.max(lengths).item() if lengths else 0.0,
                'stdLength': np.std(lengths).item() if lengths else 0.0
            })
            # Histogram verisi (0'dan 5000'e kadar, 100'er 100'er artarak)
            hist_bins = np.arange(0, 5000 + 100, 100)  # 0, 100, 200, ..., 5000
            hist_values, _ = np.histogram(lengths, bins=hist_bins)
            labels = [f"{hist_bins[i]}-{hist_bins[i+1]}" for i in range(len(hist_bins)-1)]

            # JSON çıktısı
            _, buffer = cv2.imencode('.png', overlay)
            img_b64 = base64.b64encode(buffer).decode()

            return jsonify({
                'processedImage': f'data:image/png;base64,{img_b64}',
                'analysisResults': {
                    **stats,
                    'histograms': {
                        'branchLength': {
                            'labels': labels,
                            'data': hist_values.tolist()
                        }
                    }
                }
            })

        # İşlemi başlat
        return process_branch_length(img, threshold)
    
    elif analysis_type == 'angles':
        def dice_coef(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            intersection = tf.reduce_sum(y_true * y_pred)
            return (2. * intersection + 1e-5) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-5)

        def iou_score(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            intersection = tf.reduce_sum(y_true * y_pred)
            union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
            return (intersection + 1e-5) / (union + 1e-5)


        def segment_and_draw_branches(orig_img: np.ndarray,
                                    mask_img: np.ndarray,
                                    branch_model_path: str,
                                    best_model_path: str,
                                    threshold: int = 127) -> tuple:
            """
            Segment branches and draw them on the original image.
            Returns the overlay image and a list of branches (pixel coordinates).
            """
            with tf.keras.utils.custom_object_scope({'dice_coef': dice_coef, 'iou_score': iou_score}):
                branch_model = tf.keras.models.load_model(branch_model_path)
                best_model = tf.keras.models.load_model(best_model_path)

            orig = orig_img.copy()
            gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            inp = cv2.resize(gray, (256, 256)).astype(np.float32) / 255.0
            inp_tensor = inp[None, ..., None]

            pred_branch = branch_model.predict(inp_tensor)[0, ..., 0]
            pred_best = best_model.predict(inp_tensor)[0, ..., 0]

            bin_branch = (pred_branch > 0.5).astype(np.uint8)
            bin_best = (pred_best > 0.5).astype(np.uint8)
            combined = cv2.bitwise_and(bin_branch, bin_best)

            mask_resized = cv2.resize(mask_img, (256, 256), interpolation=cv2.INTER_NEAREST) // 255
            filtered = combined * mask_resized

            _, bw = cv2.threshold((filtered * 255).astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
            skel = skeletonize(bw // 255).astype(np.uint8)

            # Find branch segments
            endpoints = []
            h, w = skel.shape
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if skel[y, x]:
                        neigh = skel[y - 1:y + 2, x - 1:x + 2]
                        if neigh.sum() == 2:
                            endpoints.append((x, y))
            visited = set()
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

            branches = []
            for y in range(h):
                for x in range(w):
                    if skel[y, x] and (x, y) not in visited:
                        b = dfs(x, y)
                        if len(b) > 1 and any(p in endpoints for p in b):
                            branches.append(b)


            # Compute angles and draw
            from matplotlib.colors import hsv_to_rgb 
            overlay = orig.copy()
            scale_x = orig.shape[1] / 256
            scale_y = orig.shape[0] / 256
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
                        angles.append({'points': [p1r, p2r], 'angle': float(round(angle,2)), 'distance': float(round(dist,2))})
                        # draw line and label
                        color = tuple(int(c*255) for c in hsv_to_rgb([angle/360,1,1]))
                        cv2.line(overlay, p1r, p2r, color, 2)
                        cv2.putText(overlay, f"{round(angle,1)} deg", ((p1r[0]+p2r[0])//2, (p1r[1]+p2r[1])//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


            # Statistics
            values = [a['angle'] for a in angles]
            def resultant_vector_length(angles):
                radians = np.deg2rad(angles)
                sum_cos = np.sum(np.cos(radians))
                sum_sin = np.sum(np.sin(radians))
                R = np.sqrt(sum_cos**2 + sum_sin**2) / len(angles)
                return float(R)

            def angular_entropy(angles, bins=36):
                hist, _ = np.histogram(angles, bins=bins, range=(0, 360), density=True)
                hist = hist[hist > 0]  # sıfır olmayanlar
                entropy = -np.sum(hist * np.log(hist))
                return float(entropy)
            from scipy.stats import skew
            def angle_skewness(angles):
                return float(skew(angles))

            def fractal_dimension(Z):
                """Box-counting yöntemiyle fraktal boyut hesabı."""
                assert(len(Z.shape) == 2)

                # Binary olsun
                Z = (Z > 0)
                if np.sum(Z) == 0:
                    return 0.0

                def boxcount(Z, k):
                    S = np.add.reduceat(
                        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                        np.arange(0, Z.shape[1], k), axis=1)

                    return len(np.where((S > 0) & (S < k*k))[0])

                # Kutu boyutları (ölçekler)
                p = min(Z.shape)
                n = 2**np.floor(np.log2(p))
                sizes = 2**np.arange(int(np.log2(n)), 1, -1)

                counts = []
                for size in sizes:
                    counts.append(boxcount(Z, int(size)))

                coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
                return float(-coeffs[0])
            def max_branch_order(branches, endpoints):
                    """Dal listesinden maksimum branch sırası."""
                    max_depth = 0
                    for branch in branches:
                        endpoint_count = sum(1 for p in branch if p in endpoints)
                        if endpoint_count >= 2:  # gerçek bir dallanma
                            depth = len(branch)
                            if depth > max_depth:
                                max_depth = depth
                    return int(max_depth)   
            def node_degree_mean(skel):
                """İskelet üzerinden düğüm derecesi ortalaması."""
                h, w = skel.shape
                degrees = []
                for y in range(1, h-1):
                    for x in range(1, w-1):
                        if skel[y, x]:
                            neigh = skel[y-1:y+2, x-1:x+2]
                            degree = np.sum(neigh) - 1  # kendisini çıkar
                            degrees.append(degree)
                if degrees:
                    return float(np.mean(degrees))
                else:
                    return 0.0   
            from scipy.spatial import ConvexHull
            def convex_hull_compactness(branches):
                """Branch noktalarından convex hull yoğunluk ölçümü."""
                points = []
                for branch in branches:
                    points.extend(branch)
                points = np.array(points)
                if len(points) < 3:
                    return 0.0
                hull = ConvexHull(points)
                perimeter = hull.area  # 2D için "area" çevre olur
                area = hull.volume     # 2D için "volume" alan olur
                if area == 0:
                    return 0.0
                compactness = perimeter**2 / (4 * np.pi * area)
                return float(compactness)     
            stats = {
                'average': float(np.mean(values)) if values else 0.0,
                'min': float(np.min(values)) if values else 0.0,
                'max': float(np.max(values)) if values else 0.0,
                'std': float(np.std(values)) if values else 0.0,
                'resultantVectorLength': resultant_vector_length(values) if values else 0.0,
                'angularEntropy': angular_entropy(values) if values else 0.0,
                'angleSkewness': angle_skewness(values) if values else 0.0,
                'fractalDimension': fractal_dimension(skel) if np.sum(skel) else 0.0,
                'maxBranchOrder': max_branch_order(branches, endpoints) if branches else 0,
                'nodeDegreeMean': node_degree_mean(skel) if np.sum(skel) else 0.0,
                'convexHullCompactness': convex_hull_compactness(branches) if branches else 0.0,
                'angles': angles
            }
            return overlay, stats
        MASK_PATH = r"C:\Users\safak\Desktop\pc12_dataset\safak\backend\branch\023.png"
        BRANCH_MODEL_PATH = r"C:\Users\safak\Desktop\pc12_dataset\safak\backend\best_branch_model.hdf5"
        BEST_MODEL_PATH = r"C:\Users\safak\Desktop\pc12_dataset\safak\backend\best_model.hdf5"
            # Load mask image
        mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
            # Compute angles and draw
        overlay, stats = segment_and_draw_branches(
                orig_img=img,
                mask_img=mask,
                branch_model_path=BRANCH_MODEL_PATH,
                best_model_path=BEST_MODEL_PATH,
                threshold=threshold
            )
        _, buffer = cv2.imencode('.png', overlay)
        img_b64 = base64.b64encode(buffer).decode()
        # Angles analizi kısmına ekle (yaklaşık 500. satır)
        angles = stats.get('angles', [])  # Bu zaten angle objelerinin listesi
        if angles:  # Eğer açı verisi varsa
            angle_values = [a['angle'] for a in angles]  # Sadece angle değerlerini al
            angle_bins = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
            angle_hist, _ = np.histogram(angle_values, bins=angle_bins)
            
            hist_data = {
                'angles': {
                    'labels': ['0-30', '30-60', '60-90', '90-120', '120-150', 
                            '150-180', '180-210', '210-240', '240-270', 
                            '270-300', '300-330', '330-360'],
                    'data': angle_hist.tolist()
                }
            }
        else:  # Eğer açı verisi yoksa
            hist_data = {
                'angles': {
                    'labels': [],
                    'data': []
                }
            }

        return jsonify({
            'processedImage': f'data:image/png;base64,{img_b64}',
            'analysisResults': {
                **stats,  # Mevcut istatistikleri koru
                'histograms': hist_data  # Histogram verilerini ekle
            }
        })

        
    else:
        processed_img = img
    
    # Base64'e çevir
    _, buffer = cv2.imencode('.png', processed_img)
    return jsonify({
        'processedImage': f'data:image/png;base64,{base64.b64encode(buffer).decode()}'
    })

# Eksik olan kısım burada!
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # ✅ Eklendi