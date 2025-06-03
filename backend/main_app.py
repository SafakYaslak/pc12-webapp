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
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
from skimage.measure import regionprops, label
import networkx as nx
import math as math
def boxcount(Z, k):
                    S = np.add.reduceat(
                            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                            np.arange(0, Z.shape[1], k), axis=1)
                    return np.count_nonzero(S)

def compute_fractal_dimension(Z):
                    """
                    Box-counting algorithm to approximate the fractal dimension of a 2D binary image Z.
                    """
                    assert len(Z.shape) == 2, "Input image must be 2D"
                    Z = (Z > 0)
                    p = min(Z.shape)
                    n = 2**np.floor(np.log2(p))
                    sizes = 2**np.arange(np.floor(np.log2(n)), 1, -1)
                    counts = []
                    for size in sizes:
                        counts.append(boxcount(Z, int(size)))
                    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
                    return -coeffs[0]

def build_skeleton_graph(skeleton):
                    """
                    Build a graph from a binary skeleton image using 8-connected neighbors.
                    """
                    G = nx.Graph()
                    h, w = skeleton.shape
                    indices = np.argwhere(skeleton > 0)
                    for y, x in indices:
                        G.add_node((x, y))
                    for y, x in indices:
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                nx_, ny_ = x + dx, y + dy
                                if 0 <= ny_ < h and 0 <= nx_ < w and skeleton[ny_, nx_]:
                                    G.add_edge((x, y), (nx_, ny_))
                    return G

def compute_max_branch_order(skeleton):
                    """
                    Maximum branch order is approximated from the skeleton graph:
                    For each node with degree >2, we consider (degree - 2) as local branch order.
                    Returns the maximum such value.
                    """
                    G = build_skeleton_graph(skeleton)
                    max_order = 0
                    for node in G.nodes():
                        d = G.degree[node]
                        if d > 2:
                            max_order = max(max_order, d - 2)
                    return max_order

def compute_average_node_degree(skeleton):
                    """
                    Returns the average node degree over the skeleton graph.
                    """
                    G = build_skeleton_graph(skeleton)
                    if len(G.nodes()) == 0:
                        return 0.0
                    degrees = [d for n, d in G.degree()]
                    return float(np.mean(degrees))

def compute_convex_hull_compactness(skeleton):
                    """
                    Calculates compactness = (4π * area) / (perimeter^2)
                    where the convex hull is computed on the skeleton points.
                    """
                    points = np.column_stack(np.where(skeleton > 0))
                    if points.shape[0] < 3:
                        return 0.0
                    # Swap order from (row, col) to (x, y)
                    points = points[:, ::-1]
                    hull = ConvexHull(points)
                    area = hull.volume     # In 2D, volume is the area.
                    perimeter = hull.area  # In 2D, area is the perimeter.
                    if perimeter == 0:
                        return 0.0
                    return (4 * math.pi * area) / (perimeter ** 2)
def detect_cell_centers(cell_mask, min_area=50):
    """
    Hücre merkezlerini tespit eder
    """
    # Connected components ile hücreleri ayır
    num_labels, labels = cv2.connectedComponents(cell_mask.astype(np.uint8))
    cell_centers = []
    
    for i in range(1, num_labels):  # 0 arka plan
        component_mask = (labels == i).astype(np.uint8)
        area = np.sum(component_mask)
        
        if area > min_area:
            # Merkez hesapla
            moments = cv2.moments(component_mask)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                cell_centers.append((cx, cy))
    
    return cell_centers

def find_skeleton_endpoints_and_junctions(skeleton):
    """
    Skeleton üzerindeki uç noktaları ve kavşak noktalarını bulur
    """
    # 3x3 kernel ile komşu sayısını hesapla
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    # Skeleton noktalarında komşu sayısını hesapla
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    neighbor_count = neighbor_count * (skeleton > 0)
    
    # Uç noktalar: tam 1 komşu (10 + 1 = 11)
    endpoints = np.where(neighbor_count == 11)
    endpoints = list(zip(endpoints[1], endpoints[0]))  # (x, y) format
    
    # Kavşak noktaları: 3 veya daha fazla komşu (10 + 3+ = 13+)
    junctions = np.where(neighbor_count >= 13)
    junctions = list(zip(junctions[1], junctions[0]))  # (x, y) format
    
    return endpoints, junctions

def trace_branch_from_point(skeleton, start_point, visited, max_length=1000):
    """
    Belirli bir noktadan başlayarak branch'i takip eder
    """
    path = [start_point]
    current = start_point
    visited.add(current)
    
    # 8-bağlantı için komşuluk
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    while len(path) < max_length:
        found_next = False
        
        # Mevcut noktanın komşularını kontrol et
        for dx, dy in directions:
            next_x, next_y = current[0] + dx, current[1] + dy
            
            # Sınırları kontrol et
            if (0 <= next_x < skeleton.shape[1] and 
                0 <= next_y < skeleton.shape[0] and
                skeleton[next_y, next_x] > 0 and
                (next_x, next_y) not in visited):
                
                # Komşu sayısını kontrol et (kavşak kontrolü)
                neighbor_count = 0
                for dx2, dy2 in directions:
                    nx, ny = next_x + dx2, next_y + dy2
                    if (0 <= nx < skeleton.shape[1] and 
                        0 <= ny < skeleton.shape[0] and
                        skeleton[ny, nx] > 0):
                        neighbor_count += 1
                
                # Eğer kavşak noktasıysa (3+ komşu) ve path başlangıcı değilse dur
                if neighbor_count >= 3 and len(path) > 1:
                    path.append((next_x, next_y))
                    return path
                
                current = (next_x, next_y)
                path.append(current)
                visited.add(current)
                found_next = True
                break
        
        if not found_next:
            break
    
    return path

def assign_branches_to_cells(branches, cell_centers, max_distance=30):
    """
    Branch'leri en yakın hücrelere atar
    """
    if not cell_centers or not branches:
        return []
    
    cell_centers_array = np.array(cell_centers)
    branch_assignments = []
    
    for i, branch in enumerate(branches):
        if len(branch) < 2:
            continue
            
        start_point = np.array(branch[0])
        
        # En yakın hücre merkezini bul
        distances = cdist([start_point], cell_centers_array)[0]
        closest_cell_idx = np.argmin(distances)
        closest_distance = float(distances[closest_cell_idx])  # Convert to native Python float
        
        # Eğer yeterince yakınsa ata
        if closest_distance <= max_distance:
            branch_assignments.append({
                'branch_id': int(i + 1),  # Convert to native Python int
                'branch_path': [(int(x), int(y)) for x, y in branch],  # Convert coordinates to native Python ints
                'start_cell': int(closest_cell_idx),  # Convert to native Python int
                'start_point': (int(branch[0][0]), int(branch[0][1])),  # Convert to native Python ints
                'end_point': (int(branch[-1][0]), int(branch[-1][1])),  # Convert to native Python ints
                'length': float(calculate_branch_length(branch)),  # Convert to native Python float
                'distance_to_cell': float(closest_distance)  # Convert to native Python float
            })
    
    return branch_assignments

def calculate_branch_length(branch_path):
    """
    Branch uzunluğunu hesaplar
    """
    if len(branch_path) < 2:
        return 0
    
    total_length = 0
    for i in range(len(branch_path) - 1):
        p1 = np.array(branch_path[i])
        p2 = np.array(branch_path[i + 1])
        total_length += np.linalg.norm(p2 - p1)
    
    return total_length

def separate_merged_branches(skeleton, cell_centers, min_branch_length=20):
    """
    Birleşen branch'leri ayırır ve hücre merkezlerine atar
    """
    # Uç noktaları ve kavşakları bul
    endpoints, junctions = find_skeleton_endpoints_and_junctions(skeleton)
    
    # Ziyaret edilen noktaları takip et
    visited = set()
    all_branches = []
    
    # Önce uç noktalardan başla
    for endpoint in endpoints:
        if endpoint not in visited:
            branch = trace_branch_from_point(skeleton, endpoint, visited)
            if len(branch) >= min_branch_length:
                all_branches.append(branch)
    
    # Sonra kavşak noktalarından başla
    for junction in junctions:
        if junction not in visited:
            # Kavşak noktasından her yöne branch trace et
            directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            for dx, dy in directions:
                start_point = (junction[0] + dx, junction[1] + dy)
                if (0 <= start_point[0] < skeleton.shape[1] and 
                    0 <= start_point[1] < skeleton.shape[0] and
                    skeleton[start_point[1], start_point[0]] > 0 and
                    start_point not in visited):
                    
                    branch = trace_branch_from_point(skeleton, start_point, visited)
                    if len(branch) >= min_branch_length:
                        all_branches.append(branch)
    
    # Branch'leri hücrelere ata
    branch_assignments = assign_branches_to_cells(all_branches, cell_centers)
    
    return branch_assignments

def visualize_improved_branches(original_img, branch_assignments, cell_centers, mask_branches=None):
    """
    Geliştirilmiş branch görselleştirmesi:
    - Sadece model çıktısı branchler: Kırmızı (kalın)
    - Sadece maskeden gelen branchler: Mavi (ince)
    - Üst üste binen branchler: Yeşil (orta kalınlık)
    """
    overlay_img = original_img.copy()
    
    # Önce hücre merkezlerini çiz (sarı daireler)
    for i, (cx, cy) in enumerate(cell_centers):
        cv2.circle(overlay_img, (cx, cy), 30 , (255, 0, 0), -1)  # Sarı dolu daire
        cv2.putText(overlay_img, f"C{i+1}", (cx-10, cy-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 4)
 
    
    # Model branch'lerinin mask'ini oluştur
    h, w = original_img.shape[:2]
    model_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Model branch'lerini mask'e çiz
    for assignment in branch_assignments:
        branch_path = assignment['branch_path']
        if len(branch_path) > 1:
            for i in range(len(branch_path) - 1):
                cv2.line(model_mask, branch_path[i], branch_path[i+1], 1, 8)
    
    # Maske branch'lerinin mask'ini oluştur
    mask_mask = np.zeros((h, w), dtype=np.uint8)
    
    if mask_branches is not None:
        for branch_path in mask_branches:
            if len(branch_path) > 1:
                for i in range(len(branch_path) - 1):
                    cv2.line(mask_mask, branch_path[i], branch_path[i+1], 1, 3)
    
    # Kesişim mask'ini oluştur
    intersection_mask = cv2.bitwise_and(model_mask, mask_mask)
    
    # Sadece model ve sadece mask alanlarını bul
    only_model_mask = cv2.bitwise_and(model_mask, cv2.bitwise_not(intersection_mask))
    only_mask_mask = cv2.bitwise_and(mask_mask, cv2.bitwise_not(intersection_mask))
    
    # Renkleri uygula
    overlay_img[only_model_mask > 0] = (0, 0, 255)    # Kırmızı (sadece model)
    #overlay_img[only_mask_mask > 0] = (255, 0, 0)     # Mavi (sadece mask)
    overlay_img[intersection_mask > 0] = (0, 255, 0)   # Yeşil (kesişim)
    
    total_branches = 0  # Toplam branch sayısını takip et
    
    # Branch numaralarını yaz (model çıktıları için)
    for assignment in branch_assignments:
        total_branches += 1
        branch_path = assignment['branch_path']
        start_point = assignment['start_point']
        end_point = assignment['end_point']
        
        # Branch numarasını ortaya yaz
        mid_point = (
            (start_point[0] + end_point[0]) // 2,
            (start_point[1] + end_point[1]) // 2
        )
        # Beyaz arka planlı metin
        # cv2.putText(overlay_img, f"B{total_branches}", mid_point, 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
        # cv2.putText(overlay_img, f"B{total_branches}", mid_point, 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
    
    # Maskeden gelen branch'ler için numara vermeye devam et
    if mask_branches is not None:
        for branch_path in mask_branches:
            if len(branch_path) > 1:
                total_branches += 1
                # Branch numarasını ortaya yaz
                mid_point = (
                    (branch_path[0][0] + branch_path[-1][0]) // 2,
                    (branch_path[0][1] + branch_path[-1][1]) // 2
                )
                # # Beyaz arka planlı metin
                # cv2.putText(overlay_img, f"B{total_branches}", mid_point, 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
                # cv2.putText(overlay_img, f"B{total_branches}", mid_point, 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
    
    return overlay_img

# ANA FONKSİYON - Mevcut kodunuzun 'branch' kısmına entegre edilecek
def improved_branch_analysis(original_img, img_gray_resized_norm, mask_resized_binary, 
                           branch_seg_model, common_best_model, threshold_param):
    """
    Geliştirilmiş branch analizi
    """
    # 1. Skeleton oluştur (mevcut fonksiyonunuzu kullan)
    skeleton_256 = _get_branch_segmentation_skeleton(
        img_gray_resized_norm, mask_resized_binary, 
        branch_seg_model, common_best_model, threshold_param
    )
    
    # 2. Hücre merkezlerini tespit et
    # Not: Burada cell mask'i de kullanmanız gerekebilir
    # Şimdilik branch mask'inden hücre merkezlerini tahmin ediyoruz
    cell_centers_256 = detect_cell_centers(mask_resized_binary)
    
    # 3. Geliştirilmiş branch separation
    branch_assignments = separate_merged_branches(skeleton_256, cell_centers_256)
    
    # 4. Koordinatları orijinal boyuta ölçekle
    scale_x = original_img.shape[1] / 256.0
    scale_y = original_img.shape[0] / 256.0
    
    # Ölçeklenmiş branch assignments
    scaled_assignments = []
    for assignment in branch_assignments:
        scaled_path = [(int(x * scale_x), int(y * scale_y)) 
                      for x, y in assignment['branch_path']]
        scaled_assignment = assignment.copy()
        scaled_assignment['branch_path'] = scaled_path
        scaled_assignment['start_point'] = (int(assignment['start_point'][0] * scale_x), 
                                          int(assignment['start_point'][1] * scale_y))
        scaled_assignment['end_point'] = (int(assignment['end_point'][0] * scale_x), 
                                        int(assignment['end_point'][1] * scale_y))
        scaled_assignment['length'] = calculate_branch_length(scaled_path)
        scaled_assignments.append(scaled_assignment)
    
    # Hücre merkezlerini de ölçekle
    scaled_cell_centers = [(int(x * scale_x), int(y * scale_y)) 
                          for x, y in cell_centers_256]
    
    # 5. Görselleştirme
    overlay_img = visualize_improved_branches(original_img, scaled_assignments, scaled_cell_centers)
    
    # 6. İstatistikler
    branch_lengths = [assignment['length'] for assignment in scaled_assignments]
    
    results = {
        'totalBranches': len(scaled_assignments),
        'cellsDetected': len(scaled_cell_centers),
        'averageLength': np.mean(branch_lengths) if branch_lengths else 0,
        'minLength': np.min(branch_lengths) if branch_lengths else 0,
        'maxLength': np.max(branch_lengths) if branch_lengths else 0,
        'stdLength': np.std(branch_lengths) if branch_lengths else 0,
        'branchDetails': [
            {
                'id': assignment['branch_id'],
                'length': assignment['length'],
                'startCell': assignment['start_cell'],
                'distanceToCell': assignment['distance_to_cell']
            }
            for assignment in scaled_assignments
        ],
        'histograms': {
            'branchLength': {
                'labels': [f"Branch {assignment['branch_id']}" for assignment in scaled_assignments],
                'data': [round(assignment['length'], 2) for assignment in scaled_assignments]
            }
        }
    }
    
    return overlay_img, results
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
            # Dinamik maske yolu oluştur
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
            # Dinamik maske yolu oluştur
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

            # ===== YENİ GELİŞTİRİLMİŞ ALGORİTMA =====
            # Cell mask'i de yükle (branch mask'i yerine cell mask kullanabilirsiniz)
            cell_mask_filename = f"{image_name.split('.')[0]}.png"
            cell_mask_path = os.path.join(CELL_MASKS_PATH, cell_mask_filename)
            
            cell_centers_256 = []
            if os.path.exists(cell_mask_path):
                cell_mask = cv2.imread(cell_mask_path, cv2.IMREAD_GRAYSCALE)
                if cell_mask is not None:
                    cell_mask_resized = cv2.resize(cell_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                    cell_mask_binary = (cell_mask_resized > 127).astype(np.uint8)
                    cell_centers_256 = detect_cell_centers(cell_mask_binary)
            
            # Eğer cell mask yoksa, branch mask'inden tahmin et
            if not cell_centers_256:
                cell_centers_256 = detect_cell_centers(mask_resized_binary)
            
            # Skeleton oluştur
            skeleton_256 = _get_branch_segmentation_skeleton(
                img_gray_resized_norm, mask_resized_binary, 
                branch_seg_model, common_best_model, threshold_param
            )
            
            # Geliştirilmiş branch detection
            branch_assignments = separate_merged_branches(skeleton_256, cell_centers_256, min_branch_length=15)
            
            # Koordinatları orijinal boyuta ölçekle
            scale_x = original_img.shape[1] / 256.0
            scale_y = original_img.shape[0] / 256.0
            
            # Mask'ten branch'leri çıkar
            mask_orig = cv2.resize(mask_for_branch, (original_img.shape[1], original_img.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask_orig > 127).astype(np.uint8)
            mask_skeleton = skeletonize(mask_bin).astype(np.uint8)
            
            # Mask'ten branch'leri tespit et
            mask_branches = []
            visited_mask = set()
            h, w = mask_skeleton.shape
            
            def trace_mask_branch(x, y):
                stack = [(x, y)]
                branch = []
                while stack:
                    cx, cy = stack.pop()
                    if (cx, cy) not in visited_mask:
                        visited_mask.add((cx, cy))
                        branch.append((cx, cy))
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                nx, ny = cx + dx, cy + dy
                                if (0 <= nx < w and 0 <= ny < h and 
                                    mask_skeleton[ny, nx] and 
                                    (nx, ny) not in visited_mask):
                                    stack.append((nx, ny))
                return branch
            
            # Maskeden branchleri bul
            for y in range(h):
                for x in range(w):
                    if mask_skeleton[y, x] and (x, y) not in visited_mask:
                        branch = trace_mask_branch(x, y)
                        if len(branch) > 15:  # Min uzunluk kontrolü
                            mask_branches.append(branch)
            
            # Branch assignments'ları ölçekle
            scaled_assignments = []
            for assignment in branch_assignments:
                scaled_path = [(int(x * scale_x), int(y * scale_y)) 
                            for x, y in assignment['branch_path']]
                scaled_assignment = assignment.copy()
                scaled_assignment['branch_path'] = scaled_path
                scaled_assignment['start_point'] = (int(assignment['start_point'][0] * scale_x), 
                                                int(assignment['start_point'][1] * scale_y))
                scaled_assignment['end_point'] = (int(assignment['end_point'][0] * scale_x), 
                                                int(assignment['end_point'][1] * scale_y))
                scaled_assignment['length'] = calculate_branch_length(scaled_path)
                scaled_assignments.append(scaled_assignment)
            
            # Hücre merkezlerini ölçekle
            scaled_cell_centers = [(int(x * scale_x), int(y * scale_y)) 
                                for x, y in cell_centers_256]
            
            # Görselleştirme
            overlay_img = visualize_improved_branches(original_img, scaled_assignments, 
                                                   scaled_cell_centers, mask_branches)
            
            # Detaylı branch bilgileri
            branch_details = []
            for idx, assignment in enumerate(scaled_assignments, start=1):  # 1'den başlayarak numaralandır
        
                branch_path = assignment['branch_path']
                if len(branch_path) > 1:
                        # Branch'i çiz
                        for i in range(len(branch_path) - 1):
                            cv2.line(overlay_img, branch_path[i], branch_path[i+1], (0, 0, 255), 2)
                        
                        # Sadece branch numarasını yaz
                        mid_point = (
                            (assignment['start_point'][0] + assignment['end_point'][0]) // 2,
                            (assignment['start_point'][1] + assignment['end_point'][1]) // 2
                        )
                        
                        # Beyaz arka planlı metin
                        text = f"B{idx}"
                        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        
                        # Arka plan dikdörtgeni
                        cv2.rectangle(overlay_img, 
                                    (mid_point[0] - text_width//2 - 5, mid_point[1] - text_height//2 - 5),
                                    (mid_point[0] + text_width//2 + 5, mid_point[1] + text_height//2 + 5),
                                    (255, 255, 255), -1)
                        
                        # Branch numarasını yaz
                        cv2.putText(overlay_img, text, 
                                    (mid_point[0] - text_width//2, mid_point[1] + text_height//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)       
                branch_details.append({
                    'branchId': idx,  # idx kullanarak 1'den başlat
                    'length': round(assignment['length'], 2),
                    'startCell': assignment.get('start_cell', -1),
                    'startPoint': assignment['start_point'],
                    'endPoint': assignment['end_point'],
                    'distanceToCell': round(assignment.get('distance_to_cell', 0), 2)
                })
            
            # Histogram verileri - branch numaralarını da 1'den başlat
            hist_labels = [f"Branch {i+1}" for i in range(len(scaled_assignments))]  # 1'den başlayarak etiketler
            hist_data = [round(assignment['length'], 2) for assignment in scaled_assignments]
            
            results = {
                'totalBranches': len(scaled_assignments),
                'cellsDetected': len(scaled_cell_centers),
                'averageLength': round(np.mean(hist_data), 2) if hist_data else 0,
                'minLength': round(np.min(hist_data), 2) if hist_data else 0,
                'maxLength': round(np.max(hist_data), 2) if hist_data else 0,
                'stdLength': round(np.std(hist_data), 2) if hist_data else 0,
                'branchDetails': branch_details,
                'histograms': { 
                    'branchLength': { 
                        'labels': hist_labels, 
                        'data': hist_data 
                    }
                }
            }
            
            # Mask ve model branch'lerini ayrı ayrı işle
            mask_orig = cv2.resize(mask_for_branch, (original_img.shape[1], original_img.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask_orig > 127).astype(np.uint8)
            
            # Model çıktısından gelen branch'ler için görselleştirme
            overlay_img = visualize_improved_branches(original_img, scaled_assignments, 
                                                   scaled_cell_centers, mask_branches)
            
            # Base64'e çevir
            encoded_image = encode_image_to_base64(overlay_img)
            
            return jsonify({
                'processedImage': encoded_image,
                'analysisResults': results
            })
            
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
            # Dinamik maske yolu oluştur
            mask_filename = f"{image_name.split('.')[0]}.png"
            mask_path = os.path.join(BRANCH_MASKS_PATH, mask_filename)
            
            if not os.path.exists(mask_path):
                return jsonify({"error": f"Maske bulunamadı: {mask_path}"}), 404
                
            mask_for_branch = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_for_branch is None:
                return jsonify({"error": f"Dal maskesi yüklenemedi: {mask_path}"}), 500

            # Görüntü işleme
            img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            img_gray_resized = cv2.resize(img_gray, (256, 256))
            img_gray_resized_norm = img_gray_resized.astype(np.float32) / 255.0
            
            mask_resized = cv2.resize(mask_for_branch, (256, 256), interpolation=cv2.INTER_NEAREST)
            mask_resized_binary = (mask_resized > 127).astype(np.uint8)

            # Cell merkezlerini tespit et
            cell_mask_filename = f"{image_name.split('.')[0]}.png"
            cell_mask_path = os.path.join(CELL_MASKS_PATH, cell_mask_filename)
            
            cell_centers_256 = []
            if os.path.exists(cell_mask_path):
                cell_mask = cv2.imread(cell_mask_path, cv2.IMREAD_GRAYSCALE)
                if cell_mask is not None:
                    cell_mask_resized = cv2.resize(cell_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                    cell_mask_binary = (cell_mask_resized > 127).astype(np.uint8)
                    cell_centers_256 = detect_cell_centers(cell_mask_binary)

            if not cell_centers_256:
                cell_centers_256 = detect_cell_centers(mask_resized_binary)

            # Skeleton oluştur
            skeleton_256 = _get_branch_segmentation_skeleton(
                img_gray_resized_norm, mask_resized_binary, 
                branch_seg_model, common_best_model, threshold_param
            )
            
            # Branch'leri tespit et
            branch_assignments = separate_merged_branches(skeleton_256, cell_centers_256)
            
            # Koordinatları ölçekle
            scale_x = original_img.shape[1] / 256.0
            scale_y = original_img.shape[0] / 256.0
            
            # Ölçeklenmiş branch assignments
            scaled_assignments = []
            for assignment in branch_assignments:
                scaled_path = [(int(x * scale_x), int(y * scale_y)) 
                            for x, y in assignment['branch_path']]
                scaled_assignment = assignment.copy()
                scaled_assignment['branch_path'] = scaled_path
                scaled_assignment['start_point'] = (int(assignment['start_point'][0] * scale_x), 
                                                int(assignment['start_point'][1] * scale_y))
                scaled_assignment['end_point'] = (int(assignment['end_point'][0] * scale_x), 
                                            int(assignment['end_point'][1] * scale_y))
                scaled_assignment['length'] = calculate_branch_length(scaled_path)
                scaled_assignments.append(scaled_assignment)

            # Hücre merkezlerini ölçekle
            scaled_cell_centers = [(int(x * scale_x), int(y * scale_y)) 
                                for x, y in cell_centers_256]

            # Mask'ten branch'leri tespit et
            mask_orig = cv2.resize(mask_for_branch, (original_img.shape[1], original_img.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask_orig > 127).astype(np.uint8)
            mask_skeleton = skeletonize(mask_bin).astype(np.uint8)
            
            # Mask branch'lerini bul
            mask_branches = []
            visited_mask = set()
            h, w = mask_skeleton.shape
            
            def trace_mask_branch(x, y):
                stack = [(x, y)]
                branch = []
                while stack:
                    cx, cy = stack.pop()
                    if (cx, cy) not in visited_mask:
                        visited_mask.add((cx, cy))
                        branch.append((cx, cy))
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                nx, ny = cx + dx, cy + dy
                                if (0 <= nx < w and 0 <= ny < h and 
                                    mask_skeleton[ny, nx] and 
                                    (nx, ny) not in visited_mask):
                                    stack.append((nx, ny))
                return branch
            
            for y in range(h):
                for x in range(w):
                    if mask_skeleton[y, x] and (x, y) not in visited_mask:
                        branch = trace_mask_branch(x, y)
                        if len(branch) > 15:
                            mask_branches.append(branch)

            # Branch görselleştirmesi
            overlay_img = visualize_improved_branches(original_img, scaled_assignments, 
                                                scaled_cell_centers, mask_branches)

            # Model branch'lerinin uzunluklarını hesapla
            lengths = [assignment['length'] for assignment in scaled_assignments]

            # Her branch için uzunluk bilgisini görüntü üzerine yaz
            for idx, assignment in enumerate(scaled_assignments):
                branch_path = assignment['branch_path']
                length = assignment['length']
                
                # Branch'in orta noktasını bul
                mid_point = (
                    (assignment['start_point'][0] + assignment['end_point'][0]) // 2,
                    (assignment['start_point'][1] + assignment['end_point'][1]) // 2
                )
                
                # Uzunluk etiketini yaz
                text = f"B{idx+1}: {length:.1f}"
                font_scale = 1.1
                font_thickness = 2

                # Yazı boyutunu al
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

                # Arka plan dikdörtgeni
                cv2.rectangle(overlay_img, 
                            (mid_point[0] - text_width // 2 - 10, mid_point[1] - text_height // 2 - 10),
                            (mid_point[0] + text_width // 2 + 10, mid_point[1] + text_height // 2 + 10),
                            (255, 255, 255), -1)

                # Yazıyı çiz (bunu da büyütmelisin, örnek olarak ekliyorum)
                cv2.putText(overlay_img, text, 
                            (mid_point[0] - text_width // 2, mid_point[1] + text_height // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
                
                # # Metni yaz
                # cv2.putText(overlay_img, text, 
                #         (mid_point[0] - text_width//2, mid_point[1] + text_height//2),
                #         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

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
            hist_bins = np.arange(0, max(lengths) + 100, 100) if lengths else np.arange(0, 100, 100)
            hist_values, _ = np.histogram(lengths, bins=hist_bins)
            hist_labels = [f"{int(hist_bins[i])}-{int(hist_bins[i+1])}" for i in range(len(hist_bins)-1)]

            encoded_image = encode_image_to_base64(overlay_img)
            return jsonify({
                'processedImage': encoded_image,
                'analysisResults': {
                    **basic_stats,
                    **advanced_stats,
                    'branchDetails': [
                        {
                            'id': idx + 1,
                            'length': float(length),
                            'startPoint': assignment['start_point'],
                            'endPoint': assignment['end_point']
                        }
                        for idx, (length, assignment) in enumerate(zip(lengths, scaled_assignments))
                    ],
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
    elif analysis_type == 'angles':
        try:
            # Branch analizinden gelen tüm işlemleri aynen alıyoruz
            mask_filename = f"{image_name.split('.')[0]}.png"
            mask_path = os.path.join(BRANCH_MASKS_PATH, mask_filename)
            
            if not os.path.exists(mask_path):
                return jsonify({"error": f"Maske bulunamadı: {mask_path}"}), 404
                
            mask_for_branch = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_for_branch is None:
                return jsonify({"error": f"Dal maskesi yüklenemedi: {mask_path}"}), 500

            img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            img_gray_resized = cv2.resize(img_gray, (256, 256))
            img_gray_resized_norm = img_gray_resized.astype(np.float32) / 255.0
            
            mask_resized = cv2.resize(mask_for_branch, (256, 256), interpolation=cv2.INTER_NEAREST)
            mask_resized_binary = (mask_resized > 127).astype(np.uint8)

            # Cell merkezlerini tespit et
            cell_mask_filename = f"{image_name.split('.')[0]}.png"
            cell_mask_path = os.path.join(CELL_MASKS_PATH, cell_mask_filename)
            
            cell_centers_256 = []
            if os.path.exists(cell_mask_path):
                cell_mask = cv2.imread(cell_mask_path, cv2.IMREAD_GRAYSCALE)
                if cell_mask is not None:
                    cell_mask_resized = cv2.resize(cell_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                    cell_mask_binary = (cell_mask_resized > 127).astype(np.uint8)
                    cell_centers_256 = detect_cell_centers(cell_mask_binary)

            if not cell_centers_256:
                cell_centers_256 = detect_cell_centers(mask_resized_binary)

            # Skeleton oluştur ve branch'leri tespit et
            skeleton_256 = _get_branch_segmentation_skeleton(
                img_gray_resized_norm, mask_resized_binary, 
                branch_seg_model, common_best_model, threshold_param
            )
            
            branch_assignments = separate_merged_branches(skeleton_256, cell_centers_256)
            
            # Ölçekleme faktörleri
            scale_x = original_img.shape[1] / 256.0
            scale_y = original_img.shape[0] / 256.0
            
            # Model branch'lerini ölçekle
            scaled_assignments = []
            for assignment in branch_assignments:
                scaled_path = [(int(x * scale_x), int(y * scale_y)) 
                            for x, y in assignment['branch_path']]
                scaled_assignment = assignment.copy()
                scaled_assignment['branch_path'] = scaled_path
                scaled_assignment['start_point'] = (int(assignment['start_point'][0] * scale_x), 
                                                    int(assignment['start_point'][1] * scale_y))
                scaled_assignment['end_point'] = (int(assignment['end_point'][0] * scale_x), 
                                                int(assignment['end_point'][1] * scale_y))
                scaled_assignment['length'] = calculate_branch_length(scaled_path)
                scaled_assignments.append(scaled_assignment)

            # Mask'ten branch'leri tespit et
            mask_orig = cv2.resize(mask_for_branch, (original_img.shape[1], original_img.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask_orig > 127).astype(np.uint8)
            mask_skeleton = skeletonize(mask_bin).astype(np.uint8)
            
            # Mask branch'lerini bul
            mask_branches = []
            visited_mask = set()
            h, w = mask_skeleton.shape
            
            def trace_mask_branch(x, y):
                stack = [(x, y)]
                branch = []
                while stack:
                    cx, cy = stack.pop()
                    if (cx, cy) not in visited_mask:
                        visited_mask.add((cx, cy))
                        branch.append((cx, cy))
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                nx, ny = cx + dx, cy + dy
                                if (0 <= nx < w and 0 <= ny < h and 
                                    mask_skeleton[ny, nx] and 
                                    (nx, ny) not in visited_mask):
                                    stack.append((nx, ny))
                return branch
            
            for y in range(h):
                for x in range(w):
                    if mask_skeleton[y, x] and (x, y) not in visited_mask:
                        branch = trace_mask_branch(x, y)
                        if len(branch) > 15:
                            mask_branches.append(branch)

            # Önce branch'leri çiz
            scaled_cell_centers = [(int(x * scale_x), int(y * scale_y)) for x, y in cell_centers_256]
            overlay_img = visualize_improved_branches(original_img, scaled_assignments, 
                                                    scaled_cell_centers, mask_branches)

            # Kesişen branch'leri bul
            h, w = original_img.shape[:2]
            model_mask = np.zeros((h, w), dtype=np.uint8)
            mask_mask = np.zeros((h, w), dtype=np.uint8)

            for assignment in scaled_assignments:
                branch_path = assignment['branch_path']
                if len(branch_path) > 1:
                    for i in range(len(branch_path) - 1):
                        cv2.line(model_mask, branch_path[i], branch_path[i+1], 1, 8)

            for branch_path in mask_branches:
                if len(branch_path) > 1:
                    for i in range(len(branch_path) - 1):
                        cv2.line(mask_mask, branch_path[i], branch_path[i+1], 1, 3)

            intersection_mask = cv2.bitwise_and(model_mask, mask_mask)

            # Açıları hesapla ve çiz
            angles = []
            for assignment in scaled_assignments:
                branch_path = assignment['branch_path']
                if len(branch_path) > 1:
                    start_point = assignment['start_point']
                    end_point = assignment['end_point']
                    
                    # Açı hesaplama
                    dx = end_point[0] - start_point[0]
                    dy = end_point[1] - start_point[1]
                    angle = (math.degrees(math.atan2(-dy, dx)) + 360) % 360
                    angles.append({
                        'branchId': assignment.get('branch_id', -1),
                        'startPoint': start_point,
                        'endPoint': end_point,
                        'angle': float(round(angle, 2)),
                        'length': float(round(assignment['length'], 2))
                    })

                    # Ok çizimi: 200px uzunluk, beyaz kontur + sarı iç
                    dist = math.hypot(dx, dy)
                    ux, uy = (dx/dist, dy/dist) if dist > 0 else (1.0, 0.0)
                    arrow_length = 200
                    arrow_end = (
                        int(start_point[0] + ux * arrow_length),
                        int(start_point[1] + uy * arrow_length)
                    )
                    # Kontur
                    cv2.arrowedLine(overlay_img, start_point, arrow_end, (255,255,255), thickness=8, tipLength=0.3)
                    # İç
                    cv2.arrowedLine(overlay_img, start_point, arrow_end, (255,255,255), thickness=4, tipLength=0.3)
                    
                    # Açı etiketi
                    mid_point = ((start_point[0] + arrow_end[0]) // 2, (start_point[1] + arrow_end[1]) // 2)
                    text = f"{round(angle,1)}"
                    font_scale, thickness = 1.2, 3
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    pad = 10
                    cv2.rectangle(overlay_img,
                                  (mid_point[0]-tw//2-pad, mid_point[1]-th//2-pad),
                                  (mid_point[0]+tw//2+pad, mid_point[1]+th//2+pad),
                                  (255,255,255), -1)
                    cv2.putText(overlay_img, text,
                                (mid_point[0]-tw//2, mid_point[1]+th//2),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness)

            # Açı hesaplama ve çizim işlemleri tamamlandıktan sonra:
            angles = []
            for assignment in scaled_assignments:
                branch_path = assignment['branch_path']
                if len(branch_path) > 1:
                    start_point = assignment['start_point']
                    end_point = assignment['end_point']
                    
                    # Açı hesaplama
                    dx = end_point[0] - start_point[0]
                    dy = end_point[1] - start_point[1]
                    angle = (math.degrees(math.atan2(-dy, dx)) + 360) % 360
                    angles.append({
                        'branchId': assignment.get('branch_id', -1),
                        'startPoint': start_point,
                        'endPoint': end_point,
                        'angle': float(round(angle, 2)),
                        'length': float(round(assignment['length'], 2))
                    })

                    # Ok çizimi: 200px uzunluk, beyaz kontur + sarı iç
                    dist = math.hypot(dx, dy)
                    ux, uy = (dx/dist, dy/dist) if dist > 0 else (1.0, 0.0)
                    arrow_length = 200
                    arrow_end = (
                        int(start_point[0] + ux * arrow_length),
                        int(start_point[1] + uy * arrow_length)
                    )
                    # Kontur
                    cv2.arrowedLine(overlay_img, start_point, arrow_end, (255,255,255), thickness=8, tipLength=0.3)
                    # İç
                    cv2.arrowedLine(overlay_img, start_point, arrow_end, (255,255,255), thickness=4, tipLength=0.3)
                    
                    # Açı etiketi
                    mid_point = ((start_point[0] + arrow_end[0]) // 2, (start_point[1] + arrow_end[1]) // 2)
                    text = f"{round(angle,1)}"
                    font_scale, thickness = 1.2, 3
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    pad = 10
                    cv2.rectangle(overlay_img,
                                  (mid_point[0]-tw//2-pad, mid_point[1]-th//2-pad),
                                  (mid_point[0]+tw//2+pad, mid_point[1]+th//2+pad),
                                  (255,255,255), -1)
                    cv2.putText(overlay_img, text,
                                (mid_point[0]-tw//2, mid_point[1]+th//2),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness)

            # Açıları hesapla
            angle_values = [a['angle'] for a in angles]
            if angle_values:

                if angle_values and len(angle_values) > 0:
                    angle_array =  np.array(angle_values) 
                    avgAngle = float(angle_array.mean())
                    minAngle = float(angle_array.min())
                    maxAngle = float(angle_array.max())
                    stdDevAngle = float(angle_array.std()) if len(angle_array) > 1 else 0.0
                else:
                    avgAngle = minAngle = maxAngle = stdDevAngle = 0.0
                angleSkewness = float(scipy_stats.skew(angle_values)) if len(angle_values) > 1 else 0.0
                angleKurtosis = float(scipy_stats.kurtosis(angle_values)) if len(angle_values) > 1 else 0.0

                # Resultant vector length: Ortalama ünite vektörün normu
                vectors = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in angle_values]
                sum_vector = np.sum(vectors, axis=0)
                resultantVectorLength = float(np.linalg.norm(sum_vector) / len(angle_values))

                # Angular entropy: 12 bin (0-360 arası, bin genişliği 30) kullanılarak hesaplanıyor
                hist_counts, _ = np.histogram(angle_values, bins=np.arange(0, 361, 30))
                p = hist_counts / np.sum(hist_counts) if np.sum(hist_counts) > 0 else np.zeros_like(hist_counts)
                angularEntropy = -float(np.sum([p_val * np.log(p_val) for p_val in p if p_val > 0]))
            else:
                avgAngle = minAngle = maxAngle = stdDevAngle = angleSkewness = angleKurtosis = resultantVectorLength = angularEntropy = 0.0

            stats = {
                'totalBranches': len(angles),
                'averageAngle': avgAngle,
                'minAngle': minAngle,
                'maxAngle': maxAngle,
                'stdDevAngle': stdDevAngle,
                'angleSkewness': angleSkewness,
                'angleKurtosis': angleKurtosis,
                'resultantVectorLength': resultantVectorLength,
                'angularEntropy': angularEntropy,
                'fractalDimension': float(compute_fractal_dimension(skeleton_256)),
                'maxBranchOrder': compute_max_branch_order(skeleton_256),
                'nodeDegree': compute_average_node_degree(skeleton_256),
                'nodeDegreeAverage': compute_average_node_degree(skeleton_256),
                'convexHullCompactness': compute_convex_hull_compactness(skeleton_256),
                #zort
                'angleDetails': angles,
                'histograms': {
                    'angles': {
                        'labels': [f"{i}-{i+30}" for i in range(0, 360, 30)],
                        'data': np.histogram(angle_values, bins=np.arange(0, 361, 30))[0].tolist()
                    }
                }
            }

            encoded_image = encode_image_to_base64(overlay_img)
            return jsonify({
                'processedImage': encoded_image,
                'analysisResults': stats
            })

        except Exception as e:
            app.logger.error(f"'angles' analizi sırasında hata: {str(e)}", exc_info=True)
            return jsonify({"error": f"'angles' analizi sırasında hata: {str(e)}"}), 500

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