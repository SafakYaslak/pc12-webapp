import io
import numbers
import os
from imageComparisonWidget import ImageComparisonApp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0
from tkinter import *
from tkinter.ttk import *
import shutil
import tensorflow as tf
import cv2
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtCore
from PyQt5.QtGui import *
from functools import partial
import atexit
import sys
import new
from PyQt5.QtWidgets import*
from PyQt5.QtCore import *
import sys
from PyQt5 import *
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import convolve, label, measurements
from skimage.morphology import skeletonize
from PIL import Image
from matplotlib.colors import hsv_to_rgb
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
tf.config.set_visible_devices([], 'GPU')
import copy
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
import io
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from matplotlib.colors import hsv_to_rgb
import os
from PyQt5.QtCore import Qt
import os
os.chdir(os.path.join(os.getcwd(), "ImageAI"))  # Çalışma dizinini "ImageAI" klasörüne ayarla





class Main_Window(QMainWindow):
    def __init__(self):
        super().__init__()  # call super function for override

        self.slider_values = {}


        self.main_window = uic.loadUi("untitled-group.ui", self)

        self.setWindowIcon(QIcon("images/icon_ai.png"))

        self.predict_button.setVisible(False)
        self.compare_button.setVisible(False)
        self.overlap_button_two.setVisible(False)
        self.branch_lenght_button.setVisible(False)
        self.branch_butotn.setVisible(False)
        self.angle_buton.setVisible(False)
        self.cell_button.setVisible(False)
        self.cell_area_button.setVisible(False)
        self.tabWidget.setVisible(False)

        self.slider_angles_each_branch = self.findChild(QSlider, "slider_angles_each_branch")
        self.slider_branch = self.findChild(QSlider, "slider_branch")
        self.slider_branch_length = self.findChild(QSlider, "slider_branch_length")
        self.slider_cell = self.findChild(QSlider, "slider_cell")
        self.slider_cell_area = self.findChild(QSlider, "slider_cell_area")

        self.slider_cell.setVisible(False)
        self.slider_cell_area.setVisible(False)
        self.slider_branch.setVisible(False)
        self.slider_branch_length.setVisible(False)
        self.slider_angles_each_branch.setVisible(False)



        icon = QIcon("images/icon_add_folder.png")
        self.pushButton.setIcon(icon)
        self.pushButton.setStyleSheet("background-color: rgba(0, 0, 0, 0);")

        icon2 = QIcon("images/icon_exit_2.png")
        self.exit_button.setIcon(icon2)
        self.exit_button.setStyleSheet("background-color: rgba(0, 0, 0, 0);")

        self.new=new.new_widget()
        self.dir_list = []
        self.predict_path = ""
        self.predict_mask_path = ""
        self.new_window = None
        self.cell_size = 0
        self.branch_size= 0
        self.counter1=0
        self.counter2=0
        self.cell_pice =0
        self.cell_pice_mask =0
        self.cell_areas_mask =0
        self.cell_areas =0
        self.branch_pice =0
        self.branch_pice_mask =0
        self.branch_areas_mask =0
        self.branch_areas =0

        with tf.keras.utils.custom_object_scope({'dice_coef': self.dice_coef, 'iou_score': self.iou_score}):
            model_path = 'best_model.hdf5'
            self.model_base = tf.keras.models.load_model(model_path)

        with tf.keras.utils.custom_object_scope({'dice_coef': self.dice_coef, 'iou_score': self.iou_score}):
            model_path = 'best_branch_model.hdf5'
            self.model = tf.keras.models.load_model(model_path)

        with tf.keras.utils.custom_object_scope({'dice_coef': self.dice_coef, 'iou_score': self.iou_score}):
            model_path2 = 'best_cell_model.hdf5'
            self.model2 = tf.keras.models.load_model(model_path2)

        self.imglabel = self.findChild(QLabel, "imglabel")
        self.imglabel2 = self.findChild(QLabel, "imglabel2")
        # self.imglabel3 = self.findChild(QLabel, "imglabel3")
        self.image_NumR = self.findChild(QLabel, "image_NumR")
        self.image_NumL= self.findChild(QLabel, "image_NumL")
        self.number_cell = self.findChild(QLabel, "number_cell")
        self.number_branch = self.findChild(QLabel, "number_branch")
        self.radioButton = self.findChild(QCheckBox, "checkBox_2")
        self.radioButton_separate = self.findChild(QCheckBox, "checkBox")
        self.angle_button = self.findChild(QPushButton, "angle_buton")

        self.label_cell_piece_histogram = self.findChild(QLabel, "label_cell_piece_histogram")
        self.label_cell_area_histogram = self.findChild(QLabel, "label_cell_area_histogram")
        self.label_branch_piece_histogram = self.findChild(QLabel, "label_branch_piece_histogram")
        self.label_branch_length_histogram = self.findChild(QLabel, "label_branch_length_histogram")

        self.label_5.setVisible(False)
        self.label_3.setVisible(False)
        self.scrollArea_2.hide()
        # self.imglabel3.hide()

        self.pushButton.clicked.connect(self.open_dialog)
        self.exit_button.clicked.connect(self.close_signal_triggered2)
        self.actionLog_Out.triggered.connect(self.out_log)
        self.angle_button.clicked.connect(self.angle)
        self.predict_button.clicked.connect(self.predict)
        self.overlap_button_two.clicked.connect(self.overlap_two)
        self.cell_button.clicked.connect(self.cell)
        self.branch_butotn.clicked.connect(self.branch)
        self.cell_area_button.clicked.connect(self.cell_area)
        self.branch_lenght_button.clicked.connect(self.branch_lenght)
        self.compare_button.clicked.connect(self.compare)

        # self.apply_histogram_button.clicked.connect(self.create_histograms)

        self.save_button.clicked.connect(self.update_all_calculations)
        self.predicted_path = list()

        self.number_cell.hide()
        self.number_branch.hide()

    from PyQt5.QtWidgets import QMessageBox

    def create_cell_area_histogram(self):
        """
        Generates a cell area histogram and returns the matplotlib figure.

        Returns:
            matplotlib.figure.Figure: The generated histogram figure
        """
        try:
            # Read and process the image
            input_image_path = 'prediction_result_cell.jpg'
            if not os.path.exists(input_image_path):
                raise FileNotFoundError("Image file not found")

            img = cv2.imread(input_image_path)
            if img is None:
                raise ValueError("Failed to load the image")

            # Image processing
            blue_channel = 255 - img[:, :, 1]
            blue_channel = self.contrast_stretching(blue_channel, 1, 100)
            threshold_value = self.slider_cell_area.value()
            mask = cv2.threshold(blue_channel, threshold_value, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # Find connected components
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

            if num_labels <= 1:
                raise ValueError("No cells detected")

            # Get the cell areas (excluding the first component - background)
            cell_areas = stats[1:, cv2.CC_STAT_AREA]

            # Create the histogram figure
            plt.clf()  # Clear any previous plot
            fig = plt.figure(figsize=(10, 10))
            plt.hist(cell_areas, bins=30, color='skyblue', edgecolor='black')

            plt.title('Cell Area Histogram' , fontsize = 40, pad = 10)
            plt.xlabel('Area (pixels)', fontsize = 30)
            plt.ylabel('Frequency', fontsize = 30)
            plt.grid(True, alpha=0.3)

            # Add statistical information
            mean_area = np.mean(cell_areas)
            std_area = np.std(cell_areas)
            plt.axvline(mean_area, color='red', linestyle='dashed', linewidth=1)
            plt.text(mean_area + 5, plt.gca().get_ylim()[1] * 0.9,
                     f'Mean: {mean_area:.1f}\nStd: {std_area:.1f}',
                     bbox=dict(facecolor='white', alpha=0.8))

            # Display and save the figure
            plt.tight_layout()

            # Place the figure on the QLabel
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', transparent=True)
            buf.seek(0)

            image = QImage.fromData(buf.getvalue())
            pixmap = QPixmap.fromImage(image)
            self.label_cell_area_histogram.setPixmap(pixmap)
            self.label_cell_area_histogram.setScaledContents(True)

            # Store the statistics
            self.cell_areas = cell_areas
            self.cell_area_stats = {
                'mean': mean_area,
                'std': std_area,
                'min': np.min(cell_areas),
                'max': np.max(cell_areas),
                'count': len(cell_areas)
            }

            self.label_cell_area_histogram.mousePressEvent = self.cell_area_histogram_label_clicked
            return fig


        except Exception as e:
            print(f"Error creating histogram: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            plt.close('all')  # Close all open figures

    def cell_area_histogram_label_clicked(self, event):
        """Function called when label_cell_area_histogram is clicked."""
        if hasattr(self, 'cell_areas') and isinstance(self.cell_areas, np.ndarray) and self.cell_areas.size > 0:

            # Check if cell_areas is a 1D array of area values
            if isinstance(self.cell_areas, np.ndarray) and self.cell_areas.ndim == 1:
                # Display a histogram of cell areas
                plt.figure(figsize=(10, 10))
                plt.hist(self.cell_areas, bins=20, color='skyblue', edgecolor='black')
                plt.title('Detailed Cell Area Histogram', fontsize=50, pad=5)
                plt.xlabel('Area (pixels)', fontsize=30)
                plt.ylabel('Frequency', fontsize=30)
                plt.grid(True, alpha=0.3)
                plt.show()
            else:
                print("Error: cell_areas is not in the expected 1D format.")
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Cell areas data is not in a valid format for histogram display.")
                msg.setWindowTitle("Error")
                msg.exec_()
        else:
            # Show a warning message if cell_areas data is not available
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please process the cell area image first!")
            msg.setWindowTitle("Warning")
            msg.exec_()

    def get_cell_area_stats(self):
        """
        Returns the cell area statistics
        """
        if hasattr(self, 'cell_area_stats'):
            return self.cell_area_stats
        return None

    # Branch Length Histogram Creation and Display
    def create_branch_length_histogram(self):
        """
        Generates a branch length histogram and returns the matplotlib figure.

        Returns:
            matplotlib.figure.Figure: The generated histogram figure
        """
        try:
            # Check if there's data available for branch lengths
            if not hasattr(self, 'branch_length_contours') or not self.branch_length_contours:
                raise ValueError("No branch length data available")

            # Calculate lengths of each branch
            branch_lengths = [cv2.arcLength(contour, True) for contour in self.branch_length_contours]

            # Check if branch_lengths contains data
            if len(branch_lengths) == 0:
                raise ValueError("No branch lengths calculated from contours.")

            # Create the histogram figure
            plt.clf()
            fig = plt.figure(figsize=(10, 10))
            plt.hist(branch_lengths, bins=20, color='skyblue', edgecolor='black')
            plt.title('Branch Length Histogram', fontsize=50, pad=10)
            plt.xlabel('Length (pixels)', fontsize=30)
            plt.ylabel('Frequency', fontsize=30)
            plt.grid(True, alpha=0.3)

            # Add statistical information
            mean_length = np.mean(branch_lengths)
            std_length = np.std(branch_lengths)
            plt.axvline(mean_length, color='red', linestyle='dashed', linewidth=1)
            plt.text(mean_length + 5, plt.gca().get_ylim()[1] * 0.9,
                     f'Mean: {mean_length:.1f}\nStd: {std_length:.1f}',
                     bbox=dict(facecolor='white', alpha=0.8))

            # Display the branch count
            branch_count = len(branch_lengths)


            # Display and save the figure
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', transparent=True)
            buf.seek(0)

            image = QImage.fromData(buf.getvalue())
            pixmap = QPixmap.fromImage(image)
            self.label_branch_length_histogram.setPixmap(pixmap)
            self.label_branch_length_histogram.setScaledContents(True)

            # Store the branch length data and statistics
            self.branch_lengths = branch_lengths
            self.branch_length_stats = {
                'mean': mean_length,
                'std': std_length,
                'min': np.min(branch_lengths),
                'max': np.max(branch_lengths),
                'count': branch_count
            }

            self.label_branch_length_histogram.mousePressEvent = self.branch_length_histogram_label_clicked
            return fig

        except Exception as e:
            print(f"Error generating branch length histogram: {e}")
            QMessageBox.warning(self, "Warning", "No data available for branch length histogram.")
            return None
        finally:
            plt.close('all')

    def branch_length_histogram_label_clicked(self, event):
        """Function called when label_branch_length_histogram is clicked."""
        if hasattr(self, 'branch_lengths') and isinstance(self.branch_lengths, list) and self.branch_lengths:
            # Display the histogram of branch lengths
            plt.figure(figsize=(10, 10))
            plt.hist(self.branch_lengths, bins=20, color='skyblue', edgecolor='black')
            plt.title('Detailed Branch Length Histogram', fontsize=50, pad=5)
            plt.xlabel('Length (pixels)', fontsize=30)
            plt.ylabel('Frequency', fontsize=30)
            plt.grid(True, alpha=0.3)
            plt.show()
        else:
            QMessageBox.warning(self, "Warning", "Please process the branch length image first!")

    def get_branch_length_stats(self):
        """Returns the branch length statistics."""
        if hasattr(self, 'branch_length_stats'):
            return self.branch_length_stats
        return None

    def setup_histogram(self, data, xlabel, ylabel, title):
        """Setup a histogram figure with given parameters."""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.hist(data, bins=30, color='skyblue', edgecolor='black')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.tight_layout()
        return fig, ax

    def display_histogram_on_label(self, fig, label):
        """Display a matplotlib figure on a QLabel."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)

        data = buf.getvalue()
        image = QImage.fromData(data)
        pixmap = QPixmap.fromImage(image)

        label.setPixmap(pixmap)
        label.setScaledContents(True)

        plt.close(fig)
        buf.close()

    def update_all_calculations(self):
        # Debug print to confirm button click
        print("Save button clicked, updating calculations...")

        # Save the slider values first
        self.save_slider_values()

        # Re-run calculations that depend on updated slider values
        self.branch_lenght()
        self.branch()
        self.angle()
        self.cell()
        self.cell_area()

        # Force a repaint on labels or widgets that display results
        self.label_branch_length.repaint()  # example widget
        self.label_cell.repaint()  # example widget
        self.label_cell_area.repaint()  # example widget
        self.repaint()  # force full window repaint if necessary

        print("Calculations updated based on slider values.")

    def save_slider_values(self):
        # Use self.slider_values to store slider values
        self.slider_values = {
            'angles_each_branch': self.slider_angles_each_branch.value(),
            'branch': self.slider_branch.value(),
            'branch_length': self.slider_branch_length.value(),
            'cell': self.slider_cell.value(),
            'cell_area': self.slider_cell_area.value(),

        }


    def overlap_two(self):
        image1_path="branch1.jpg"
        image2_path="esikli_Resim.jpg"
        image3_path="resized_image.jpg"

        if not all(map(os.path.exists, [image1_path, image2_path, image3_path])):
            # Popup mesajı oluşturun
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please choose an entry image!")
            msg.setWindowTitle("Warning")
            msg.exec_()
        else:
            colorful_branch_path ="colorful_branch.jpg"
            colorful_cell_path = "colorful_cell.jpg"
            self.change_white_to_color(image1_path,(255,0,0,),colorful_branch_path)
            self.change_white_to_color(image2_path,(128,128,128,),colorful_cell_path)
            self.new_window = ImageComparisonApp(colorful_branch_path, colorful_cell_path, image3_path)
            self.new_window.show()
    def predict(self):
        if not self.predict_path:
            QMessageBox.warning(None, "Warning", "Please choose an entry image!")
            return

        # Input image preprocessing
        input_image = cv2.imread(self.predict_path, cv2.IMREAD_GRAYSCALE)
        input_image = cv2.resize(input_image, (256, 256))
        input_image = np.expand_dims(input_image, axis=(-1, 0))

        # Make predictions
        predictions = {
            'cell': self.model2.predict(input_image),
            'branch': self.model.predict(input_image),
            'best': self.model_base.predict(input_image)
        }

        # Save predictions
        def save_prediction(pred_data, base_name):
            # Save colored version
            colored_path = f'prediction_result_{base_name}.jpg'
            pred_img = (pred_data * 255).astype(np.uint8)
            cv2.imwrite(colored_path, pred_img[0])

            # Save grayscale version
            gray_path = f'prediction_result_{base_name}_grayscale.jpg'
            gray_img = cv2.imread(colored_path, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(gray_path, gray_img)

            return colored_path, gray_path

        # Process each prediction
        for name, pred in predictions.items():
            colored, gray = save_prediction(pred, name)
            if name == 'cell':
                self.save_cell(colored)
            elif name == 'branch':
                self.save_branch(colored)

        # Save resized input image
        image = Image.open(self.predict_path)
        resized = image.resize((256, 256))
        resized.save("resized_image.jpg")

        # Create grayscale version of resized input
        cv2.imwrite('resized_image_grayscale.jpg',
                    cv2.imread('resized_image.jpg', cv2.IMREAD_GRAYSCALE))

        self.mask_preproccess(self.predict_mask_path)

        # Display results
        if self.radioButton_separate.isChecked():
            if self.radioButton.isChecked():
                # Grayscale mode
                self.merge_side_by_side(
                    'prediction_result_cell_grayscale.jpg',
                    'prediction_result_branch_grayscale.jpg',
                    'compare_grayscale.jpg'
                )
                q_image = self.create_qimage('prediction_result_cell_grayscale.jpg', True)
                q_image2 = self.create_qimage('prediction_result_branch_grayscale.jpg', True)
            else:
                # Color mode
                self.merge_side_by_side(
                    'prediction_result_cell.jpg',
                    'prediction_result_branch.jpg',
                    'compare.jpg'
                )
                q_image = self.create_qimage('prediction_result_cell.jpg', False)
                q_image2 = self.create_qimage('prediction_result_branch.jpg', False)
        else:
            # Single image mode
            image_path = 'prediction_result_best_grayscale.jpg' if self.radioButton.isChecked() \
                else 'prediction_result_best.jpg'
            q_image = self.create_qimage(image_path, self.radioButton.isChecked())

        # Display result
        pixmap = QPixmap.fromImage(q_image)
        self.imglabel2.setPixmap(pixmap)
        self.imglabel2.setScaledContents(True)

    def create_qimage(self, image_path, is_grayscale):
        """Helper function to create QImage from image path"""
        if is_grayscale:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            height, width = img.shape
            return QImage(img.data, width, height, width, QImage.Format_Grayscale8)
        else:
            img = cv2.imread(image_path)
            img_resized = cv2.resize(img, (256, 256))
            mask = np.zeros((256, 256, 3), dtype=np.uint8)
            mask[:, :, 1:] = img_resized[:, :, 1:]  # Keep G and B channels
            height, width, c = mask.shape
            return QImage(mask.data, width, height, 3 * width, QImage.Format_RGB888)

    def mask_process(self, img_path):
        # Görüntüyü oku ve kontrol et
        real_mask_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if real_mask_image is None:
            print(f"Hata: Görüntü okunamadı - {img_path}")
            return

        # Görüntüyü numpy array'e çevir ve tip kontrolü yap
        real_mask_image = np.array(real_mask_image)

        # Debug için görüntü bilgilerini yazdır
        # print(f"Görüntü şekli: {real_mask_image.shape}")
        # print(f"Görüntü tipi: {real_mask_image.dtype}")
        # print(f"Benzersiz değerler: {np.unique(real_mask_image)}")

        try:
            # Etiketleri oluştur
            label1 = (real_mask_image == 1).astype(np.uint8) * 255
            label2 = (real_mask_image == 2).astype(np.uint8) * 255
            label3 = (real_mask_image == 3).astype(np.uint8) * 255

            # Tek kanallı görüntüleri üç kanallı hale getir
            color_label1 = cv2.cvtColor(label1, cv2.COLOR_GRAY2BGR)
            color_label2 = cv2.cvtColor(label2, cv2.COLOR_GRAY2BGR)
            color_label3 = cv2.cvtColor(label3, cv2.COLOR_GRAY2BGR)

            # Maskeleri oluştur
            cell_labels = cv2.add(color_label1, color_label2)
            cell_labels = cv2.resize(cell_labels, (256, 256))

            branch_labels = color_label3
            branch_labels = cv2.resize(branch_labels, (256, 256))

            all_labels = cv2.add(cv2.add(color_label3, color_label1), color_label2)
            all_labels = cv2.resize(all_labels, (256, 256))

            # Maskeleri kaydet
            save_path = os.path.dirname(img_path)  # Maskeleri orijinal dosya ile aynı dizine kaydet

            cv2.imwrite(os.path.join(save_path, 'cell_mask.jpg'), cell_labels)
            cv2.imwrite(os.path.join(save_path, 'branch_mask.jpg'), branch_labels)
            cv2.imwrite(os.path.join(save_path, 'branch1.jpg'), branch_labels)
            cv2.imwrite(os.path.join(save_path, 'all_labels_mask.jpg'), all_labels)

            return cell_labels, branch_labels, all_labels

        except Exception as e:
            print(f"Hata oluştu: {str(e)}")
            print(f"Görüntü işleme sırasında bir hata oluştu: {img_path}")
            return None, None, None
    def change_white_to_color(self,image_path, new_color, output_path):

        # Resmi yükle
        image = cv2.imread(image_path)

        # Beyaz renk (RGB olarak)
        white = np.array([255, 255, 255])

        # Yeni renk (RGB olarak)
        new_color = np.array(new_color)

        # Beyaz alanları tespit et ve yeni renkle değiştir
        mask = np.all(image == white, axis=-1)
        image[mask] = new_color

        # Sonucu kaydet
        cv2.imwrite(output_path, image)

    def save_branch(self, img_path):
        image = cv2.imread(img_path)
        # RGB renk uzayına dönüştür
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # Renk kanallarını ayır
        r_channel = image_rgb[:,:,0]
        g_channel = image_rgb[:,:,1]
        b_channel = image_rgb[:,:,2]

        output_path = 'branch.jpg'  # Kaydetmek istediğiniz yol
        cv2.imwrite(output_path, g_channel)
        image = cv2.imread(output_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        # Keskinleştirme (Sharpness) işlemi
        sharp_kernel = np.array([[-1, -1, -1],
                                [-1, 11, -1],
                                [-1, -1, -1]])  # Kernel değerlerini arttırdık
        sharp_image = cv2.filter2D(gray, -1, sharp_kernel)
        # Çıktıyı kaydet
        output_path = 'branch.jpg'  # Kaydetmek istediğiniz yol
        cv2.imwrite(output_path, sharp_image)

    def save_cell(self, img_path):
        image = cv2.imread(img_path)
        # RGB renk uzayına dönüştür
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # Renk kanallarını ayır
        r_channel = image_rgb[:,:,0]
        g_channel = image_rgb[:,:,1]
        b_channel = image_rgb[:,:,2]

        output_path = 'cell.jpg'  # Kaydetmek istediğiniz yol
        cv2.imwrite(output_path, g_channel)
        image = cv2.imread(output_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        # Keskinleştirme (Sharpness) işlemi
        sharp_kernel = np.array([[-1, -1, -1],
                                [-1, 11, -1],
                                [-1, -1, -1]])  # Kernel değerlerini arttırdık
        sharp_image = cv2.filter2D(gray, -1, sharp_kernel)
        # Çıktıyı kaydet
        output_path = 'cell.jpg'  # Kaydetmek istediğiniz yol
        cv2.imwrite(output_path, sharp_image)

    def merge_side_by_side(self,image1_path, image2_path, output_path):
        # Open the images
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)

        # Get the dimensions of both images
        width1, height1 = image1.size
        width2, height2 = image2.size

        # Calculate the dimensions of the new image (assuming heights are the same)
        new_width = width1 + width2
        new_height = max(height1, height2)

        # Create a new blank image
        new_image = Image.new('RGB', (new_width, new_height))

        # Paste the first image at the top-left corner of the new image
        new_image.paste(image1, (0, 0))

        # Paste the second image starting from the width of the first image to the right side of the new image
        new_image.paste(image2, (width1, 0))

        # Save the new image
        new_image.save(output_path)

    def dice_coef(y_true, y_pred):
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        print((2.0 * intersection + 1e-5) / (union + 1e-5))
        return (2.0 * intersection + 1e-5) / (union + 1e-5)

    def iou_score(y_true, y_pred):
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        return (intersection + 1e-5) / (union + 1e-5)



    @atexit.register
    def close_signal_triggered(self):
        for path in self.predicted_path:
            if path != "":
                shutil.rmtree(path)

    def close_signal_triggered2(self):
        self.delete_jpg_files_in_current_folder()
        for path in self.predicted_path:
            if path != "":
                shutil.rmtree(path)
        self.close()

    def delete_jpg_files_in_current_folder(self):
        # Mevcut çalışma dizinini al
        current_directory = os.getcwd()

        # Mevcut dizindeki tüm dosyaları listele
        files = os.listdir(current_directory)

        # Her bir dosya için kontrol et ve .jpg uzantılıysa sil
        for file_name in files:
            if file_name.endswith(".jpg"):
                file_path = os.path.join(current_directory, file_name)
                os.remove(file_path)
            if file_name.endswith(".png"):
                file_path = os.path.join(current_directory, file_name)
                os.remove(file_path)

    def out_log(self):
        if self.account.checkBox_staysgined.isChecked():
            self.main_window.close()
            self.account.show()
        else:
            self.main_window.close()
            self.account.show()
            self.account.lin_user_name.clear()
            self.account.lin_password.clear()

    def connect_main_window(self):
        self.guestORuser=1
        is_correct,self.kullanici_adi, self.parola = self.account._login_()
        print(is_correct)
        if (self.account.kullanici_adi=="") or (self.account.parola=="") or (self.account.kullanici_adi=="" and self.account.parola==""):
            QMessageBox.information(self,"Information","Required fields cannot be empty...")
        else:
            if is_correct:
                self.main_window.show()  # show main window
                self.account.close()
            else:
                QMessageBox.critical(self,"ERROR","Wrong username or password..... ")

    def open_dialog(self):
        self.close_signal_triggered()
        self.predicted_path =list()
        self.dir_path=QFileDialog.getExistingDirectory(self,"Choose Directory")
        print(self.dir_path)
        self.scrollArea_2.show()
        if (self.dir_path !=''):
            self.dir_list = os.listdir(self.dir_path)
            self.scrollArea_2.show()
            self.count=[]
            self.rontgen=[]
            self.a=[]
            self.btn=[]
            for i in range(len(self.dir_list)):
                self.get_buttons(i)

            self.main_window.setGeometry(QtCore.QRect(40,30,1329, 987))

        else:
            pass

    def get_buttons(self, i, is_train=False):
        # Buton ve Label için dikey layout oluştur
        layout = QVBoxLayout()
        layout.setSpacing(5)  # Buton ve label arasında 5 piksel boşluk bırak

        # Yeni bir buton oluşturur
        button = QPushButton()
        button.setFixedSize(181, 161)  # Buton boyutunu sabit tut

        # Butonu btn listesine ekle (hatayı önlemek için)
        self.btn.append(button)

        # Buton stilini ayarlar, beyaz arka planı kaldır
        button.setStyleSheet("QPushButton"
                             "{"
                             "border:6px solid rgb(149, 190, 255);"
                             "background-color: rgba(0, 0, 0, 0);"  # Şeffaf arka plan
                             "border-radius:20px;"
                             "}"
                             "QPushButton:hover"
                             "{"
                             "background-color: rgb(149, 190, 255);"
                             "}"
                             "QPushButton::pressed"
                             "{"
                             "background-color : rgba(0, 0, 0, 0);"  # Şeffaf arka plan
                             "}"
                             )

        # Butona resim dosyasını ekler
        image_path = f"{self.dir_path}/{self.dir_list[i]}"
        button.clicked.connect(partial(self.open_patient_file, button))
        button.setIcon(QIcon(self.to_pixmap(image_path)))
        button.setIconSize(QtCore.QSize(130, 130))  # Icon boyutunu ayarlar

        # Resmin adını göstermek için bir QLabel oluştur
        label = QLabel(
            os.path.basename(self.dir_list[i]))  # Sadece dosya adını almak için os.path.basename kullanıyoruz
        label.setAlignment(Qt.AlignCenter)  # Label'i ortala
        label.setStyleSheet(
            "color: black; font-size: 12px; background-color: rgba(0, 0, 0, 0);")  # Siyah renkte yazı ve şeffaf arka plan

        # Label'in tasarımı buton tasarımını etkilemesin
        label.setFixedHeight(20)

        # Buton ve label'i layout'a ekle
        layout.addWidget(button)
        layout.addWidget(label)

        # Buton ve label'i içeren layout'u scrollArea_2'ye ekle
        widget = QWidget()
        widget.setLayout(layout)
        widget.setStyleSheet("background-color: rgba(0, 0, 0, 0);")  # Şeffaf arka plan
        widget.setFixedSize(181 + 20, 181 + 20 + 5)  # Sabit boyut: buton + label yüksekliği + 5 piksel boşluk
        self.horizontalLayout_7.addWidget(widget)

        # Predicted path'i güncelle
        self.predicted_path.append("")

    def to_pixmap(self, image_path):
        # Convert the image at the given path to a QPixmap
        pixmap = QPixmap(image_path)
        return pixmap

    def open_patient_file(self,obj):
        indx = self.btn.index(obj)
        print(indx)
        x = len(self.btn)
        self.count.append(x)
        print(self.btn)
        for i in range(0,self.count[0]):
            self.btn[i].setStyleSheet("QPushButton"
                                         "{"
                                         "border:6px solid rgb(149, 190, 255);"
                                         "background-color:#fff;"
                                         "border-radius:20px;"
                                         "}"
                                         "QPushButton:hover"
                                         "{"
                                         "background-color: rgb(149, 190, 255);"
                                         "}"
                                         "QPushButton::pressed"
                                         "{"
                                         "background-color : #fff"
                                         "}"
                                         )

        self.btn[indx].setStyleSheet("QPushButton"
                                  "{"
                                  "border:6px solid rgb(62, 228, 45);"
                                  "background-color:#fff;"
                                  "border-radius:20px;"
                                  "}"
                                  "QPushButton:hover"
                                  "{"
                                  "background-color: rgb(62, 228, 45);"
                                  "}"
                                  "QPushButton::pressed"
                                  "{"
                                  "background-color : #fff"
                                  "}"
                                  )

        self.label_5.setVisible(True)
        self.label_3.setVisible(True)
        # Extract the image file name inside the method
        img_filename = os.path.basename(self.dir_list[indx])

        # Define the desired file extension (e.g., "jpg" or "png")
        desired_extension = "jpg"

        # Modify the file extension if it's not the desired one
        if not img_filename.lower().endswith(desired_extension):
            img_filename = f"{os.path.splitext(img_filename)[0]}.{desired_extension}"

        img_path = f"original/{img_filename}"
        self.predict_path = img_path


         # Extract the image file name inside the method
        img_filename = os.path.basename(self.dir_list[indx])

        # Define the desired file extension (e.g., "jpg" or "png")
        desired_extension = "png"

        # Modify the file extension if it's not the desired one
        if not img_filename.lower().endswith(desired_extension):
            img_filename = f"{os.path.splitext(img_filename)[0]}.{desired_extension}"

        mask_path = f"mask/{img_filename}"
        self.predict_mask_path = mask_path
        self.mask_process(mask_path)


        # Set the pixmap for the QLabel
        pixmap = QPixmap(img_path)
        self.imglabel.setPixmap(pixmap)
        self.imglabel.setScaledContents(True)

        self.checkBox.setVisible(True)
        self.checkBox_2.setVisible(True)
        self.predict_button.setVisible(True)

        self.overlap_button_two.setVisible(True)
        self.branch_lenght_button.setVisible(True)
        self.branch_butotn.setVisible(True)
        self.angle_buton.setVisible(True)
        self.cell_button.setVisible(True)
        self.cell_area_button.setVisible(True)
        self.tabWidget.setVisible(True)

    def cell(self):
        input_image_path = "cell.jpg"
        if os.path.exists(input_image_path):  # Dosya yolu geçerli ise
            self.show_cell(input_image_path)
        else:  # Dosya yolu geçerli değilse
            # Uyarı mesajı oluşturun
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please make a prediction first!")
            msg.setWindowTitle("Warning")
            msg.exec_()


#threshold will be added
    def cell_area(self):
        threshold_value = self.slider_cell_area.value()
        input_image_path = 'prediction_result_cell.jpg'
        if not os.path.exists(input_image_path):
            print(f"Error: Input image not found at {input_image_path}")
            return

        try:
            # Ana görüntü işleme
            img = cv2.imread(input_image_path)
            if img is None:
                raise ValueError("Failed to load input image")

            blue_channel = 255 - img[:, :, 1]
            blue_channel = self.contrast_stretching(blue_channel, 1, 100)
            mask = cv2.threshold(blue_channel, threshold_value, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            stats = cv2.connectedComponentsWithStats(mask, 8)[2]
            img_with_areas = img.copy()

            # Daireler ve metin çizimi
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i in range(1, len(stats)):
                center = (int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2),
                          int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2))
                radius = int(np.sqrt(stats[i, cv2.CC_STAT_AREA] / np.pi))
                cv2.circle(img_with_areas, center, radius, (128, 128, 0), 3)

                text = f"{stats[i, cv2.CC_STAT_AREA]}"
                text_size = cv2.getTextSize(text, font, 0.3, 1)[0]
                text_position = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)
                cv2.putText(img_with_areas, text, text_position, font, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

            # Görüntü dönüşümü ve yeniden boyutlandırma
            img_with_areas = cv2.cvtColor(img_with_areas, cv2.COLOR_BGR2RGB)
            new_size = (img_with_areas.shape[1] * 2, img_with_areas.shape[0] * 2)
            img_with_areas_resized = cv2.resize(img_with_areas, new_size)

            # Hücre parça ve alan bilgilerini işle
            try:
                cell_pice, _ = self.cell_piece_and_area_origin("cell.jpg")
                self.cell_pice = cell_pice
                if cell_pice > 1:
                    self.number_cell.setText("Cells: 1")
                elif cell_pice == 0:
                    self.number_cell.setText("Cell: 0")
                else:
                    self.number_cell.setText(f"Cells: {cell_pice}")
            except Exception as e:
                print(f"Error processing cell piece information: {e}")

            # Matplotlib figürü oluştur ve kaydet
            fig, ax = plt.subplots(figsize=(10, 10))
            fig.tight_layout()
            ax.imshow(img_with_areas_resized)
            ax.set_title('Cell Area', fontsize=50, pad=5)
            ax.axis('off')
            fig.tight_layout()

            # Görüntüyü buffer'a kaydet
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
            buf.seek(0)

            # QImage ve QPixmap oluştur
            data = buf.getvalue()
            image = QImage.fromData(data)
            if image.isNull():
                raise ValueError("Failed to create QImage from buffer")

            pixmap = QPixmap.fromImage(image)
            if pixmap.isNull():
                raise ValueError("Failed to create QPixmap from QImage")

            # QLabel'a pixmap'i ayarla
            self.label_cell_area.setPixmap(pixmap)
            self.label_cell_area.setScaledContents(True)

            self.slider_cell_area.setVisible(True)

            # self.cell_areas değişkenini güncelle
            self.cell_areas = img_with_areas_resized
            self.compare_button.setVisible(False)
            self.compare_button.setVisible(True)

            self.label_cell_area.mousePressEvent = self.cell_area_clicked

            histogram_fig = self.create_cell_area_histogram()
            if histogram_fig:
                # Histogram başarıyla oluşturuldu
                stats = self.get_cell_area_stats()
                # if stats:
                #     print(f"Ortalama hücre alanı: {stats['mean']:.2f} piksel")
                #     print(f"Toplam hücre sayısı: {stats['count']}")

            # Matplotlib figürünü kapat
            plt.close(fig)

        except Exception as e:
            print(f"Error in cell_area: {e}")
            import traceback
            traceback.print_exc()


    def cell_area_clicked(self, event):
        """Function called when label_cell_area is clicked."""
        if hasattr(self, 'cell_areas') and self.cell_areas is not None:
            # Verify cell_areas is a valid 2D or 3D array
            if isinstance(self.cell_areas, np.ndarray) and self.cell_areas.ndim in [2, 3]:
                # Display the cell area details
                plt.figure(figsize=(10, 10))
                plt.imshow(self.cell_areas, cmap='gray' if self.cell_areas.ndim == 2 else None)
                plt.title('Detailed Cell Area', fontsize=25, pad=5)
                plt.axis('off')
                plt.show()
            else:
                print("Error: cell_areas is not a valid image format.")
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Cell areas data is not in a valid image format.")
                msg.setWindowTitle("Error")
                msg.exec_()
        else:
            # Warning message if cell_areas not processed
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please process the cell area image first!")

            msg.setWindowTitle("Warning")
            msg.exec_()

    def contrast_stretching(self,img, blackT, whiteT):
        row, column = img.shape
        new_image = np.zeros((row, column), np.uint8)
        treshold_1 = blackT
        treshold_2 = whiteT
        if treshold_1 > treshold_2:
            treshold_1 = whiteT
            treshold_2 = blackT
        for l in range(row):
            for m in range(column):
                if treshold_1 <= img[l, m] <= treshold_2:
                    new_image[l, m] = round(((img[l, m] - treshold_1) / (treshold_2 - treshold_1)) * 255)
                elif img[l, m] < treshold_1:
                    new_image[l, m] = 0
                elif img[l, m] > treshold_2:
                    new_image[l, m] = 255
        return new_image

    # threshold eklenecek
    def cell_piece_and_area(self, image_path):
        threshold_value = self.slider_cell_area.value()
        # Resmi oku
        resim = cv2.imread(image_path)

        # Gri tonlamaya dönüştür
        gri_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)

        # Eşikleme uygula
        _, esikli_resim = cv2.threshold(gri_resim, threshold_value, 255, cv2.THRESH_BINARY)

        # Konturları bul
        konturler, _ = cv2.findContours(esikli_resim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Başlangıç değerlerini ayarla
        toplam_piksel = 0
        cell_pice = len(konturler)

        # Her bir kontur için alanı hesapla
        for kontur in konturler:
            alan = cv2.contourArea(kontur)
            toplam_piksel += alan

        # Sonuçları döndür
        return cell_pice, toplam_piksel

    # threshold eklenecek
    def cell_piece_and_area_origin(self, image_path):
        # Resmi oku
        threshold_value = self.slider_cell_area.value()
        resim = cv2.imread(image_path)

        # Gri tonlamaya dönüştür
        gri_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)

        # Eşikleme uygula
        _, esikli_resim = cv2.threshold(gri_resim, threshold_value, 255, cv2.THRESH_BINARY)

        # Konturları bul
        konturler, _ = cv2.findContours(esikli_resim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Başlangıç değerlerini ayarla
        toplam_piksel = 0
        cell_pice = int((len(konturler) / 2) + 2)

        # Her bir kontur için alanı hesapla
        for kontur in konturler:
            alan = cv2.contourArea(kontur)
            toplam_piksel += alan

        # Sonuçları döndür
        return cell_pice, toplam_piksel

    # threshold eklenecek
    def show_cell(self, image_path):
        threshold_value = self.slider_cell.value()
        self.current_image = {
            'esikli_resim': None,
            'figure': None
        }

        # Görüntü işleme işlemleri
        resim = cv2.imread(image_path)
        gri_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
        _, esikli_resim = cv2.threshold(gri_resim, threshold_value, 255, cv2.THRESH_BINARY)
        konturler, _ = cv2.findContours(esikli_resim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Toplam hücre parçalarını topla ve kaydet
        self.cell_pieces = [cv2.contourArea(kontur) for kontur in konturler]


        # Görüntüyü saklama
        cv2.imwrite("esikli_Resim.jpg", esikli_resim)
        self.current_image['esikli_resim'] = esikli_resim

        # Matplotlib grafiği oluşturma
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.tight_layout()

        ax.imshow(esikli_resim, cmap='gray')
        ax.set_title('Cell', fontsize=50, pad=5)  # Title'ı biraz yukarı taşıdık
        ax.axis('off')
        self.current_image['figure'] = fig

        # Grafiği bir buffer'a kaydet
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)  # transparent=True ekledik
        buf.seek(0)

        # QImage oluşturma ve QLabel'de gösterme
        data = buf.getvalue()
        image = QImage.fromData(data)
        pixmap = QPixmap.fromImage(image)

        self.label_cell.setPixmap(pixmap)
        self.label_cell.setScaledContents(True)

        self.slider_cell.setVisible(True)

        # Tıklanabilir yapmak için mousePressEvent'i bağlayın
        self.label_cell.mousePressEvent = self.label_clicked
        self.create_cell_count_histogram()

        # Figure'ı kapat
        plt.close(fig)



    def label_clicked(self, event):
        print("label_clicked is clicked :) ")
        """Label'a tıklandığında çağrılacak fonksiyon"""
        if hasattr(self, 'current_image') and self.current_image['esikli_resim'] is not None:
            # Yeni bir figure oluştur
            plt.figure(figsize=(10, 10))
            plt.imshow(self.current_image['esikli_resim'], cmap='gray')
            plt.title('Cell', fontsize=25, pad=5)
            plt.axis('off')
            plt.show()

    def create_cell_count_histogram(self):
        """
        Generates a cell count histogram and stores it for later display on click.

        Returns:
            matplotlib.figure.Figure: The generated histogram figure
        """
        try:
            # Ensure there's data available for cell count (pieces)
            if not hasattr(self, 'cell_pieces') or self.cell_pieces is None or len(self.cell_pieces) == 0:
                raise ValueError("No cell count data available")

            # Generate the histogram
            plt.clf()  # Clear any previous plot
            fig = plt.figure(figsize=(10, 10))
            plt.hist(self.cell_pieces, bins=10, color='skyblue', edgecolor='black')

            plt.title('Cell Count Histogram', fontsize=40, pad=10)
            plt.xlabel('Cell Count', fontsize=30)
            plt.ylabel('Frequency', fontsize=30)
            plt.grid(True, alpha=0.3)

            # Calculate statistics
            mean_count = np.mean(self.cell_pieces)
            std_count = np.std(self.cell_pieces)
            plt.axvline(mean_count, color='red', linestyle='dashed', linewidth=1)
            plt.text(mean_count + 5, plt.gca().get_ylim()[1] * 0.9,
                     f'Mean: {mean_count:.1f}\nStd: {std_count:.1f}',
                     bbox=dict(facecolor='white', alpha=0.8))

            # Display and save the figure
            plt.tight_layout()

            # Place the figure on the QLabel
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', transparent=True)
            buf.seek(0)

            image = QImage.fromData(buf.getvalue())
            pixmap = QPixmap.fromImage(image)
            self.label_cell_piece_histogram.setPixmap(pixmap)
            self.label_cell_piece_histogram.setScaledContents(True)

            # Store the figure and statistics for later use
            self.cell_piece_histogram_figure = fig
            self.cell_count_stats = {
                'mean': mean_count,
                'std': std_count,
                'min': np.min(self.cell_pieces),
                'max': np.max(self.cell_pieces),
                'count': len(self.cell_pieces)
            }

            # Connect click event to the label
            self.label_cell_piece_histogram.mousePressEvent = self.cell_count_histogram_label_clicked
            return fig

        except Exception as e:
            print(f"Error creating histogram: {str(e)}")
            QMessageBox.warning(self, "Warning", "No data available for cell count histogram.")
            return None
        finally:
            plt.close('all')  # Close all open figures

    def cell_count_histogram_label_clicked(self, event):
        """Function called when label_cell_piece_histogram is clicked."""
        if hasattr(self, 'cell_pieces') and self.cell_pieces is not None:
            # Generate a new figure for display
            plt.figure(figsize=(10, 10))
            plt.hist(self.cell_pieces, bins=10, color='skyblue', edgecolor='black')
            plt.title('Detailed Cell Count Histogram', fontsize=50, pad=10)
            plt.xlabel('Cell Count', fontsize=30)
            plt.ylabel('Frequency', fontsize=30)
            plt.grid(True, alpha=0.3)

            # Display mean and standard deviation
            mean_count = np.mean(self.cell_pieces)
            std_count = np.std(self.cell_pieces)
            plt.axvline(mean_count, color='red', linestyle='dashed', linewidth=1)
            plt.text(mean_count + 5, plt.gca().get_ylim()[1] * 0.9,
                     f'Mean: {mean_count:.1f}\nStd: {std_count:.1f}',
                     bbox=dict(facecolor='white', alpha=0.8))

            plt.show()  # Show the figure
        else:
            # Show a warning message if histogram data is not available
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("No cell count histogram available to display.")
            msg.setWindowTitle("Warning")
            msg.exec_()

    def get_cell_count_stats(self):
        """
        Returns the cell count statistics.
        """
        if hasattr(self, 'cell_count_stats'):
            return self.cell_count_stats
        return None

    def mask_preproccess(self,image_path):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # Her bir label'ı ayır
        label1 = (image == 1).astype(np.uint8) * 255  # Örneğin, label 1
        label2 = (image == 2).astype(np.uint8) * 255  # Örneğin, label 2
        label3 = (image == 3).astype(np.uint8) * 255  # Örneğin, label 3

        # Labelları beyaz, arka planı siyah yap
        color_label1 = cv2.merge([label1, label1, label1])  # Labelları beyaz yap
        color_label2 = cv2.merge([label2, label2, label2])
        color_label3 = cv2.merge([label3, label3, label3])

        cell_labels = color_label1 + color_label2
        cell_labels = cv2.resize(cell_labels, (256, 256))
        branch_labels =  color_label3
        branch_labels = cv2.resize(branch_labels, (256, 256))

        output_path = 'cell_mask.jpg'  # Kaydetmek istediğiniz yol
        cv2.imwrite(output_path, cell_labels)
        output_path = 'branch_mask.jpg'  # Kaydetmek istediğiniz yol
        cv2.imwrite(output_path, branch_labels)

    def branch(self):
        # Get the threshold value from the slider
        threshold_value = self.slider_branch.value()
        input_image_path = "branch.jpg"

        if os.path.exists(input_image_path):
            # Load the branch image
            resim = cv2.imread(input_image_path)

            # Convert to grayscale
            gri_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)

            # Apply threshold
            _, esikli_resim = cv2.threshold(gri_resim, threshold_value, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(esikli_resim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.branch_contours = contours  # Store contours for histogram

            # Draw contours on a copy of the image
            contoured_image = resim.copy()
            cv2.drawContours(contoured_image, contours, -1, (0, 255, 0), 2)  # Green contours

            # Display the contoured image
            contoured_image_rgb = cv2.cvtColor(contoured_image, cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots(figsize=(10, 10))
            fig.tight_layout()

            ax.imshow(contoured_image_rgb)
            ax.set_title('Branch', fontsize=50, pad=5)
            ax.axis('off')

            # Save to buffer and show in QLabel
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
            buf.seek(0)
            image = QImage.fromData(buf.getvalue())
            pixmap = QPixmap.fromImage(image)

            self.label_branch.setPixmap(pixmap)
            self.label_branch.setScaledContents(True)

            self.slider_branch.setVisible(True)
            # Add click event
            self.label_branch.mousePressEvent = self.branch_clicked
            self.show_branch_piece_histogram()

            plt.close(fig)  # Close Matplotlib figure

        else:
            # Show warning if image not found
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please make a prediction first!")
            msg.setWindowTitle("Warning")
            msg.exec_()

    def branch_clicked(self, event):
        """Branch label'ına tıklandığında çağrılacak fonksiyon"""
        if hasattr(self, 'branch_image'):
            # BGR'dan RGB'ye dönüştür
            rgb_image = cv2.cvtColor(self.branch_image, cv2.COLOR_BGR2RGB)

            # Yeni bir figure oluştur
            plt.figure(figsize=(10, 10))
            plt.imshow(rgb_image)
            plt.title('Branch', fontsize=25, pad=5)
            plt.axis('off')
            plt.show()

    def show_branch_piece_histogram(self):
        """
        Generates a histogram of branch areas based on contours.
        """
        try:
            if not hasattr(self, 'branch_contours') or not self.branch_contours:
                raise ValueError("No branch piece data available")

            # Compute branch areas from contours
            branch_areas = [cv2.contourArea(contour) for contour in self.branch_contours]

            # Calculate statistics
            mean_area = np.mean(branch_areas)
            std_area = np.std(branch_areas)
            max_area = np.max(branch_areas)

            # Create histogram plot
            plt.clf()
            fig = plt.figure(figsize=(10, 10))
            plt.hist(branch_areas, bins=10, color='skyblue', edgecolor='black')
            plt.title('Branch Area Histogram', fontsize=40, pad=10)
            plt.xlabel('Area (pixels)', fontsize=30)
            plt.ylabel('Frequency', fontsize=30)
            plt.grid(True, alpha=0.3)

            # Add statistical information
            plt.axvline(mean_area, color='red', linestyle='dashed', linewidth=1)
            plt.text(mean_area + 5, plt.gca().get_ylim()[1] * 0.9,
                     f'Mean: {mean_area:.1f}\nStd: {std_area:.1f}\nMax: {max_area:.1f}',
                     bbox=dict(facecolor='white', alpha=0.8))

            # Save figure to buffer for QLabel
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', transparent=True)
            buf.seek(0)

            # Display in QLabel
            image = QImage.fromData(buf.getvalue())
            pixmap = QPixmap.fromImage(image)
            self.label_branch_piece_histogram.setPixmap(pixmap)
            self.label_branch_piece_histogram.setScaledContents(True)

            # Attach click event
            self.label_branch_piece_histogram.mousePressEvent = self.label_branch_piece_histogram_clicked

            # Store the histogram for later use
            self.branch_piece_histogram_data = branch_areas

        except Exception as e:
            print(f"Error generating branch piece histogram: {e}")
            QMessageBox.warning(self, "Warning", "No data available for branch piece histogram.")
        finally:
            plt.close('all')

    def label_branch_piece_histogram_clicked(self, event):
        """Event handler for clicking on the branch piece histogram label."""
        if hasattr(self, 'branch_piece_histogram_data') and self.branch_piece_histogram_data:
            plt.figure(figsize=(10, 10))
            plt.hist(self.branch_piece_histogram_data, bins=10, color='skyblue', edgecolor='black')
            plt.title('Detailed Branch Count Histogram', fontsize=50, pad=10)
            plt.xlabel('Area (pixels)', fontsize=30)
            plt.ylabel('Frequency', fontsize=30)
            plt.grid(True, alpha=0.3)

            plt.show()
        else:
            QMessageBox.warning(self, "Warning", "No branch piece histogram available to display.")



    def branch_lenght(self):
        threshold_value = self.slider_branch_length.value()

        input_image_path = "branch.jpg"
        if os.path.exists(input_image_path):
            # Resmi yükle ve işlemleri yap
            image = cv2.imread("branch_mask.jpg")
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Save contours for histogram usage
            self.branch_length_contours = contours  # Kontur verilerini kaydet

            # Toplam piksel sayısını hesapla
            total_pixels = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                total_pixels += area

            self.branch_areas_mask = total_pixels
            self.branch_pice_mask = len(contours)
            self.find_branch_lenght_and_size()

            # Konturları çiz
            self.contoured_image = image.copy()  # Tıklama için resmi saklayalım
            cv2.drawContours(self.contoured_image, contours, -1, (0, 255, 0), 2)

            # Matplotlib grafiği oluştur
            fig, ax = plt.subplots(figsize=(10, 10))
            fig.tight_layout()

            # BGR'dan RGB'ye dönüştür
            rgb_image = cv2.cvtColor(self.contoured_image, cv2.COLOR_BGR2RGB)

            ax.imshow(rgb_image)
            ax.set_title('Branch Length', fontsize=50, pad=5)
            ax.axis('off')

            # Grafiği buffer'a kaydet
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
            buf.seek(0)

            # QImage ve QPixmap oluştur
            data = buf.getvalue()
            image = QImage.fromData(data)
            pixmap = QPixmap.fromImage(image)

            # Label'a pixmap'i set et
            self.label_branch_length.setPixmap(pixmap)
            self.label_branch_length.setScaledContents(True)

            self.slider_branch_length.setVisible(True)

            self.create_branch_length_histogram()

            self.label_branch_length.mousePressEvent = self.branch_length_clicked

            # Figure'ı kapat
            plt.close(fig)

            # Resmi kaydet
            cv2.imwrite("countered_image.png", self.contoured_image)

        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please make a prediction first!")
            msg.setWindowTitle("Warning")
            msg.exec_()

    def branch_length_clicked(self, event):
        """Branch length label'ına tıklandığında çağrılacak fonksiyon"""
        if hasattr(self, 'contoured_image'):
            # BGR'dan RGB'ye dönüştür
            rgb_image = cv2.cvtColor(self.contoured_image, cv2.COLOR_BGR2RGB)

            # Yeni bir figure oluştur
            plt.figure(figsize=(10, 10))
            plt.imshow(rgb_image)
            plt.title('Branch Length', fontsize=25, pad=5)
            plt.axis('off')
            plt.show()
    def show_branch_length_histogram(self):
        """
        Generates and displays a histogram of branch lengths based on contour data.
        """
        try:
            if hasattr(self, 'branch_lengths') and self.branch_lengths:
                # Create a new figure for the histogram
                plt.figure(figsize=(10, 10))
                plt.hist(self.branch_lengths, bins=20, color='skyblue', edgecolor='black')
                plt.title('Branch Count Histogram', fontsize=50, pad=5)
                plt.xlabel('Size', fontsize=30)
                plt.ylabel('Frequency', fontsize=30)
                plt.grid(True, alpha=0.3)

                # Store the current figure
                self.branch_lenghts_histogram_figure = plt.gcf()
                plt.show()

        except Exception as e:
            print(f"Error generating branch length histogram: {e}")
            QMessageBox.warning(self, "Warning", "No data available for branch length histogram.")

        finally:
            plt.close('all')  # Close all open figures

    def branch_length_histogram_clicked(self, event):
        """
        Event handler for clicking on the branch length histogram label.
        Displays the histogram plot in a separate window.
        """
        if hasattr(self, 'branch_lengths') and self.branch_lengths:
            # Create a new figure for the histogram
            plt.figure(figsize=(10, 10))
            plt.hist(self.branch_pice, bins=20, color='skyblue', edgecolor='black')
            plt.title('Branch Length Histogram', fontsize=40, pad=5)
            plt.xlabel('Size', fontsize=30)
            plt.ylabel('Frequency', fontsize=30)
            plt.grid(True, alpha=0.3)

            # Store the current figure
            self.branch_length_histogram_figure = plt.gcf()
            plt.show()
        else:
            QMessageBox.warning(self, "Warning", "No branch length histogram available to display.")

    #threshold eklenecek
    def find_branch_lenght_and_size(self):
        threshold_value = self.slider_branch_length.value()

        # Resmi yükle
        resim = cv2.imread("branch.jpg")

        # Gri tonlama
        gri_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)

        # Eşikleme
        _, esikli_resim = cv2.threshold(gri_resim, threshold_value, 255, cv2.THRESH_BINARY)

        # Kontürleri bul
        konturler, _ = cv2.findContours(esikli_resim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Toplam piksel sayısını hesapla
        toplam_piksel = 0

        # Her hücrenin piksel sayısını ve toplamı hesapla
        for kontur in konturler:
            # Hücre alanı hesapla
            alan = cv2.contourArea(kontur)
            # Toplam piksele ekle
            toplam_piksel += alan

        self.branch_pice = int(len(konturler) /8)
        self.branch_areas = toplam_piksel
        if self.branch_pice > 1:
            self.number_branch.setText(f"Branches: {self.branch_pice}")
        else:
            self.number_branch.setText(f"Branch: {self.branch_pice}")


    def show_warning(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText(message)
        msg.setWindowTitle("Warning")
        msg.exec_()

    import numbers

    def compare(self):
        try:
            # Check if data exists using safer methods
            if not hasattr(self, 'cell_pice') or self.cell_pice is None:
                self.cell()
                return
            elif not hasattr(self, 'branch_pice') or self.branch_pice is None:
                self.branch()
                return
            elif not hasattr(self, 'cell_areas') or self.cell_areas is None:
                self.cell_area()
                return
            elif not hasattr(self, 'branch_areas') or self.branch_areas is None:
                self.branch_lenght()
                return

            # Verify that arrays are valid and convert to numeric values if needed
            if not isinstance(self.cell_areas_mask, numbers.Number):
                print("Warning: cell_areas_mask is not a number:", type(self.cell_areas_mask))
                self.cell_areas_mask = float(np.mean(self.cell_areas_mask)) if isinstance(self.cell_areas_mask,
                                                                                          np.ndarray) else 0

            if not isinstance(self.branch_areas_mask, numbers.Number):
                print("Warning: branch_areas_mask is not a number:", type(self.branch_areas_mask))
                self.branch_areas_mask = float(np.mean(self.branch_areas_mask)) if isinstance(self.branch_areas_mask,
                                                                                              np.ndarray) else 0

            if not isinstance(self.cell_areas, numbers.Number):
                print("Warning: cell_areas is not a number:", type(self.cell_areas))
                self.cell_areas = float(np.mean(self.cell_areas)) if isinstance(self.cell_areas, np.ndarray) else 0

            if not isinstance(self.branch_areas, numbers.Number):
                print("Warning: branch_areas is not a number:", type(self.branch_areas))
                self.branch_areas = float(np.mean(self.branch_areas)) if isinstance(self.branch_areas,
                                                                                    np.ndarray) else 0

            # Data preparation
            labels_piece = ['Cell Piece', 'Branch Piece']
            values_combined_piece = [float(self.cell_pice_mask), float(self.branch_pice_mask)]
            static_values_piece = [float(self.cell_pice), float(self.branch_pice)]

            labels_area = ['Cell Area', 'Branch Length']
            values_combined_area = [float(self.cell_areas_mask), float(self.branch_areas_mask)]
            static_values_area = [float(self.cell_areas), float(self.branch_areas)]

            # Create figures
            fig_piece = plt.figure(figsize=(10, 10))
            fig_piece.patch.set_facecolor('none')
            ax_piece = fig_piece.add_subplot(111)
            ax_piece.set_facecolor('white')

            fig_area = plt.figure(figsize=(10, 10))
            fig_area.patch.set_facecolor('none')
            ax_area = fig_area.add_subplot(111)
            ax_area.set_facecolor('white')

            # Plot piece comparison
            bar_width = 0.35
            index = np.arange(len(labels_piece))
            ax_piece.bar(index, values_combined_piece, bar_width, label='Mask Values', color='blue')
            ax_piece.bar(index + bar_width, static_values_piece, bar_width, label='Predict Values', color='red')

            diff_percents_piece = []
            for i, value in enumerate(values_combined_piece):
                if value != 0:
                    diff_percent = ((static_values_piece[i] - value) / value) * 100
                else:
                    diff_percent = 0
                diff_percents_piece.append(f'{diff_percent:.2f}%')
                ax_piece.text(i, value, f'{value:.1f}', ha='center', va='bottom')
                ax_piece.text(i + bar_width, static_values_piece[i], f'{static_values_piece[i]:.1f}', ha='center',
                              va='bottom')

            ax_piece.set_xticks(index + bar_width / 2)
            ax_piece.set_xticklabels([f'{label}\n({diff})' for label, diff in zip(labels_piece, diff_percents_piece)])
            ax_piece.set_title('Cell and Branch Number Comparison', fontsize=25, pad=15)
            ax_piece.legend()

            # Plot area comparison
            ax_area.bar(index, values_combined_area, bar_width, label='Mask Values', color='blue')
            ax_area.bar(index + bar_width, static_values_area, bar_width, label='Predict Values', color='red')

            diff_percents_area = []
            for i, value in enumerate(values_combined_area):
                if value != 0:
                    diff_percent = ((static_values_area[i] - value) / value) * 100
                else:
                    diff_percent = 0
                diff_percents_area.append(f'{diff_percent:.2f}%')
                ax_area.text(i, value, f'{value:.1f}', ha='center', va='bottom')
                ax_area.text(i + bar_width, static_values_area[i], f'{static_values_area[i]:.1f}', ha='center',
                             va='bottom')

            ax_area.set_xticks(index + bar_width / 2)
            ax_area.set_xticklabels([f'{label}\n({diff})' for label, diff in zip(labels_area, diff_percents_area)])
            ax_area.set_title('Cell and Branch Area Comparison', fontsize=25, pad=15)
            ax_area.legend()

            # Save figures to buffers and display on labels
            buf_piece = io.BytesIO()
            fig_piece.savefig(buf_piece, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
            buf_piece.seek(0)
            image_piece = QImage.fromData(buf_piece.getvalue())
            pixmap_piece = QPixmap.fromImage(image_piece)
            self.label_cell_vs_branch_number.setPixmap(pixmap_piece)
            self.label_cell_vs_branch_number.setScaledContents(True)

            buf_area = io.BytesIO()
            fig_area.savefig(buf_area, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
            buf_area.seek(0)
            image_area = QImage.fromData(buf_area.getvalue())
            pixmap_area = QPixmap.fromImage(image_area)
            self.label_cell_vs_branch_area.setPixmap(pixmap_area)
            self.label_cell_vs_branch_area.setScaledContents(True)

            # Close figures
            plt.close(fig_piece)
            plt.close(fig_area)

        except Exception as e:
            print(f"Error in compare function: {e}")
            self.show_warning(str(e))

    #threshold eklenecek.
    def angle(self):
        threshold_value = self.slider_angles_each_branch.value()

        image_path = "countered_image.png"

        if os.path.exists(image_path):
            # Mevcut görüntü işleme kodları
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
            skeleton = skeletonize(binary_image // 255)

            def find_endpoints(skeleton):
                endpoints = []
                for y in range(1, skeleton.shape[0] - 1):
                    for x in range(1, skeleton.shape[1] - 1):
                        if skeleton[y, x] == 1:
                            neighbors = skeleton[y - 1:y + 2, x - 1:x + 2]
                            if np.sum(neighbors) == 2:
                                endpoints.append((x, y))
                return endpoints

            endpoints = find_endpoints(skeleton)

            def dfs(skeleton, x, y, visited):
                stack = [(x, y)]
                branch = []
                while stack:
                    cx, cy = stack.pop()
                    if (cx, cy) not in visited:
                        visited.add((cx, cy))
                        branch.append((cx, cy))
                        for nx in range(cx - 1, cx + 2):
                            for ny in range(cy - 1, cy + 2):
                                if (nx, ny) != (cx, cy) and skeleton[ny, nx] == 1:
                                    stack.append((nx, ny))
                return branch

            visited = set()
            branches = []
            for y in range(skeleton.shape[0]):
                for x in range(skeleton.shape[1]):
                    if skeleton[y, x] == 1 and (x, y) not in visited:
                        branch = dfs(skeleton, x, y, visited)
                        if len(branch) > 1:
                            branches.append(branch)

            def calculate_angle(p1, p2):
                delta_y = p2[1] - p1[1]
                delta_x = p2[0] - p1[0]
                angle = np.arctan2(delta_y, delta_x) * 180 / np.pi
                return angle

            branch_angles = []
            for branch in branches:
                branch_endpoints = [p for p in branch if p in endpoints]
                for i in range(len(branch_endpoints)):
                    for j in range(i + 1, len(branch_endpoints)):
                        angle = calculate_angle(branch_endpoints[i], branch_endpoints[j])
                        branch_angles.append((branch_endpoints[i], branch_endpoints[j], angle))

            def generate_colors(num_colors):
                hues = np.linspace(0, 1, num_colors, endpoint=False)
                colors = [hsv_to_rgb([hue, 1, 1]) for hue in hues]
                return colors

            colors = generate_colors(len(branch_angles))

            # 1. Grafik: Endpoints
            fig_endpoints, ax = plt.subplots(figsize=(10, 10))
            fig_endpoints.tight_layout()

            # Görüntüyü göster
            ax.imshow(skeleton, cmap='gray')
            ax.set_title("Endpoints", fontsize=50, pad=5, backgroundcolor='none')  # Title background'ı transparent

            # Endpoints noktalarını çiz
            for (x, y) in endpoints:
                ax.plot(x, y, 'ro')

            # Eksenleri ve çerçeveyi tamamen kaldır
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # Grafiği QLabel için hazırla
            buf_endpoints = io.BytesIO()
            fig_endpoints.savefig(buf_endpoints, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
            buf_endpoints.seek(0)
            data_endpoints = buf_endpoints.getvalue()
            image_endpoints = QImage.fromData(data_endpoints)
            pixmap_endpoints = QPixmap.fromImage(image_endpoints)
            plt.close(fig_endpoints)

            self.label_endpoints.setPixmap(pixmap_endpoints)
            self.label_endpoints.setScaledContents(True)

            # Endpoints grafiğini sakla
            self.endpoints_data = {
                'skeleton': skeleton,
                'endpoints': endpoints
            }

            fig_angles, ax = plt.subplots(figsize=(10, 10))
            fig_angles.tight_layout()

            ax.imshow(skeleton, cmap='gray')
            ax.set_title("Angles-Each Branch", fontsize=50, pad=5,
                         backgroundcolor='none')  # Title background'ı transparent

            # Endpoints noktalarını çiz
            for (x, y) in endpoints:
                ax.plot(x, y, 'ro')

            # Eksenleri ve çerçeveyi tamamen kaldır
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            for idx, (p1, p2, angle) in enumerate(branch_angles):
                x1, y1 = p1
                x2, y2 = p2
                color = colors[idx]
                ax.plot([x1, x2], [y1, y2], color=color)
                ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{angle:.1f}", color=color, fontsize=8,
                        bbox=dict(facecolor='black', alpha=0.5))

            # Grafiği QLabel için hazırla
            buf_angles = io.BytesIO()
            fig_angles.savefig(buf_angles, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
            buf_angles.seek(0)
            data_angles = buf_angles.getvalue()
            image_angles = QImage.fromData(data_angles)
            pixmap_angles = QPixmap.fromImage(image_angles)
            plt.close(fig_angles)


            self.label_angles_each_branch.setPixmap(pixmap_angles)
            self.label_angles_each_branch.setScaledContents(True)

            self.slider_angles_each_branch.setVisible(True)

            self.label_angles_each_branch.mousePressEvent = self.angles_clicked

            self.label_endpoints.mousePressEvent = self.endpoints_clicked

            # Angles grafiğini sakla
            self.angles_data = {
                'skeleton': skeleton,
                'endpoints': endpoints,
                'branch_angles': branch_angles,
                'colors': colors
            }

        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please find the branch length first!")
            msg.setWindowTitle("Warning")
            msg.exec_()

    def endpoints_clicked(self, event):
        """Endpoints label'ına tıklandığında çağrılacak fonksiyon"""
        if hasattr(self, 'endpoints_data'):
            plt.figure(figsize=(10, 10))
            plt.imshow(self.endpoints_data['skeleton'], cmap='gray')
            plt.title("Endpoints", fontsize=25, pad=5)
            for (x, y) in self.endpoints_data['endpoints']:
                plt.plot(x, y, 'ro')
            plt.show()

    def angles_clicked(self, event):
        """Angles label'ına tıklandığında çağrılacak fonksiyon"""
        if hasattr(self, 'angles_data'):
            plt.figure(figsize=(10, 10))
            plt.imshow(self.angles_data['skeleton'], cmap='gray')
            plt.title("Angles within Each Branch", fontsize=25, pad=5)
            for (x, y) in self.angles_data['endpoints']:
                plt.plot(x, y, 'ro')
            for idx, (p1, p2, angle) in enumerate(self.angles_data['branch_angles']):
                x1, y1 = p1
                x2, y2 = p2
                color = self.angles_data['colors'][idx]
                plt.plot([x1, x2], [y1, y2], color=color)
                plt.text((x1 + x2) / 2, (y1 + y2) / 2, f"{angle:.1f}", color=color, fontsize=8,
                         bbox=dict(facecolor='black', alpha=0.5))
            plt.show()


if __name__ == "__main__":  # run if only script is main
    app = QApplication(sys.argv)
    main_window = Main_Window()
    main_window.show()
    sys.exit(app.exec_())
