import React, { useState, useEffect } from 'react';
import { Image as ImageIcon, X, ZoomIn, Check } from 'lucide-react';
import { ImageData } from '../types';
import { motion } from 'framer-motion';

interface ImageUploaderProps {
  onImageUpload: (imageData: ImageData) => void;
}

const ImageUploader: React.FC<ImageUploaderProps> = ({ onImageUpload }) => {
  const [images, setImages] = useState<ImageData[]>([]);
  const [selectedImage, setSelectedImage] = useState<ImageData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showModal, setShowModal] = useState(false);
  const [previewImage, setPreviewImage] = useState<ImageData | null>(null);

  // Görüntüleri yükle
  useEffect(() => {
    const loadImages = async () => {
      try {
        // Backend'den görüntü listesini al
        const response = await fetch('http://localhost:5000/get-images');
        const data = await response.json();
        
        if (!data.images) {
          throw new Error('No images returned from server');
        }

        const imageList = data.images.map((imageName: string) => {
          // imageName örneği: "002.jpg"
          const imageNumber = imageName.split('.')[0]; // "002"
          return {
            id: `img-${imageNumber}`,
            name: `Image ${imageNumber}`,
            dataUrl: `/images/original/${imageName}`, // .jpg uzantılı orijinal görüntü
            metadata: {
              width: 0,
              height: 0,
              fileType: 'image/jpeg', // jpg için mime type
              createdAt: new Date(),
            }
          };
        });

        setImages(imageList);
      } catch (error) {
        console.error('Error loading images:', error);
        setError('Failed to load images');
      }
    };

    loadImages();
  }, []);

  const handleImageSelect = (image: ImageData) => {
    // Dosya adını çıkar (örn: "002.jpg")
    const imageName = image.dataUrl.split('/').pop();
    
    setSelectedImage({
      ...image,
      name: imageName || image.name
    });
    
    onImageUpload({
      ...image,
      name: imageName || image.name
    });
  };

  const handleImagePreview = (image: ImageData) => {
    setPreviewImage(image);
    setShowModal(true);
  };

  return (
    <div className="space-y-4"> {/* spacing'i azalttık */}
      {/* Project Info Banner - More compact and vibrant */}
      <div className="bg-gradient-to-r from-rose-500 via-orange-400 to-amber-300 p-4 rounded-xl shadow-lg">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-white tracking-wide">
              PC12 Image Analysis
            </h1>
            <p className="text-white/90 text-sm font-medium">
              İzmir Katip Çelebi University
            </p>
          </div>
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            whileHover={{ scale: 1.05 }}
            className="bg-white/20 p-2 rounded-lg backdrop-blur-sm"
          >
            <ImageIcon size={24} className="text-white" />
          </motion.div>
        </div>
      </div>

      {/* Scrollable Image Gallery - More compact grid */}
      <div className="relative">
        <div className="max-h-[40vh] overflow-y-auto pr-4 rounded-xl bg-gradient-to-b from-orange-50 to-amber-50">
          <div className="grid grid-cols-3 gap-4 p-4"> {/* 3 kolonlu grid */}
            {images.map((image) => (
              <motion.div
                key={image.id}
                whileHover={{ scale: 1.02, y: -4 }}
                whileTap={{ scale: 0.98 }}
                className={`
                  group relative rounded-xl overflow-hidden
                  aspect-[4/3]
                  bg-gradient-to-br from-orange-100 via-amber-50 to-orange-100
                  transition-all duration-300 ease-in-out
                  shadow-lg
                  ${selectedImage?.id === image.id ? 'ring-2 ring-rose-500' : ''}
                `}
              >
                {/* Image Container */}
                <div className="absolute inset-0">
                  <img
                    src={image.dataUrl}
                    alt={image.name}
                    className="w-full h-full object-cover"
                    loading="lazy"
                  />
                  {/* Overlay Gradient */}
                  <div className="
                    absolute inset-0
                    bg-gradient-to-b from-rose-900/30 via-transparent to-rose-900/40
                    opacity-0 group-hover:opacity-100
                    transition-opacity duration-300
                  "/>
                </div>

                {/* Hover Actions */}
                <div className="
                  absolute inset-0 flex items-center justify-center gap-2
                  opacity-0 group-hover:opacity-100
                  transition-opacity duration-300
                ">
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleImagePreview(image);
                    }}
                    className="p-2 bg-white/90 rounded-full shadow-lg backdrop-blur-sm hover:bg-rose-50"
                  >
                    <ZoomIn className="w-4 h-4 text-rose-600" />
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleImageSelect(image);
                    }}
                    className="p-2 bg-white/90 rounded-full shadow-lg backdrop-blur-sm hover:bg-amber-50"
                  >
                    <Check className="w-4 h-4 text-amber-600" />
                  </motion.button>
                </div>

                {/* Image Name */}
                <div className="
                  absolute bottom-0 inset-x-0
                  bg-gradient-to-t from-rose-900/90 to-transparent
                  py-2
                ">
                  <p className="text-white text-xs font-medium text-center px-2 truncate">
                    {image.name}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Scroll Indicator */}
        <div className="absolute right-0 top-0 bottom-0 w-1">
          <motion.div
            className="w-full h-1/3 bg-gradient-to-b from-rose-500 to-amber-500 rounded-full opacity-50"
            animate={{
              y: ["0%", "200%"],
              opacity: [0.5, 0.2, 0.5]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "linear"
            }}
          />
        </div>
      </div>

      {/* Updated preview modal */}
      {showModal && previewImage && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-orange-950/70 p-4">
          <div className="relative max-w-2xl w-full">
            <button
              onClick={() => setShowModal(false)}
              className="absolute -top-10 right-0 p-2 text-white hover:text-orange-200 z-10"
            >
              <X className="w-6 h-6" />
            </button>
            <div className="bg-gradient-to-br from-orange-50 to-amber-50 rounded-xl overflow-hidden shadow-2xl">
              <div className="relative aspect-[4/3] bg-gradient-to-br from-orange-900/5 to-amber-900/5">
                <img
                  src={previewImage.dataUrl}
                  alt={previewImage.name}
                  className="absolute inset-0 w-full h-full object-contain"
                />
              </div>
              <div className="p-4 bg-gradient-to-r from-orange-50 to-amber-50 border-t border-orange-100">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-orange-900 truncate">
                    {previewImage.name}
                  </h3>
                  <button
                    onClick={() => {
                      handleImageSelect(previewImage);
                      setShowModal(false);
                    }}
                    className="px-4 py-2 bg-orange-500 text-white text-sm rounded-lg 
                                hover:bg-orange-600 transition-colors duration-200"
                  >
                    Select Image
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Updated selected image info */}
      {selectedImage && (
        <div className="fixed bottom-4 right-4 bg-gradient-to-r from-orange-50 to-amber-50 p-4 rounded-lg shadow-lg max-w-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <ImageIcon size={20} className="text-orange-600" />
              <span className="text-orange-900 font-medium">
                Selected: {selectedImage.name}
              </span>
            </div>
            <button
              onClick={() => setSelectedImage(null)}
              className="text-orange-400 hover:text-orange-600"
            >
              <X size={16} />
            </button>
          </div>
        </div>
      )}

      {/* Updated error message */}
      {error && (
        <div className="fixed top-4 right-4 bg-red-50 border-l-4 border-red-500 text-red-600 p-4 rounded-lg shadow-lg">
          {error}
          <button
            onClick={() => setError(null)}
            className="absolute top-2 right-2"
          >
            <X size={16} />
          </button>
        </div>
      )}
    </div>
  );
};

export default ImageUploader;