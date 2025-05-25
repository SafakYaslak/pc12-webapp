import React, { useState, useEffect } from 'react';
import { Image as ImageIcon, X, ZoomIn, Check } from 'lucide-react';
import { ImageData } from '../types';

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
    <div className="space-y-6">
      {/* Project Info Banner */}
      <div className="bg-gradient-to-r from-blue-50 to-blue-100 border-l-4 border-blue-500 p-6 rounded-xl shadow-lg">
        <h1 className="text-3xl font-semibold text-blue-800 tracking-wide">
          PC12 Image Analysis
        </h1>
        <p className="text-blue-700 mt-1 text-lg font-medium">
          İzmir Katip Çelebi University - PC12 Project
        </p>
      </div>

      {/* Scrollable Image Gallery */}
      <div className="relative">
        <div className="max-h-[70vh] overflow-y-auto pr-4 rounded-xl">
          <div className="grid grid-cols-2 gap-6 p-4">
            {images.map((image) => (
              <div
                key={image.id}
                className={`
                  group relative rounded-xl overflow-hidden
                  aspect-[16/9] // Daha geniş bir aspect ratio
                  bg-gray-100
                  transition-all duration-300 ease-in-out
                  hover:translate-y-[-4px]
                  hover:shadow-xl
                  ${selectedImage?.id === image.id ? 'ring-2 ring-blue-500' : ''}
                `}
              >
                {/* Image Container */}
                <div className="absolute inset-0 bg-gray-900"> {/* Koyu arka plan */}
                  <img
                    src={image.dataUrl}
                    alt={image.name}
                    className="w-full h-full object-cover" // object-contain yerine object-cover
                    onError={(e) => {
                      const target = e.target as HTMLImageElement;
                      target.src = '/placeholder-image.png';
                    }}
                    loading="lazy" // Lazy loading ekledik
                  />
                </div>
                
                {/* Semi-transparent overlay for better button visibility */}
                <div className="
                  absolute inset-0
                  bg-gradient-to-b from-black/20 via-transparent to-black/40
                  opacity-0 group-hover:opacity-100
                  transition-opacity duration-300
                "/>

                {/* Actions Buttons - Always visible at top center */}
                <div className="
                  absolute top-3 left-1/2 -translate-x-1/2
                  flex items-center space-x-3
                  z-20
                ">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleImagePreview(image);
                    }}
                    className="
                      p-2 bg-white/90 rounded-full
                      shadow-lg backdrop-blur-sm
                      hover:bg-blue-50
                      transition-all duration-200
                      transform hover:scale-110
                    "
                  >
                    <ZoomIn className="w-5 h-5 text-blue-600" />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleImageSelect(image);
                    }}
                    className="
                      p-2 bg-white/90 rounded-full
                      shadow-lg backdrop-blur-sm
                      hover:bg-green-50
                      transition-all duration-200
                      transform hover:scale-110
                    "
                  >
                    <Check className="w-5 h-5 text-green-600" />
                  </button>
                </div>

                {/* Image Name Overlay - Shows on hover */}
                <div className="
                  absolute bottom-0 inset-x-0
                  bg-gradient-to-t from-black/90 to-transparent
                  transform translate-y-full
                  group-hover:translate-y-0
                  transition-transform duration-300 ease-out
                  py-2
                ">
                  <p className="
                    text-white text-sm font-medium
                    text-center
                    px-3
                    truncate
                  ">
                    {image.name}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Scroll Indicator */}
        <div className="absolute right-0 top-0 bottom-0 w-1 bg-gray-200 rounded-full">
          <div className="w-full h-1/3 bg-blue-500 rounded-full opacity-50" />
        </div>
      </div>

      {/* Preview Modal */}
      {showModal && previewImage && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
          <div className="relative max-w-2xl w-full"> {/* max-w-4xl -> max-w-2xl */}
            <button
              onClick={() => setShowModal(false)}
              className="absolute -top-10 right-0 p-2 text-white hover:text-gray-300 z-10"
            >
              <X className="w-6 h-6" />
            </button>
            <div className="bg-white rounded-xl overflow-hidden shadow-2xl">
              <div className="relative aspect-[4/3] bg-gray-900"> {/* Sabit aspect ratio */}
                <img
                  src={previewImage.dataUrl}
                  alt={previewImage.name}
                  className="absolute inset-0 w-full h-full object-contain" 
                />
              </div>
              <div className="p-4 bg-gray-50 border-t">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-800 truncate">
                    {previewImage.name}
                  </h3>
                  <button
                    onClick={() => {
                      handleImageSelect(previewImage);
                      setShowModal(false);
                    }}
                    className="px-4 py-2 bg-blue-500 text-white text-sm rounded-lg 
                                hover:bg-blue-600 transition-colors duration-200"
                  >
                    Select Image
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Selected Image Info */}
      {selectedImage && (
        <div className="fixed bottom-4 right-4 bg-white p-4 rounded-lg shadow-lg max-w-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <ImageIcon size={20} className="text-blue-600" />
              <span className="text-blue-800 font-medium">
                Selected: {selectedImage.name}
              </span>
            </div>
            <button
              onClick={() => setSelectedImage(null)}
              className="text-gray-400 hover:text-gray-600"
            >
              <X size={16} />
            </button>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="fixed top-4 right-4 bg-red-50 text-red-600 p-4 rounded-lg shadow-lg">
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