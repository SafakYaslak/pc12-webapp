import React, { useState } from 'react';
import { Eye, EyeOff, SplitSquareHorizontal, Layers, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { ImageData } from '../types';

interface ComparisonViewProps {
  originalImage: ImageData | null;
  processedImage: ImageData | null;
  isProcessing: boolean;
}

const ComparisonView: React.FC<ComparisonViewProps> = ({ 
  originalImage, 
  processedImage,
  isProcessing
}) => {
  const [splitPosition, setSplitPosition] = useState(50);
  const [overlayMode, setOverlayMode] = useState(false);
  
  if (!originalImage) {
    return null;
  }
  
  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-4"
    >
      <div className="flex justify-between items-center">
        <div className="flex space-x-3">
          <motion.button 
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className={`
              relative px-4 py-2 rounded-lg font-medium text-sm
              transition-all duration-300
              ${overlayMode 
                ? 'text-gray-600 hover:text-rose-600' 
                : 'bg-gradient-to-r from-rose-500 to-amber-500 text-white shadow-lg'
              }
            `}
            onClick={() => setOverlayMode(false)}
          >
            <div className="flex items-center gap-2">
              <SplitSquareHorizontal size={16} />
              Split View
            </div>
          </motion.button>
          <motion.button 
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className={`
              relative px-4 py-2 rounded-lg font-medium text-sm
              transition-all duration-300
              ${!overlayMode 
                ? 'text-gray-600 hover:text-rose-600' 
                : 'bg-gradient-to-r from-rose-500 to-amber-500 text-white shadow-lg'
              }
            `}
            onClick={() => setOverlayMode(true)}
          >
            <div className="flex items-center gap-2">
              <Layers size={16} />
              Overlay
            </div>
          </motion.button>
        </div>
      </div>
      
      <motion.div 
        className="relative h-[400px] rounded-xl overflow-hidden bg-gradient-to-br from-gray-900 to-gray-800 shadow-xl"
        initial={false}
        animate={{ borderColor: isProcessing ? '#f43f5e' : '#e5e7eb' }}
      >
        <AnimatePresence mode="wait">
          {isProcessing ? (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-gray-900/90 to-gray-800/90 backdrop-blur-sm"
            >
              <motion.div 
                className="text-center space-y-4"
                initial={{ scale: 0.9 }}
                animate={{ scale: 1 }}
                exit={{ scale: 0.9 }}
              >
                <div className="relative">
                  <motion.div
                    className="w-16 h-16 rounded-full border-4 border-rose-500/30"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  />
                  <motion.div
                    className="absolute inset-0 w-16 h-16 rounded-full border-4 border-transparent border-t-amber-500"
                    animate={{ rotate: -360 }}
                    transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                  />
                </div>
                <motion.p 
                  className="text-white/90 text-sm font-medium"
                  animate={{ opacity: [0.5, 1] }}
                  transition={{ duration: 1, repeat: Infinity, repeatType: "reverse" }}
                >
                  Processing image...
                </motion.p>
              </motion.div>
            </motion.div>
          ) : (
            <>
              {/* Original Image */}
              <div 
                className="absolute top-0 left-0 w-full h-full flex items-center justify-center"
                style={{ 
                  opacity: overlayMode ? 0.5 : 1,
                  clipPath: overlayMode ? 'none' : `inset(0 ${100 - splitPosition}% 0 0)`
                }}
              >
                <img 
                  src={originalImage.dataUrl} 
                  alt="Original" 
                  className="max-h-full max-w-full object-contain"
                />
              </div>
              
              {/* Processed Image */}
              {processedImage && (
                <div 
                  className="absolute top-0 left-0 w-full h-full flex items-center justify-center"
                  style={{ 
                    clipPath: overlayMode ? 'none' : `inset(0 0 0 ${splitPosition}%)`
                  }}
                >
                  <img 
                    src={processedImage.dataUrl} 
                    alt="Processed" 
                    className="max-h-full max-w-full object-contain"
                  />
                </div>
              )}
              
              {/* Split Line */}
              {!overlayMode && processedImage && (
                <div 
                  className="absolute top-0 bottom-0 w-1 bg-white cursor-col-resize"
                  style={{ left: `${splitPosition}%` }}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    
                    const handleMouseMove = (moveEvent: MouseEvent) => {
                      const container = e.currentTarget.parentElement;
                      if (container) {
                        const rect = container.getBoundingClientRect();
                        const x = moveEvent.clientX - rect.left;
                        const newPosition = (x / rect.width) * 100;
                        setSplitPosition(Math.max(0, Math.min(100, newPosition)));
                      }
                    };
                    
                    const handleMouseUp = () => {
                      document.removeEventListener('mousemove', handleMouseMove);
                      document.removeEventListener('mouseup', handleMouseUp);
                    };
                    
                    document.addEventListener('mousemove', handleMouseMove);
                    document.addEventListener('mouseup', handleMouseUp);
                  }}
                />
              )}
            </>
          )}
        </AnimatePresence>

        {/* Labels */}
        <motion.div 
          className="absolute top-2 left-2 bg-gradient-to-r from-rose-500 to-amber-500 text-white text-xs px-3 py-1.5 rounded-full font-medium"
          whileHover={{ scale: 1.05 }}
        >
          Original
        </motion.div>
        {processedImage && (
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="absolute top-2 right-2 bg-gradient-to-r from-rose-500 to-amber-500 text-white text-xs px-3 py-1.5 rounded-full font-medium"
            whileHover={{ scale: 1.05 }}
          >
            {processedImage.analysisType} Analysis
          </motion.div>
        )}
      </motion.div>
      
      {!overlayMode && processedImage && (
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex justify-center items-center space-x-4"
        >
          <div className="text-sm font-medium text-gray-600">Original</div>
          <div className="relative w-40">
            <div className="absolute inset-0 bg-gradient-to-r from-rose-500/20 to-amber-500/20 rounded-full" />
            <input
              type="range"
              min={0}
              max={100}
              value={splitPosition}
              onChange={(e) => setSplitPosition(parseInt(e.target.value))}
              className="slider relative z-10"
            />
          </div>
          <div className="text-sm font-medium text-gray-600">Processed</div>
        </motion.div>
      )}
    </motion.div>
  );
};

export default ComparisonView;

/* CSS eklemeleri için global.css'e eklenecek:
.slider {
  @apply w-full h-2 rounded-full appearance-none bg-transparent cursor-pointer;
}

.slider::-webkit-slider-thumb {
  @apply appearance-none w-4 h-4 rounded-full bg-gradient-to-r from-rose-500 to-amber-500 
  cursor-pointer transition-all duration-200 hover:scale-110 hover:shadow-lg;
}

.slider::-moz-range-thumb {
  @apply w-4 h-4 rounded-full bg-gradient-to-r from-rose-500 to-amber-500 
  cursor-pointer transition-all duration-200 hover:scale-110 hover:shadow-lg border-none;
}
*/