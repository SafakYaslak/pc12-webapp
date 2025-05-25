import React, { useState } from 'react';
import { Eye, EyeOff, Loader } from 'lucide-react';
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
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <div className="flex space-x-3">
          <button 
            className={`btn ${overlayMode ? 'btn-ghost' : 'btn-primary'} py-1 px-3 text-sm`}
            onClick={() => setOverlayMode(false)}
          >
            Split View
          </button>
          <button 
            className={`btn ${overlayMode ? 'btn-primary' : 'btn-ghost'} py-1 px-3 text-sm`}
            onClick={() => setOverlayMode(true)}
          >
            Overlay
          </button>
        </div>
        <div className="flex items-center">
          <button className="btn btn-ghost py-1 px-2 text-sm flex items-center">
            {overlayMode ? <EyeOff size={16} className="mr-1" /> : <Eye size={16} className="mr-1" />}
            {overlayMode ? 'Hide Original' : 'Show Original'}
          </button>
        </div>
      </div>
      
      <div className="relative h-[400px] border rounded-lg overflow-hidden bg-gray-900">
        {isProcessing ? (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-100">
            <div className="text-center">
              <Loader size={32} className="animate-spin text-blue-500 mx-auto mb-2" />
              <p className="text-gray-600">Processing image...</p>
            </div>
          </div>
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
            
            {/* Labels */}
            <div className="absolute top-2 left-2 bg-black bg-opacity-50 text-white text-xs px-2 py-1 rounded">
              Original
            </div>
            {processedImage && (
              <div className="absolute top-2 right-2 bg-black bg-opacity-50 text-white text-xs px-2 py-1 rounded">
                {processedImage.analysisType} Analysis
              </div>
            )}
          </>
        )}
      </div>
      
      {!overlayMode && processedImage && (
        <div className="flex justify-center items-center space-x-2">
          <div className="text-xs text-gray-500">Original</div>
          <input
            type="range"
            min={0}
            max={100}
            value={splitPosition}
            onChange={(e) => setSplitPosition(parseInt(e.target.value))}
            className="slider w-40"
          />
          <div className="text-xs text-gray-500">Processed</div>
        </div>
      )}
    </div>
  );
};

export default ComparisonView;