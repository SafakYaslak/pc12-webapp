import React, { useState } from 'react';
import { Microscope, Upload, BarChart2, SplitSquareVertical, Sliders, Download } from 'lucide-react';
import Header from './components/Header';
import ImageUploader from './components/ImageUploader';
import AnalysisPanel from './components/AnalysisPanel';
import ResultsPanel from './components/ResultsPanel';
import ComparisonView from './components/ComparisonView';
import Footer from './components/Footer';
import { ImageData } from './types';

const App: React.FC = () => {
  const [uploadedImage, setUploadedImage] = useState<ImageData | null>(null);
  const [processedImages, setProcessedImages] = useState<Record<string, ImageData>>({});
  const [activeAnalysis, setActiveAnalysis] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleImageUpload = (imageData: ImageData) => {
    setUploadedImage(imageData);
    setProcessedImages({});
    setActiveAnalysis(null);
  };

  const handleProcessImage = async (analysisType: string, threshold: number) => {
    if (!uploadedImage) return;
    
    setIsAnalyzing(true);
    setActiveAnalysis(analysisType);
    
    try {
      // Dosya adını URL'den al
      const imageName = uploadedImage.dataUrl.split('/').pop();
      
      if (!imageName) {
        throw new Error('Invalid image name');
      }

      // Backend'e istek gönder
      const response = await fetch('http://localhost:5000/process-image', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json' 
        },
        body: JSON.stringify({ 
          imageName: imageName, // Örn: "002.jpg"
          analysisType: analysisType,
          threshold: threshold
        }),
      });
  
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
  
      const data = await response.json();
  
      // İşlenmiş görseli state'e kaydet
      const processed: ImageData = {
        ...uploadedImage,
        id: `${analysisType}-${Date.now()}`,
        name: `${analysisType} Analysis`,
        analysisType,
        dataUrl: data.processedImage,
        analysisResults: data.analysisResults
      };
  
      setProcessedImages(prev => ({
        ...prev,
        [analysisType]: processed
      }));
  
    } catch (error) {
      console.error('Image processing failed:', error);
      alert(error instanceof Error ? error.message : 'Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-grow container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Upload and Analysis Controls */}
          <div className="lg:col-span-1 space-y-6">
            <div className="card p-6 animate-fade-in">
              <h2 className="text-xl font-semibold flex items-center mb-4">
                <Upload className="mr-2 text-blue-500" size={20} />
                Image Upload
              </h2>
              <ImageUploader onImageUpload={handleImageUpload} />
            </div>
            
            {uploadedImage && (
              <div className="card p-6 animate-slide-in">
                <h2 className="text-xl font-semibold flex items-center mb-4">
                  <Microscope className="mr-2 text-blue-500" size={20} />
                  Analysis Tools
                </h2>
                <AnalysisPanel 
                  onAnalyze={handleProcessImage} 
                  disabled={isAnalyzing}
                  activeAnalysis={activeAnalysis}
                />
              </div>
            )}
          </div>
          
          {/* Center/Right Columns - Results and Visualization */}
          <div className="lg:col-span-2 space-y-6">
            {!uploadedImage && (
              <div className="card p-8 text-center animate-fade-in">
                <Microscope className="mx-auto text-blue-500 mb-4" size={48} />
                <h2 className="text-2xl font-semibold mb-2">Welcome to ImageAI Analysis</h2>
                <p className="text-gray-600 mb-4">
                  Upload an image to begin analyzing cell and branch structures.
                </p>
                <p className="text-sm text-gray-500">
                  Supports JPG, PNG, and TIFF formats up to 10MB.
                </p>
              </div>
            )}
            
            {uploadedImage && (
              <>
                <div className="card p-6 animate-fade-in">
                  <h2 className="text-xl font-semibold flex items-center mb-4">
                    <SplitSquareVertical className="mr-2 text-blue-500" size={20} />
                    Visualization
                  </h2>
                  <ComparisonView 
                    originalImage={uploadedImage} 
                    processedImage={activeAnalysis ? processedImages[activeAnalysis] : null}
                    isProcessing={isAnalyzing}
                  />
                </div>
                
                {activeAnalysis && processedImages[activeAnalysis] && (
                  <div className="card p-6 animate-slide-in">
                    <h2 className="text-xl font-semibold flex items-center mb-4">
                      <BarChart2 className="mr-2 text-blue-500" size={20} />
                      Analysis Results
                    </h2>
                    <ResultsPanel 
                      analysisType={activeAnalysis}
                      imageData={processedImages[activeAnalysis]}
                    />
                    <div className="mt-4 flex justify-end">
                      <button className="btn btn-primary flex items-center">
                        <Download size={16} className="mr-2" />
                        Export Results
                      </button>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default App;