import React, { useState } from 'react';
import { 
  Microscope, Upload, BarChart2, SplitSquareVertical, 
  Sliders, Download, Sparkles, Zap, Activity 
} from 'lucide-react';
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
    <div className="min-h-screen flex flex-col bg-[radial-gradient(ellipse_at_top_left,_var(--tw-gradient-stops))] from-fuchsia-100 via-orange-100 to-amber-100 animate-gradient-xy relative overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -left-40 w-80 h-80 bg-rose-300 rounded-full mix-blend-multiply filter blur-xl opacity-30 animate-blob"></div>
        <div className="absolute top-0 -right-20 w-72 h-72 bg-orange-300 rounded-full mix-blend-multiply filter blur-xl opacity-30 animate-blob animation-delay-2000"></div>
        <div className="absolute -bottom-32 left-20 w-72 h-72 bg-amber-300 rounded-full mix-blend-multiply filter blur-xl opacity-30 animate-blob animation-delay-4000"></div>
      </div>

      <Header />
      
      <main className="flex-grow container mx-auto px-4 py-8 relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column */}
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-gradient-to-br from-rose-200 via-orange-200 to-amber-200 rounded-2xl p-1 shadow-lg hover:shadow-2xl transition-all duration-500 group hover:scale-[1.02]">
              <div className="bg-white/40 backdrop-blur-md rounded-xl p-6 hover:bg-white/50 transition-all">
                <h2 className="text-xl font-semibold flex items-center mb-4 bg-gradient-to-r from-rose-600 to-amber-600 bg-clip-text text-transparent">
                  <Upload className="mr-2 text-rose-600 group-hover:scale-125 transition-transform duration-500" size={24} />
                  Image Upload
                </h2>
                <ImageUploader onImageUpload={handleImageUpload} />
              </div>
            </div>
            
            {uploadedImage && (
              <div className="bg-gradient-to-br from-amber-100 via-orange-50 to-rose-100 rounded-2xl p-1 shadow-lg hover:shadow-2xl transition-all duration-300 animate-fade-in-up group">
                <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 hover:bg-white/90 transition-all">
                  <h2 className="text-xl font-semibold flex items-center mb-4 bg-gradient-to-r from-amber-500 to-rose-500 bg-clip-text text-transparent">
                    <Microscope className="mr-2 text-amber-500 group-hover:scale-110 transition-transform" size={20} />
                    Analysis Tools
                  </h2>
                  <AnalysisPanel 
                    onAnalyze={handleProcessImage} 
                    disabled={isAnalyzing}
                    activeAnalysis={activeAnalysis}
                  />
                </div>
              </div>
            )}
          </div>
          
          {/* Right Columns */}
          <div className="lg:col-span-2 space-y-6">
            {!uploadedImage ? (
              <div className="bg-gradient-to-br from-rose-200 via-amber-200 to-orange-200 rounded-2xl p-1 shadow-lg group hover:shadow-2xl transition-all duration-500 hover:scale-[1.01]">
                <div className="bg-white/40 backdrop-blur-md rounded-xl p-8 text-center hover:bg-white/50 transition-all">
                  <div className="relative">
                    <Microscope className="mx-auto text-transparent bg-gradient-to-r from-rose-600 to-amber-600 bg-clip-text mb-4 group-hover:scale-125 transition-transform duration-500" size={64} />
                    <Sparkles className="absolute top-0 right-1/4 text-amber-600 animate-pulse-fast" size={20} />
                    <Zap className="absolute bottom-0 left-1/4 text-rose-600 animate-bounce-slow" size={20} />
                  </div>
                  <h2 className="text-2xl font-semibold mb-2 bg-gradient-to-r from-rose-500 to-amber-500 bg-clip-text text-transparent">
                    Welcome to ImageAI Analysis
                  </h2>
                  <p className="text-gray-600 mb-4">
                    Upload an image to begin analyzing cell and branch structures.
                  </p>
                  <p className="text-sm bg-gradient-to-r from-rose-400 to-amber-400 bg-clip-text text-transparent">
                    Supports JPG, PNG, and TIFF formats up to 10MB.
                  </p>
                </div>
              </div>
            ) : (
              <>
                <div className="bg-gradient-to-br from-orange-100 via-rose-50 to-amber-100 rounded-2xl p-1 shadow-lg hover:shadow-2xl transition-all duration-300 group">
                  <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 hover:bg-white/90 transition-all">
                    <h2 className="text-xl font-semibold flex items-center mb-4 bg-gradient-to-r from-orange-500 to-rose-500 bg-clip-text text-transparent">
                      <SplitSquareVertical className="mr-2 text-orange-500 group-hover:scale-110 transition-transform" size={20} />
                      Visualization
                      <Activity className="ml-2 text-rose-500 animate-pulse" size={16} />
                    </h2>
                    <ComparisonView 
                      originalImage={uploadedImage} 
                      processedImage={activeAnalysis ? processedImages[activeAnalysis] : null}
                      isProcessing={isAnalyzing}
                    />
                  </div>
                </div>
                
                {activeAnalysis && processedImages[activeAnalysis] && (
                  <div className="bg-gradient-to-br from-rose-100 via-amber-50 to-orange-100 rounded-2xl p-1 shadow-lg hover:shadow-2xl transition-all duration-300 animate-fade-in-up group">
                    <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 hover:bg-white/90 transition-all">
                      <h2 className="text-xl font-semibold flex items-center mb-4 bg-gradient-to-r from-rose-500 to-orange-500 bg-clip-text text-transparent">
                        <BarChart2 className="mr-2 text-rose-500 group-hover:scale-110 transition-transform" size={20} />
                        Analysis Results
                      </h2>
                      <ResultsPanel 
                        analysisType={activeAnalysis}
                        imageData={processedImages[activeAnalysis]}
                      />
                      <div className="mt-4 flex justify-end">
                        <button className="bg-gradient-to-r from-rose-500 to-amber-500 hover:from-rose-600 hover:to-amber-600 text-white px-6 py-2 rounded-lg flex items-center transform hover:scale-105 transition-all duration-200 shadow-lg">
                          <Download size={16} className="mr-2" />
                          Export Results
                          <Sparkles className="ml-2 animate-pulse" size={16} />
                        </button>
                      </div>
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