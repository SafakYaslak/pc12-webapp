import React, { useState } from 'react';
import {
  Microscope, Upload, BarChart2, SplitSquareVertical,
  Sliders, Download, Sparkles, Zap, Activity
} from 'lucide-react';
import Header from './components/Header'; // Varsayılan: Bu bileşenlerin var olduğunu ve doğru yolda olduğunu varsayıyorum
import ImageUploader from './components/ImageUploader'; // Varsayılan
import AnalysisPanel from './components/AnalysisPanel';   // Varsayılan
import ResultsPanel from './components/ResultsPanel';     // Varsayılan
import ComparisonView from './components/ComparisonView'; // Varsayılan
import Footer from './components/Footer';                 // Varsayılan
import { ImageData } from './types'; // types.ts dosyanızın doğru yolda olduğundan emin olun

const App: React.FC = () => {
  const [uploadedImage, setUploadedImage] = useState<ImageData | null>(null);
  const [processedImages, setProcessedImages] = useState<Record<string, ImageData>>({});
  const [activeAnalysis, setActiveAnalysis] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [cellVisualization, setCellVisualization] = useState<string | null>(null);

  const handleImageUpload = (imageData: ImageData) => {
    setUploadedImage(imageData);
    setProcessedImages({});
    setActiveAnalysis(null);
    setCellVisualization(null);
  };

  const handleProcessImage = async (analysisType: string, threshold: number) => {
    if (!uploadedImage) return;

    setIsAnalyzing(true);
    setActiveAnalysis(analysisType);

    try {
      const imageName = uploadedImage.dataUrl.split('/').pop();

      if (!imageName) {
        throw new Error('Invalid image name');
      }

      const response = await fetch('http://localhost:5000/process-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          imageName: imageName,
          analysisType: analysisType,
          threshold: threshold
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (analysisType === 'cell' && data.cellVisualization) {
        setCellVisualization(data.cellVisualization);
      } else {
        setCellVisualization(null);
      }

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

  const handleExportResults = () => {
    if (activeAnalysis && processedImages[activeAnalysis] && processedImages[activeAnalysis].analysisResults) {
      const results = processedImages[activeAnalysis].analysisResults;
      let dataStr = `Analysis Type: ${activeAnalysis}\n\n`;

      // Helper function to format nested objects/arrays
      const formatValue = (value: any, indentLevel = 0): string => {
        const indent = '  '.repeat(indentLevel);
        if (Array.isArray(value)) {
          let arrayStr = "";
          if (value.every(item => typeof item !== 'object' || item === null)) { // Check if it's an array of simple values
            return `${indent}[${value.join(', ')}]\n`;
          }
          value.forEach((item, index) => {
            arrayStr += `${indent}${index}:\n${formatValue(item, indentLevel + 1)}`;
          });
          return arrayStr;
        } else if (typeof value === 'object' && value !== null) {
          let objectStr = "";
          for (const [itemKey, itemValue] of Object.entries(value)) {
            objectStr += `${indent}${itemKey}: ${formatValue(itemValue, indentLevel + 1)}`;
          }
          // If objectStr is empty, it means the object itself was just processed (e.g. empty object)
          // or all its children were handled inline (like simple arrays).
          // In this case, we might not want an extra newline or just indicate it's an object.
          // For simplicity now, if it's an object that leads to a nested structure, it will start a new line via its children.
          return objectStr; // Return the accumulated string for the object
        } else {
          // For primitive values, just return them (newline will be added by parent if it's a direct child of a key)
          return `${value}\n`;
        }
      };

      // Revised main formatting loop
      for (const [key, value] of Object.entries(results)) {
        if (key === 'histograms') {
          continue; // Skip 'histograms'
        }

        dataStr += `${key}: `; // Add the main key

        if (key === 'angleDetails') {
          dataStr += "\n"; // Newline after "angleDetails:"
          if (Array.isArray(value)) {
            value.forEach((detail, index) => {
              dataStr += `  ${index}:\n${formatValue(detail, 2)}`; // Indent items
            });
          } else if (typeof value === 'object' && value !== null) {
            for (const [detailKey, detailValue] of Object.entries(value as Record<string, any>)) {
              dataStr += `  ${detailKey}:\n${formatValue(detailValue, 2)}`; // Indent items
            }
          }
        } else if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
            dataStr += "\n"; // Newline for objects
            dataStr += formatValue(value, 1); // Format object with initial indent
        } else if (Array.isArray(value)) {
            // Handle top-level arrays that are not angleDetails
            if (value.every(item => typeof item !== 'object' || item === null)) {
                dataStr += `[${value.join(', ')}]\n`;
            } else {
                dataStr += "\n";
                value.forEach((item, index) => {
                    dataStr += `  ${index}:\n${formatValue(item, 2)}`;
                });
            }
        }
         else {
          dataStr += `${value}\n`; // Primitive values
        }
      }


      const blob = new Blob([dataStr.trim()], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${activeAnalysis}-analysis-results.txt`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } else {
      alert('No analysis results available to export.');
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
                      cellVisualization={cellVisualization}
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
                        <button
                          onClick={handleExportResults}
                          className="bg-gradient-to-r from-rose-500 to-amber-500 hover:from-rose-600 hover:to-amber-600 text-white px-6 py-2 rounded-lg flex items-center transform hover:scale-105 transition-all duration-200 shadow-lg"
                          disabled={isAnalyzing || !processedImages[activeAnalysis]?.analysisResults}
                        >
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