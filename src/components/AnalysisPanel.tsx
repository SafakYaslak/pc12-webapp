import React, { useState } from 'react';
import { 
  CircleDot, 
  GitBranch, 
  Move, 
  Square, 
  Sliders as SlidersIcon
} from 'lucide-react';
import { AnalysisOption, ThresholdValues } from '../types';

interface AnalysisPanelProps {
  onAnalyze: (analysisType: string, threshold: number) => void; // Threshold parametresi eklendi
  disabled: boolean;
  activeAnalysis: string | null;
}

const AnalysisPanel: React.FC<AnalysisPanelProps> = ({ 
  onAnalyze, 
  disabled,
  activeAnalysis
}) => {
  const [thresholds, setThresholds] = useState<ThresholdValues>({
    'cell': 128,
    'branch': 128,
    'cellArea': 128,
    'branchLength': 128,
    'angles': 128
  });

  const analysisOptions: AnalysisOption[] = [
    {
      id: 'cell',
      name: 'Cell Detection',
      description: 'Identify and count individual cells',
      icon: <CircleDot size={18} />,
      defaultThreshold: 128,
      thresholdRange: { min: 0, max: 255, step: 1 }
    },
    {
      id: 'branch',
      name: 'Branch Detection',
      description: 'Detect branching structures',
      icon: <GitBranch size={18} />,
      defaultThreshold: 128,
      thresholdRange: { min: 0, max: 255, step: 1 }
    },
    {
      id: 'cellArea',
      name: 'Cell Area',
      description: 'Measure area of detected cells',
      icon: <Square size={18} />,
      defaultThreshold: 128,
      thresholdRange: { min: 0, max: 255, step: 1 }
    },
    {
      id: 'branchLength',
      name: 'Branch Length',
      description: 'Calculate length of branches',
      icon: <Move size={18} />,
      defaultThreshold: 128,
      thresholdRange: { min: 0, max: 255, step: 1 }
    },
    {
      id: 'angles',
      name: 'Branch Angles',
      description: 'Measure angles between branches',
      icon: <SlidersIcon size={18} />,
      defaultThreshold: 128,
      thresholdRange: { min: 0, max: 255, step: 1 }
    }
  ];

  const handleThresholdChange = (analysisId: string, value: number) => {
    setThresholds(prev => ({
      ...prev,
      [analysisId]: value
    }));
  };

  const handleAnalyze = (analysisId: string) => {
    if (disabled) return;
    onAnalyze(analysisId, thresholds[analysisId]); // Threshold değerini gönder
  };

  return (
    <div className="space-y-4">
      {analysisOptions.map((option) => (
        <div 
          key={option.id}
          className={`border rounded-lg p-3 transition-all duration-200 ${
            activeAnalysis === option.id 
              ? 'border-blue-500 bg-blue-50' 
              : 'border-gray-200 hover:border-blue-300'
          }`}
        >
          <div className="flex justify-between items-center mb-2">
            <div className="flex items-center">
              <div className={`p-1.5 rounded-md mr-2 ${
                activeAnalysis === option.id 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-100 text-gray-600'
              }`}>
                {option.icon}
              </div>
              <div>
                <h3 className="font-medium text-gray-800">{option.name}</h3>
                <p className="text-xs text-gray-500">{option.description}</p>
              </div>
            </div>
            <button
              onClick={() => handleAnalyze(option.id)}
              disabled={disabled}
              className={`btn ${
                activeAnalysis === option.id
                  ? 'btn-primary'
                  : 'btn-ghost'
              } py-1 px-3 text-sm`}
            >
              {activeAnalysis === option.id ? 'Active' : 'Analyze'}
            </button>
          </div>
          
          {activeAnalysis === option.id && option.thresholdRange && (
            <div className="pt-2 border-t border-gray-200 mt-2">
              <div className="flex justify-between items-center text-xs text-gray-600 mb-1">
                <span>Threshold</span>
                <span>{thresholds[option.id]}</span>
              </div>
              <input
                type="range"
                min={option.thresholdRange.min}
                max={option.thresholdRange.max}
                step={option.thresholdRange.step}
                value={thresholds[option.id]}
                onChange={(e) => handleThresholdChange(option.id, parseInt(e.target.value))}
                className="slider"
                aria-label={`${option.name} threshold`}
              />
            </div>
          )}
        </div>
      ))}
      
      {disabled && (
        <div className="text-sm text-gray-500 animate-pulse text-center py-2">
          Processing image...
        </div>
      )}
    </div>
  );
};

export default AnalysisPanel;