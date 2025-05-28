import React, { useState } from 'react';
import { 
  CircleDot, 
  GitBranch, 
  Move, 
  Square, 
  Sliders as SlidersIcon
} from 'lucide-react';
import { AnalysisOption, ThresholdValues } from '../types';
import { motion } from 'framer-motion';

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
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-4"
    >
      {analysisOptions.map((option) => (
        <motion.div 
          key={option.id}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className={`
            border rounded-xl p-4 transition-all duration-300
            backdrop-blur-sm shadow-lg
            ${activeAnalysis === option.id 
              ? 'bg-gradient-to-r from-rose-500/10 to-amber-500/10 border-rose-500/50' 
              : 'border-white/20 hover:border-rose-300/50 bg-white/40'
            }
          `}
        >
          <div className="flex justify-between items-center mb-3">
            <div className="flex items-center gap-3">
              <motion.div 
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.5 }}
                className={`p-2.5 rounded-lg ${
                  activeAnalysis === option.id 
                    ? 'bg-gradient-to-r from-rose-500 to-amber-500 text-white' 
                    : 'bg-gray-100/80 text-gray-600 hover:bg-rose-100/80'
                }`}
              >
                {option.icon}
              </motion.div>
              <div>
                <h3 className="font-medium text-gray-800">{option.name}</h3>
                <p className="text-sm text-gray-500">{option.description}</p>
              </div>
            </div>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => handleAnalyze(option.id)}
              disabled={disabled}
              className={`
                relative px-4 py-2 rounded-lg font-medium text-sm
                transition-all duration-300 overflow-hidden
                ${activeAnalysis === option.id
                  ? 'bg-gradient-to-r from-rose-500 to-amber-500 text-white shadow-lg'
                  : 'bg-white/80 text-gray-700 hover:text-rose-600 border border-gray-200'
                }
              `}
            >
              <span className="relative z-10">
                {activeAnalysis === option.id ? 'Active' : 'Analyze'}
              </span>
              {activeAnalysis === option.id && (
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-rose-600 to-amber-600"
                  initial={{ x: '100%' }}
                  animate={{ x: '-100%' }}
                  transition={{ repeat: Infinity, duration: 3, ease: 'linear' }}
                />
              )}
            </motion.button>
          </div>
          
          {activeAnalysis === option.id && option.thresholdRange && (
            <motion.div 
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="pt-3 border-t border-rose-100/50 mt-2"
            >
              <div className="flex justify-between items-center text-sm text-gray-600 mb-2">
                <span>Threshold</span>
                <span className="font-mono bg-gradient-to-r from-rose-500 to-amber-500 bg-clip-text text-transparent">
                  {thresholds[option.id]}
                </span>
              </div>
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-rose-500/20 to-amber-500/20 rounded-full" />
                <input
                  type="range"
                  min={option.thresholdRange.min}
                  max={option.thresholdRange.max}
                  step={option.thresholdRange.step}
                  value={thresholds[option.id]}
                  onChange={(e) => handleThresholdChange(option.id, parseInt(e.target.value))}
                  className="slider relative z-10"
                  aria-label={`${option.name} threshold`}
                />
              </div>
            </motion.div>
          )}
        </motion.div>
      ))}
      
      {disabled && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-sm text-gray-500 text-center py-3"
        >
          <div className="flex items-center justify-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-bounce" style={{ animationDelay: '0ms' }} />
            <div className="w-1.5 h-1.5 rounded-full bg-rose-400 animate-bounce" style={{ animationDelay: '150ms' }} />
            <div className="w-1.5 h-1.5 rounded-full bg-rose-300 animate-bounce" style={{ animationDelay: '300ms' }} />
          </div>
          <p className="mt-2">Processing image...</p>
        </motion.div>
      )}
    </motion.div>
  );
};

export default AnalysisPanel;

/*
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