import React from 'react';
import { 
  BarChart2, 
  Table,
  Sparkles,
  Activity,
  LineChart,
  TrendingUp,
  PieChart,
  Zap
} from 'lucide-react';
import { 
  Tabs, TabsList, TabsTrigger, TabsContent 
} from './ui/Tabs';
import { ImageData } from '../types';
import HistogramChart from './charts/HistogramChart';
import StatisticsTable from './charts/StatisticsTable';

interface ResultsPanelProps {
  analysisType: string;
  imageData: ImageData;
}

const ResultsPanel: React.FC<ResultsPanelProps> = ({ analysisType, imageData }) => {
  const [activeTab, setActiveTab] = React.useState('histogram');

  // Helper function to generate histogram configuration
  const getHistogramConfig = (type: string) => {
    const configs: Record<string, any> = {
      cell: {
        backgroundColor: 'rgba(54, 162, 235, 0.6)',
        borderColor: 'rgba(54, 162, 235, 1)',
        xAxisLabel: 'Cell Count',
        yAxisLabel: 'Frequency',
        chartTitle: 'Cell Count Distribution'
      },
      branch: {
        backgroundColor: 'rgba(75, 192, 192, 0.6)',
        borderColor: 'rgba(75, 192, 192, 1)',
        xAxisLabel: 'Branch Length (px)',
        yAxisLabel: 'Frequency',
        chartTitle: 'Branch Length Distribution (Unique)'
      },
      cellArea: {
        backgroundColor: 'rgba(255, 159, 64, 0.6)',
        borderColor: 'rgba(255, 159, 64, 1)',
        xAxisLabel: 'Cell Area (px²)',
        yAxisLabel: 'Frequency',
        chartTitle: 'Cell Area Distribution'
      },
      angles: {
        backgroundColor: 'rgba(255, 99, 132, 0.6)',
        borderColor: 'rgba(255, 99, 132, 1)',
        xAxisLabel: 'Angle (°)',
        yAxisLabel: 'Frequency',
        chartTitle: 'Angle Distribution'
      },
      branchLength: {
        backgroundColor: 'rgba(153, 102, 255, 0.6)',
        borderColor: 'rgba(153, 102, 255, 1)',
        xAxisLabel: 'Branch Length (px)',
        yAxisLabel: 'Frequency',
        chartTitle: 'Branch Length Distribution'
      }
    };
    return configs[type] || {};
  };

  // Process results from backend
  const realResults = imageData.analysisResults ? (() => {
    const defaultHistogram = { labels: [], data: [] };
    
    if (analysisType === 'cell') {
      const ar = imageData.analysisResults;
      const cellStats = {
        'cellCount': ar.cellCount ?? 0,
        'cellDensity': ar.cellDensity ?? 0,
        'meanPerimeter': ar.meanPerimeter ?? 0,
        'meanFeret': ar.meanFeret ?? 0,
        'meanEccentricity': ar.meanEccentricity ?? 0,
        'meanAspectRatio': ar.meanAspectRatio ?? 0,
        'meanCentroidDist': ar.meanCentroidDist ?? 0,
        'meanBBoxWidth': ar.meanBBoxWidth ?? 0,
        'meanBBoxHeight': ar.meanBBoxHeight ?? 0
      };
      
      return {
        statistics: cellStats,
        histogram: {
          ...(ar.histograms?.cellCount || defaultHistogram),
          ...getHistogramConfig('cell')
        }
      };
    }

    if (analysisType === 'branch') {
      const ar = imageData.analysisResults;
      const branchStats = {
        'Branch Count': ar.totalBranches ?? 0,
      };
      
      return {
        statistics: branchStats,
        histogram: {
          ...(ar.histograms?.branchLength || defaultHistogram),
          ...getHistogramConfig('branch')
        }
      };
    }

    if (analysisType === 'branchLength') {
      const ar = imageData.analysisResults;
      const branchLengthStats = {
        'totalBranches': ar.totalBranches ?? 0,
        'averageLength': ar.averageLength ?? 0,
        'minLength': ar.minLength ?? 0,
        'maxLength': ar.maxLength ?? 0,
        'stdDeviation': ar.stdLength ?? 0,
        'medianLength': ar.medianLength ?? 0,
        'varianceLength': ar.varianceLength ?? 0,
        'longest5Mean': ar.longest5Mean ?? 0,
        'shortest5Mean': ar.shortest5Mean ?? 0,
        'percentile25': ar.percentile25 ?? 0,
        'percentile75': ar.percentile75 ?? 0,
        'iqr': ar.iqr ?? 0,
        'lengthSum': ar.lengthSum ?? 0,
        'lengthSkewness': ar.lengthSkewness ?? 0
      };
    
      return {
        statistics: branchLengthStats,
        histogram: {
          ...(ar.histograms?.branchLength || defaultHistogram),
          ...getHistogramConfig('branchLength')
        }
      };
    }
    if (analysisType === 'angles') {
      const ar = imageData.analysisResults;
      const angleStats = {
        'averageAngle': ar.average ?? 0,
        'minAngle': ar.min ?? 0,
        'maxAngle': ar.max ?? 0,
        'stdDevAngle': ar.std ?? 0,
        'resultantVectorLength': ar.resultantVectorLength ?? 0,
        'angularEntropy': ar.angularEntropy ?? 0,
        'angleSkewness': ar.angleSkewness ?? 0,
        'fractalDimension': ar.fractalDimension ?? 0,
        'maxBranchOrder': ar.maxBranchOrder ?? 0,
        'nodeDegree': ar.nodeDegreeMean ?? 0,
        'convexHullCompactness': ar.convexHullCompactness ?? 0
      };

      return {
        statistics: angleStats,
        histogram: {
          ...(ar.histograms?.angles || defaultHistogram),
          ...getHistogramConfig('angles')
        }
      };
    }

    if (analysisType === 'cellArea') {
      const ar = imageData.analysisResults;
      const areaStats = {
        'totalCells': ar.totalCells ?? 0,
        'meanCellArea': ar.averageArea ?? 0,
        'minCellArea': ar.minArea ?? 0,
        'maxCellArea': ar.maxArea ?? 0,
        'stdDevCellArea': ar.std ?? 0
      };

      return {
        statistics: areaStats,
        histogram: {
          ...(ar.histograms?.cellArea || defaultHistogram),
          ...getHistogramConfig('cellArea')
        }
      };
    }

    // Default fallback
    return {
      statistics: {
        count: 0,
        mean: '0',
        min: '0',
        max: '0',
        std: '0'
      },
      histogram: defaultHistogram
    };
  })() : null;

  if (!realResults) {
    return <div className="text-gray-500">No analysis results available</div>;
  }

  const getResultTitle = () => {
    switch (analysisType) {
      case 'cell': return 'Cell Detection Metrics';
      case 'branch': return 'Branch Detection Metrics ';
      case 'cellArea': return 'Cell Area (px²) Metrics';
      case 'branchLength': return 'Branch Length (px) Metrics';
      case 'angles': return 'Branch Angles (°) Metrics';
      default: return 'Analysis Results';
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="mb-6 p-1 bg-gradient-to-r from-rose-100/50 via-orange-100/50 to-amber-100/50 rounded-lg backdrop-blur-sm">
          <TabsTrigger 
            value="histogram" 
            className="flex items-center space-x-2 transition-all duration-300 hover:scale-105"
          >
            <div className="relative">
              <BarChart2 size={18} className="text-rose-500 transition-transform group-hover:scale-110" />
              <Sparkles size={10} className="absolute -top-1 -right-1 text-amber-500 animate-pulse" />
            </div>
            <span>Histogram</span>
          </TabsTrigger>
          <TabsTrigger 
            value="statistics" 
            className="flex items-center space-x-2 transition-all duration-300 hover:scale-105"
          >
            <div className="relative">
              <Table size={18} className="text-orange-500 transition-transform group-hover:scale-110" />
              <Activity size={10} className="absolute -top-1 -right-1 text-rose-500 animate-pulse" />
            </div>
            <span>Statistics</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent 
          value="histogram" 
          className="animate-fade-in transform transition-all duration-500"
        >
          <div className="bg-gradient-to-br from-rose-50/30 via-orange-50/30 to-amber-50/30 backdrop-blur-sm rounded-xl p-4 shadow-lg hover:shadow-xl transition-all duration-300 group">
            <div className="flex items-center space-x-3 mb-4">
              <LineChart className="text-rose-500 group-hover:scale-110 transition-transform duration-300" size={24} />
              <h3 className="text-xl font-medium bg-gradient-to-r from-rose-600 to-amber-600 bg-clip-text text-transparent">
                {getResultTitle()} Distribution
              </h3>
              <TrendingUp className="text-amber-500 animate-bounce-slow ml-auto" size={20} />
            </div>
            
            {/* Container for histogram */}
            <div className="w-full overflow-hidden">
              <HistogramChart data={realResults.histogram} />
            </div>
          </div>
        </TabsContent>

        <TabsContent 
          value="statistics" 
          className="animate-fade-in transform transition-all duration-500"
        >
          <div className="bg-gradient-to-br from-orange-50/30 via-rose-50/30 to-amber-50/30 backdrop-blur-sm rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 group">
            <div className="flex items-center space-x-3 mb-6">
              <PieChart className="text-orange-500 group-hover:scale-110 transition-transform duration-300" size={24} />
              <h3 className="text-xl font-medium bg-gradient-to-r from-orange-600 to-rose-600 bg-clip-text text-transparent">
                {getResultTitle()} Statistics
              </h3>
              <Zap className="text-rose-500 animate-pulse ml-auto" size={20} />
            </div>
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-br from-orange-500/5 via-rose-500/5 to-amber-500/5 opacity-50 group-hover:opacity-70 transition-opacity duration-300 rounded-xl" />
              <StatisticsTable statistics={realResults.statistics} />
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ResultsPanel;