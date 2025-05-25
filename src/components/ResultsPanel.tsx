import React from 'react';
import { 
  Tabs, TabsList, TabsTrigger, TabsContent 
} from './ui/Tabs';
import { 
  BarChart2, 
  Table
} from 'lucide-react';
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
        'Count': ar.cellCount ?? 0,
        'Mean Perimeter (px)': (ar.meanPerimeter ?? 0).toFixed(2),
        'Mean Feret Diameter (px)': (ar.meanFeret ?? 0).toFixed(2),
        'Mean Eccentricity': (ar.meanEccentricity ?? 0).toFixed(2),
        'Mean Aspect Ratio': (ar.meanAspectRatio ?? 0).toFixed(2),
        'Mean Centroid Dist. (px)': (ar.meanCentroidDist ?? 0).toFixed(2),
        'Mean BBox Width (px)': (ar.meanBBoxWidth ?? 0).toFixed(2),
        'Mean BBox Height (px)': (ar.meanBBoxHeight ?? 0).toFixed(2),
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
        'Total Branches': ar.totalBranches ?? 0,
        'Average Length (px)': (ar.averageLength ?? 0).toFixed(2),
        'Min Length (px)': (ar.minLength ?? 0).toFixed(2),
        'Max Length (px)': (ar.maxLength ?? 0).toFixed(2),
        'Std Deviation (px)': (ar.stdLength ?? 0).toFixed(2),
        'Median Length (px)': (ar.medianLength ?? 0).toFixed(2),
        'Variance Length (px²)': (ar.varianceLength ?? 0).toFixed(2),
        'Longest 5 Mean (px)': (ar.longest5Mean ?? 0).toFixed(2),
        'Shortest 5 Mean (px)': (ar.shortest5Mean ?? 0).toFixed(2),
        '25th Percentile (px)': (ar.percentile25 ?? 0).toFixed(2),
        '75th Percentile (px)': (ar.percentile75 ?? 0).toFixed(2),
        'Interquartile Range (px)': (ar.iqr ?? 0).toFixed(2),
        'Total Length Sum (px)': (ar.lengthSum ?? 0).toFixed(2),
        'Length Skewness': (ar.lengthSkewness ?? 0).toFixed(2),
        'Length Kurtosis': (ar.lengthKurtosis ?? 0).toFixed(2)
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
        'Average Angle (°)': (ar.average ?? 0).toFixed(2),
        'Min Angle (°)': (ar.min ?? 0).toFixed(2),
        'Max Angle (°)': (ar.max ?? 0).toFixed(2),
        'Std Dev (°)': (ar.std ?? 0).toFixed(2),
        'Resultant Vector Length': (ar.resultantVectorLength ?? 0).toFixed(3),
        'Angular Entropy': (ar.angularEntropy ?? 0).toFixed(3),
        'Angle Skewness': (ar.angleSkewness ?? 0).toFixed(3),
        'Fractal Dimension': (ar.fractalDimension ?? 0).toFixed(2),
        'Max Branch Order': ar.maxBranchOrder ?? 0,
        'Node Degree Mean': (ar.nodeDegreeMean ?? 0).toFixed(2),
        'Convex Hull Compactness': (ar.convexHullCompactness ?? 0).toFixed(2)
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
        'Total Cells': ar.totalCells ?? 0,
        'Mean Cell Area (px²)': (ar.averageArea ?? 0).toFixed(2),
        'Minimum Cell Area (px²)': (ar.minArea ?? 0).toFixed(2),
        'Maximum Cell Area (px²)': (ar.maxArea ?? 0).toFixed(2),
        'Std Dev Cell Area (px²)': (ar.std ?? 0).toFixed(2),
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
    <div>
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="mb-4">
          <TabsTrigger value="histogram" className="flex items-center">
            <BarChart2 size={16} className="mr-1" />
            Histogram
          </TabsTrigger>
          <TabsTrigger value="statistics" className="flex items-center">
            <Table size={16} className="mr-1" />
            Statistics
          </TabsTrigger>
        </TabsList>

        <TabsContent value="histogram" className="animate-fade-in">
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="text-lg font-medium text-gray-800 mb-4">
              {getResultTitle()} Distribution
            </h3>
            <div className="h-64">
              <HistogramChart data={realResults.histogram} />
            </div>
          </div>
        </TabsContent>

        <TabsContent value="statistics" className="animate-fade-in">
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="text-lg font-medium text-gray-800 mb-4">
              {getResultTitle()} Statistics
            </h3>
            <StatisticsTable statistics={realResults.statistics} />
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ResultsPanel;