export interface ImageData {
  id: string;
  name: string;
  file?: File;
  dataUrl: string;
  analysisType?: string;
  metadata?: {
    width?: number;
    height?: number;
    fileSize?: number;
    fileType?: string;
    createdAt?: Date;
  };
  analysisResults?: AnalysisResults;
}

export interface AnalysisResults {
  // --- Cell Detection sonuçları ---
  cellCount?: number;
  cellDensity?: number;
  meanPerimeter?: number;
  meanFeret?: number;
  meanEccentricity?: number;
  meanAspectRatio?: number;
  meanCentroidDist?: number;
  meanBBoxWidth?: number;
  meanBBoxHeight?: number;

  // --- Cell Area Analysis ---
  totalCells?: number;
  averageArea?: number;
  minArea?: number;
  maxArea?: number;
  std?: number;

  // --- Branch Analysis ---
  totalBranches?: number;
  averageLength?: number;
  minLength?: number;
  maxLength?: number;
  stdLength?: number;

  // --- Branch Length Advanced Statistics ---
  medianLength?: number;
  varianceLength?: number;
  longest5Mean?: number;
  shortest5Mean?: number;
  percentile25?: number;
  percentile75?: number;
  iqr?: number;
  lengthSum?: number;
  lengthSkewness?: number;
  lengthKurtosis?: number;

  // --- Angles Analysis ---
  average?: number;
  min?: number;
  max?: number;
  resultantVectorLength?: number;
  angularEntropy?: number;
  angleSkewness?: number;
  fractalDimension?: number;
  maxBranchOrder?: number;
  nodeDegreeMean?: number;
  convexHullCompactness?: number;

  // --- Histogramlar ---
  histograms?: {
    cellArea?: HistogramData;
    cellCount?: HistogramData;
    branchLength?: HistogramData;
    branchCount?: HistogramData;
    angles?: HistogramData;
  };

  // --- Eski genel istatistik yapısı (Opsiyonel) ---
  statistics?: {
    [key: string]: {
      min: number;
      max: number;
      mean: number;
      std: number;
    };
  };

  analysisType?: string;
}

export interface HistogramData {
  labels: string[];
  data: number[];
  backgroundColor?: string;
  borderColor?: string;
  xAxisLabel?: string;
  yAxisLabel?: string;
  chartTitle?: string;
}

export interface AnalysisOption {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  defaultThreshold?: number;
  thresholdRange?: {
    min: number;
    max: number;
    step: number;
  };
}

export interface ThresholdValues {
  [key: string]: number;
}

export interface Statistics {
  count: number;
  mean: number;
  min: number;
  max: number;
  std: number;
}
