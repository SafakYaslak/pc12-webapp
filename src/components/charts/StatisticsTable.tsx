import React, { useState } from 'react';
import { 
  TrendingUp, 
  BarChart,
  Maximize2, 
  Target, 
  Activity, 
  LineChart, 
  PieChart,
  Info,
  X
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface Statistics {
  [key: string]: string | number;
}

interface StatisticsTableProps {
  statistics: Statistics;
}

// Icon mapping for different statistics
const statIconMap: Record<string, React.ReactNode> = {
  mean: <TrendingUp className="w-5 h-5 text-emerald-500" />,
  median: <BarChart className="w-5 h-5 text-blue-500" />,
  mode: <Target className="w-5 h-5 text-purple-500" />,
  stdDev: <Activity className="w-5 h-5 text-rose-500" />,
  variance: <PieChart className="w-5 h-5 text-amber-500" />,
  range: <Maximize2 className="w-5 h-5 text-cyan-500" />,
};

const formatLabel = (key: string) => {
  return key
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/_/g, ' ')
    .replace(/^\w/, c => c.toUpperCase());
};

interface StatisticDescription {
  title: string;
  description: string;
  medicalContext: string;
}

  // Kapsamlı açıklamalar
  const statisticDescriptions: Record<string, StatisticDescription> = {
  // Cell Detection Metrics
  cellCount: {
  title: "Cell Count",
  description: "Total number of cells detected in the image. This value shows the total number of cell objects detected by the automatic cell segmentation algorithm.",
  medicalContext: "Indicates cell density and population size. A critical parameter for evaluating treatment efficacy, monitoring cell proliferation, and analyzing tissue health."
  },
  cellDensity: {
  title: "Cell Density",
  description: "Number of cells per unit area. This metric shows the density of cells in a given surface area.",
  medicalContext: "Indicates tissue organization and cell distribution. High cell density is generally associated with active proliferation and tight cell packing."
  },
  meanPerimeter: {
  title: "Average Perimeter",
  description: "Average perimeter length of cells (in pixels). It represents the average of the total length of cell borders for all cells.",
  medicalContext: "Reflects the complexity of the cell membrane and morphological changes. Significant changes in perimeter length are observed during cell damage, apoptosis processes and cell differentiation."
  },
  meanFeret: {
  title: "Average Feret Diameter",
  description: "Average of the longest distance of the cell (in pixels). Feret diameter represents the maximum distance between any two points of the cell.",
  medicalContext: "It is used to evaluate cell size in terms of maximum length. It is an important indicator in the follow-up of cell elongation, morphological changes and disease processes."
  },
  meanEccentricity: {
  title: "Average Eccentricity",
  description: "Degree of deviation from circularity of cells (value between 0-1). Value 0 indicates perfect circle, value 1 indicates maximum elongation.",
  medicalContext: "Indicates irregularity and elongation of cell shape. Increased eccentricity in cancer cells may be an indicator of metastatic potential and cellular stress response."
  },
  meanAspectRatio: {
  title: "Average Aspect Ratio",
  description: "Average of the width/height ratio of cells. This ratio indicates how long and narrow the cell is.",
  medicalContext: "Assess the symmetry of cell morphology. A critical parameter in neuron differentiation, cell migration and tissue organization processes."
  },
  meanCentroidDist: {
  title: "Average Center Distance",
  description: "Average distance of cells from the image center (in pixels). Shows the spatial distribution of cells within the image.",
  medicalContext: "Shows the distribution pattern of cells. Provides important information for the analysis of cell aggregation, colony formation, and tissue organization processes."
  },
  meanBBoxWidth: {
  title: "Average Bounding Box Width",
  description: "Average width of the rectangle surrounding cells (in pixels). The width of the minimum rectangle that completely encloses each cell.",
  medicalContext: "Shows the horizontal size distribution of cells. Provides a basic reference value for cell size normalization and morphological analysis."
  },
  meanBBoxHeight: {
  title: "Average Bounding Box Height",
  description: "Average height of the rectangle surrounding the cells (in pixels). It is the height value of the minimum rectangle that completely encloses each cell.",
  medicalContext: "Shows the vertical size distribution of cells. Used for quantitative analysis of cellular elongation and morphological changes."
  },

  // Branch Detection Metrics
  branchCount: {
  title: "Branch Count",
  description: "Total number of branches (neurites) detected. It is the value obtained by automatically counting all extensions originating from neuron cells.",
  medicalContext: "Shows neuron branching complexity and synaptic connection potential. It is a critical parameter in the follow-up of neurodegenerative diseases, neuron development and synaptogenesis processes."
  },

  // Cell Area Metrics
  totalCells: {
  title: "Total Cells",
  description: "Total number of analyzed cells. Total number of cells successfully segmented by the image processing algorithm.",
  medicalContext: "Shows population size and cell density. Basic data for proliferation analysis, treatment efficacy evaluation and tissue homeostasis studies.",
  },
  meanCellArea: {
  title: "Mean Cell Area",
  description: "Mean surface area of ​​cells (in square pixels). Arithmetic average of area values ​​of all segmented cells.",
  medicalContext: "Used for general assessment of cell size. An indicator of cell cycle, metabolic activity and cellular stress responses.",
  },
  minCellArea: {
  title: "Minimum Cell Area",
  description: "Smallest cell area (in square pixels). Surface area value of the smallest cell in the population.",
  medicalContext:"Used for size assessment of the smallest cells. Important for detection of cell fragmentation, apoptotic processes and early stages of cell division."
  },
  maxCellArea: {
  title: "Maximum Cell Area",
  description: "The largest cell area (in square pixels). It is the surface area value of the largest cell in the population.",
  medicalContext: "Used for size assessment of the largest cells. It is a critical parameter for detection of cellular hypertrophy, multinucleated cell formation and abnormal cell growth."
  },
  stdDevCellArea: {
  title: "Cell Area Standard Deviation",
  description: "Variability of cell areas (in square pixels). It shows the width of the distribution of cell sizes around the mean.",
  medicalContext: "Evaluates the homogeneity of cell sizes. High values ​​indicate heterogeneity in the cell population, low values ​​indicate uniform growth."
  },

  // Branch Length Metrics
  totalBranches: {
  title: "Total Branch Count",
  description: "Total number of branches detected. Total number of neurite extensions originating from neuron cells.",
  medicalContext: "Shows the complexity of the neural network and synaptic connection capacity. Used in the quantitative assessment of neuron maturation, synaptogenesis and neural plasticity."
  },
  averageLength: {
  title: "Average Length",
  description: "Average length of branches (in pixels). Arithmetic average of length values ​​of all neurite branches.",
  medicalContext: "Shows the neurite extension capacity and growth potential. Important parameter in the monitoring of neuron development, axon myelination and neurodegenerative processes."
  },
  minLength: {
  title: "Minimum Length",
  description: "Length of the shortest branch (in pixels). It is the length value of the shortest neurite extension in the population.",
  medicalContext: "Shows the shortest neurite extensions. It is used to determine the initial branching processes, growth cone activity and minimal extension thresholds."
  },
  maxLength: {
  title: "Maximum Length",
  description: "Length of the longest branch (in pixels). It is the length value of the longest neurite extension in the population.",
  medicalContext: "Shows the maximum neurite extension capacity. It is the critical parameter in the evaluation of axon growth, target-finding capacity and neural connectivity potential." },
  stdDeviation: {
  title: "Standard Deviation",
  description: "Variability of branch lengths (in pixels). Shows the width of the distribution of neurite lengths around the mean.",
  medicalContext: "Evaluates the homogeneity of neurite lengths. Used to detect uniform growth patterns or heterogeneous branching processes."
  },
  medianLength: {
  title: "Median Length",
  description: "Median value of branch lengths (in pixels). Robust statistical parameter showing the central tendency of the length distribution.",
  medicalContext: "Shows the typical neurite length. Reflects the true central length characteristic of the population, unaffected by outliers."
  },
  varianceLength: {
  title: "Variance",
  description: "Variability of branch lengths (in pixels). It is a measure of variability calculated as the square of the standard deviation.",
  medicalContext: "It quantitatively shows the magnitude of variability in neurite lengths. It is used to evaluate the consistency of growth processes and homogeneity of neuron maturation."
  },
  longest5Mean: {
  title: "Average of Longest 5 Branches",
  description: "Average length of the longest 5 branches (in pixels). It is the average length of neurites with maximal growth capacity.",
  medicalContext: "It shows the maximum growth potential and elite growth performance. It is used to evaluate major axon tracts and long-distance connections."
  },
  shortest5Mean: {
  title: "Average of 5 Shortest Branches",
  description: "Average length of the 5 shortest branches (in pixels). Average length of neurites showing minimal extension.",
  medicalContext: "Shows the minimum growth limit and growth inhibition effects. Important for the analysis of local interneuron connections and short-range synaptic networks."
  },
  percentile25: {
  title: "25th Percentile",
  description: "Lower quartile of branch lengths (in pixels). Length limit below which 25% of the population remains.",
  medicalContext: "Shows the distribution of short branches and characteristics of the minimal growth segment. Used for the evaluation of early branching processes." },
  percentile75: {
  title: "75th Percentile",
  description: "Upper quartile of branch lengths (in pixels). The length limit below which 75% of the population lies.",
  medicalContext: "Shows the distribution of long branches and the characteristics of the mature growth segment. Important for the analysis of mature neurite extensions and long-distance connections."
  },
  iqr: {
  title: "Quadrants A",

  description: "Difference between 25th and 75th percentiles (in pixels). Shows the width of the length distribution of the middle 50%.",

  medicalContext: "Shows the width of the central distribution of branch lengths. Used in the assessment of typical growth range and neurite extension consistency."},

  lengthSum: {

  title: "Total Length",

  description: "Total length of all branches (in pixels). Total neurite network length per neuron.",

  medicalContext: "Shows the total neurite capacity and synaptic potential. Critical parameter in the overall assessment of neural connectivity, metabolic demand and growth capacity.",

  },

  lengthSkewness: {

  title: "Length Skewness",

  description: "Distribution asymmetry of branch lengths. Positive values ​​indicate a distribution skewed to the right, negative values ​​indicate a distribution skewed to the left.",

  medicalContext: "Evaluates the symmetry of the length distribution. Normal development typically shows positive skew (few very long branches)." },

  // Branch Angles Metrics
  averageAngle: {
  title: "Average Angle",
  description: "Average branching angle of branches (in degrees). Arithmetic mean of angular values ​​at neurite branching points.",
  medicalContext: "Indicates neurite branching pattern and growth orientation. Critical parameter for optimal synaptic distribution and efficient neural network organization." },
  minAngle: {
  title: "Minimum Angle",
  description: "Smallest branching angle (in degrees). Angle value of the narrowest branching point in the population.",
  medicalContext: "Indicates the narrowest branching angle and dense growth patterns. Indicates adaptive branching response seen in dense neural tissue areas." },
  maxAngle: {
  title: "Maximum Angle",
  description: "Maximum branching angle (in degrees). The angle value of the largest branching point in the population.",
  medicalContext: "Shows the largest branching angle and maximal spreading patterns. Indicates growth strategies for covering large areas.",
  },
  stdDevAngle: {
  title: "Angle Standard Deviation",
  description: "Variability of branching angles (in degrees). Shows the width of the distribution of angle values ​​around the mean.",
  medicalContext: "Evaluates the consistency of branching angles and stereotypical growth patterns. Low values ​​indicate uniform branching, high values ​​indicate variable branching patterns.",
  },
  
  resultantVectorLength: {
  title: "Result Vector Length",
  description: "Strength of direction of angular distribution (value between 0 and 1). Values ​​close to 1 indicate strong directionality, values ​​close to 0 indicate random distribution.",
  medicalContext: "Shows the preferred direction of branching and growth bias. Used for quantitative analysis of chemotactic gradient responses and directional growth processes."
  },
  angularEntropy: {
  title: "Angular Entropy",
  description: "Irregularity and information content of the angle distribution. High entropy indicates a variety of angles, low entropy indicates a uniform angle.",
  medicalContext: "Indicates the randomness and degree of organization of the branching pattern. Important parameter to distinguish stochastic vs. deterministic growth processes."
  },
  angleSkewness: {
  title: "Angle Skewness",
  description: "Asymmetry of the angle distribution. Positive skewness indicates a tendency toward large angles, negative skewness indicates a tendency toward small angles.",
  medicalContext: "Evaluates the symmetry and branching bias of the angular distribution. Used to analyze the effects of specific growth conditions on angular preferences." },
  fractalDimension: {
  title: "Fractal Dimension",
  description: "Complexity and self-similarity of the branching pattern. It takes values ​​from 1 to 2, with higher values ​​indicating more complex, convoluted structures.",
  medicalContext: "Shows the geometric complexity and area coverage efficiency of the neurite tree. Used for quantitative measurement of dendritic complexity and synaptic integration capacity." },
  maxBranchOrder: {
  title: "Maximum Branch Order",
  description: "Highest branching level. It is the number of hierarchical levels starting from the cell body to the farthest branching point.",
  medicalContext: "Shows the depth of the branching hierarchy and the maturation level of the dendritic tree. It is an important indicator of signal integration capacity and neural complexity."
  },
  nodeDegree: {
  title: "Node Degree Average",
  description: "Average number of branches at branching points. Average of the number of branches originating from each bifurcation point.",
  medicalContext: "Indicates branching complexity and synaptic integration potential. Higher values ​​indicate more complex dendritic trees and advanced signal processing capacity."
  },
  convexHullCompactness: {
  title: "Convex Envelope Compactness",
  description: "Density and area usage efficiency of branching structure. Convex area covered by neurite network and real dal density ratio.",
  medicalContext: "It shows the space usage efficiency and spatial organization quality of the neurite tree. It is an important parameter for effective synaptic coverage and optimal neural wiring."
  }
  };


const StatisticsTable: React.FC<StatisticsTableProps> = ({ statistics }) => {
  const [activePopover, setActivePopover] = useState<string | null>(null);
  const [hoveredRow, setHoveredRow] = useState<string | null>(null);

  const getStatisticInfo = (key: string) => {
    // Anahtar eşleştirme haritası
    const keyMap: Record<string, string> = {
        'cellCount': 'cellCount',
        'cellDensity': 'cellDensity',
        'meanPerimeter': 'meanPerimeter',
        'meanFeret': 'meanFeret',
        'meanEccentricity': 'meanEccentricity',
        'meanAspectRatio': 'meanAspectRatio',
        'meanCentroidDist': 'meanCentroidDist',
        'meanBBoxWidth': 'meanBBoxWidth',
        'meanBBoxHeight': 'meanBBoxHeight',
        // Branch analizi için
        'totalBranches': 'totalBranches',
        'averageLength': 'averageLength',
        'minLength': 'minLength',
        'maxLength': 'maxLength',
        'stdLength': 'stdDeviation',
        'medianLength': 'medianLength',
        'varianceLength': 'varianceLength',
        'longest5Mean': 'longest5Mean',
        'shortest5Mean': 'shortest5Mean',
        'percentile25': 'percentile25',
        'percentile75': 'percentile75',
        'iqr': 'iqr',
        'lengthSum': 'lengthSum',
        'lengthSkewness': 'lengthSkewness',
        // Cell Area analizi için
        'totalCells': 'totalCells',
        'averageArea': 'meanCellArea',
        'minArea': 'minCellArea',
        'maxArea': 'maxCellArea',
        'std': 'stdDevCellArea'
    };

    // Eşleştirilen anahtarı bul
    const mappedKey = keyMap[key] || key;
    
    // Eşleştirilen anahtara göre açıklamayı bul
    const info = statisticDescriptions[mappedKey];
    
    if (!info) {
        console.log(`Eşleşme bulunamadı: ${key} -> ${mappedKey}`); // Debug için
        return {
            title: formatLabel(key),
            description: "Description for this metric has not yet been added.",
            medicalContext: "The medical significance of this metric has not yet been added"
        };
    }
    
    return info;
  };

  console.log('Gelen istatistikler:', statistics);
  console.log('Mevcut açıklamalar:', Object.keys(statisticDescriptions));

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="relative w-full"
    >
      <div className="w-full bg-gradient-to-br from-white/40 to-white/80 backdrop-blur-sm rounded-xl p-1">
        <div className="w-full rounded-lg border border-gray-200/50 shadow-xl overflow-hidden">
          <table className="w-full divide-y divide-gray-200/50">
            <thead className="bg-gradient-to-r from-rose-100/50 to-orange-100/50">
              <tr>
                <motion.th 
                  whileHover={{ backgroundColor: 'rgba(255,255,255,0.1)' }}
                  className="w-[45%] px-6 py-4 text-left text-sm font-semibold text-gray-700"
                >
                  <div className="flex items-center gap-2">
                    <span>Metrics</span>
                    <motion.div
                      animate={{ rotate: [0, 360] }}
                      transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                      className="w-1 h-1 rounded-full bg-rose-500"
                    />
                  </div>
                </motion.th>
                <th className="w-[45%] px-6 py-4 text-left text-sm font-semibold text-gray-700">Values</th>
                <th className="w-[10%] px-2" />
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200/50">
              {Object.entries(statistics)
                .filter(([key]) => !['averageAngle', 'minAngle', 'maxAngle', 'stdDevAngle', 'nodeDegree'].includes(key))
                .map(([key, value]) => (
                  <motion.tr
                    key={key}
                    onHoverStart={() => setHoveredRow(key)}
                    onHoverEnd={() => setHoveredRow(null)}
                    className="group relative"
                  >
                    <motion.td 
                      className="w-[45%] px-6 py-4 whitespace-nowrap"
                      initial={false}
                      animate={{
                        backgroundColor: hoveredRow === key ? 'rgba(244,63,94,0.05)' : 'transparent'
                      }}
                    >
                      <div className="flex items-center space-x-3">
                        <motion.div 
                          className="flex-shrink-0"
                          whileHover={{ scale: 1.2, rotate: 360 }}
                          transition={{ duration: 0.3 }}
                        >
                          {statIconMap[key.toLowerCase()] || <LineChart className="w-5 h-5 text-gray-400" />}
                        </motion.div>
                        <div className="text-sm font-medium text-gray-900">
                          {getStatisticInfo(key).title}
                        </div>
                      </div>
                    </motion.td>
                    <motion.td 
                      className="w-[45%] px-6 py-4 whitespace-nowrap"
                      initial={false}
                      animate={{
                        backgroundColor: hoveredRow === key ? 'rgba(244,63,94,0.05)' : 'transparent'
                      }}
                    >
                      <motion.div
                        className="text-sm font-mono"
                        initial={{ color: '#374151' }}
                        animate={{
                          background: hoveredRow === key 
                            ? 'linear-gradient(to right, #f43f5e, #f59e0b)' 
                            : 'none',
                          WebkitBackgroundClip: hoveredRow === key ? 'text' : 'none',
                          WebkitTextFillColor: hoveredRow === key ? 'transparent' : '#374151'
                        }}
                      >
                        {value}
                      </motion.div>
                    </motion.td>
                    <td className="w-[10%] px-2 py-4 relative">
                      <button
                        onClick={() => setActivePopover(activePopover === key ? null : key)}
                        className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-full 
                          transition-all duration-200 
                          hover:shadow-lg hover:shadow-rose-200/50 
                          hover:bg-rose-100/50"
                      >
                        <Info 
                          size={16} 
                          className="text-rose-500" 
                        />
                      </button>
                    </td>
                  </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Enhanced Modal */}
      <AnimatePresence>
        {activePopover && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50"
            onClick={() => setActivePopover(null)}
          >
            <div className="fixed inset-0 bg-black/20 backdrop-blur-sm" />
            <div className="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-full max-w-lg p-4">
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.9, opacity: 0 }}
                className="relative bg-white rounded-xl shadow-2xl"
                onClick={e => e.stopPropagation()}
              >
                <motion.div 
                  className="bg-gradient-to-r from-rose-500 to-amber-500 rounded-t-xl p-4"
                  whileHover={{ backgroundPosition: ['0%', '100%'] }}
                  transition={{ duration: 3, repeat: Infinity, repeatType: "reverse" }}
                >
                  <div className="flex justify-between items-center">
                    <motion.h3 
                      initial={{ x: -20, opacity: 0 }}
                      animate={{ x: 0, opacity: 1 }}
                      className="text-lg font-semibold text-white"
                    >
                      {getStatisticInfo(activePopover).title}
                    </motion.h3>
                    <motion.button
                      whileHover={{ rotate: 90 }}
                      whileTap={{ scale: 0.9 }}
                      onClick={() => setActivePopover(null)}
                      className="p-1 hover:bg-white/20 rounded-full transition-colors"
                    >
                      <X size={20} className="text-white" />
                    </motion.button>
                  </div>
                </motion.div>
                
                <motion.div 
                  className="p-6 space-y-4 max-h-96 overflow-y-auto"
                  initial={{ y: 20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.1 }}
                >
                  <div className="prose prose-rose max-w-none">
                    <motion.h4 
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="text-rose-600 font-medium mb-2 flex items-center gap-2"
                    >
                      <span>Description</span>
                      <motion.div
                        animate={{ scale: [1, 1.2, 1] }}
                        transition={{ duration: 2, repeat: Infinity }}
                        className="w-1 h-1 rounded-full bg-rose-500"
                      />
                    </motion.h4>
                    <motion.p 
                      initial={{ y: 10, opacity: 0 }}
                      animate={{ y: 0, opacity: 1 }}
                      className="text-gray-600 text-sm leading-relaxed"
                    >
                      {getStatisticInfo(activePopover).description}
                    </motion.p>
                  </div>

                  <motion.div 
                    initial={{ y: 10, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.2 }}
                    className="mt-4 pt-4 border-t border-rose-100"
                  >
                    <h4 className="text-rose-600 font-medium mb-2">Medical Importance</h4>
                    <p className="text-gray-600 text-sm leading-relaxed">
                      {getStatisticInfo(activePopover).medicalContext}
                    </p>
                  </motion.div>

                  <motion.div 
                    initial={{ y: 10, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.3 }}
                    className="mt-4 pt-4 border-t border-rose-100"
                  >
                    <h4 className="text-rose-600 font-medium mb-2">Metric Value</h4>
                    <div className="flex items-center gap-2">
                      <motion.p
                        className="text-gray-900 font-mono text-lg"
                      >
                        {statistics[activePopover]} {/* Değeri doğrudan göster */}
                      </motion.p>
                      <motion.div
                        className="h-0.5 flex-1 rounded-full bg-gradient-to-r from-rose-500 to-amber-500 opacity-50"
                        animate={{ 
                          scaleX: [0, 1],
                          opacity: [0.3, 0.6] 
                        }}
                        transition={{ 
                          duration: 2, 
                          repeat: Infinity,
                          repeatType: "reverse" 
                        }}
                      />
                    </div>
                  </motion.div>
                </motion.div>
              </motion.div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default StatisticsTable;
