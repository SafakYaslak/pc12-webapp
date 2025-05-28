import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { HistogramData } from '../../types';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface HistogramChartProps {
  data: HistogramData;
}

const HistogramChart: React.FC<HistogramChartProps> = ({ data }) => {
  // Return empty state if no data is available
  if (!data.labels?.length || !data.data?.length) {
    return (
      <div className="flex items-center justify-center h-64 bg-gradient-to-br from-rose-50/50 to-amber-50/50 rounded-xl">
        <p className="text-gray-500">No histogram data available</p>
      </div>
    );
  }

  // Configure chart data
  const chartData = {
    labels: data.labels,
    datasets: [{
      label: data.yAxisLabel || 'Frequency',
      data: data.data,
      backgroundColor: (context: any) => {
        const ctx = context.chart.ctx;
        const gradient = ctx.createLinearGradient(0, 0, 0, 300);
        gradient.addColorStop(0, 'rgba(244, 63, 94, 0.7)');    // rose-500
        gradient.addColorStop(0.5, 'rgba(251, 146, 60, 0.7)'); // orange-400
        gradient.addColorStop(1, 'rgba(251, 191, 36, 0.7)');   // amber-400
        return gradient;
      },
      borderColor: 'rgba(255, 255, 255, 0.9)',
      borderWidth: 2,
      borderRadius: 8,
      hoverBackgroundColor: (context: any) => {
        const ctx = context.chart.ctx;
        const gradient = ctx.createLinearGradient(0, 0, 0, 300);
        gradient.addColorStop(0, 'rgba(244, 63, 94, 0.9)');
        gradient.addColorStop(0.5, 'rgba(251, 146, 60, 0.9)');
        gradient.addColorStop(1, 'rgba(251, 191, 36, 0.9)');
        return gradient;
      },
      hoverBorderColor: 'white',
      hoverBorderWidth: 3,
    }]
  };

  // Update options configuration
  const options: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    layout: {
      padding: {
        top: 10,    // Reduced from 20
        right: 15,  // Reduced from 20
        bottom: 10, // Reduced from 20
        left: 15    // Reduced from 20
      }
    },
    plugins: {
      legend: {
        display: true,
        position: 'top' as const,
        labels: {
          font: {
            family: 'Inter',
            size: 12,
            weight: '500'
          },
          color: '#374151',
          usePointStyle: true,
          padding: 20,
        }
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleFont: {
          family: 'Inter',
          size: 14,
          weight: '600'
        },
        bodyFont: {
          family: 'Inter',
          size: 13,
          weight: '400'
        },
        padding: 12,
        cornerRadius: 8,
        displayColors: false,
        callbacks: {
          title: (items) => `Range: ${items[0].label}`,
          label: (context) => `Count: ${context.parsed.y}`,
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
          drawBorder: false,
        },
        ticks: {
          font: {
            family: 'Inter',
            size: 12,
            weight: '400'
          },
          color: '#374151',
          padding: 8,
          maxTicksLimit: 8
        }
      },
      x: {
        grid: {
          display: false
        },
        ticks: {
          font: {
            family: 'Inter',
            size: 12,
            weight: '400'
          },
          color: '#374151',
          maxRotation: 45,
          minRotation: 45,
          autoSkip: true,
          maxTicksLimit: 10
        }
      }
    },
    animation: {
      duration: 1500,
      easing: 'easeInOutQuart'
    },
    transitions: {
      active: {
        animation: {
          duration: 300
        }
      }
    },
    interaction: {
      intersect: false,
      mode: 'index'
    }
  };

  return (
    <div className="relative w-full h-[280px] p-4 bg-gradient-to-br from-rose-50/30 via-orange-50/30 to-amber-50/30 backdrop-blur-sm rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 group">
      <div className="absolute inset-0 bg-gradient-to-br from-rose-500/5 via-orange-500/5 to-amber-500/5 opacity-50 group-hover:opacity-70 transition-opacity duration-300 rounded-xl" />
      <div className="relative w-full h-full">
        <Bar 
          data={chartData} 
          options={{
            ...options,
            maintainAspectRatio: false,
            responsive: true,
            layout: {
              padding: {
                top: 5,
                right: 10,
                bottom: 5,
                left: 10
              }
            }
          }} 
        />
      </div>
    </div>
  );
};

export default HistogramChart;