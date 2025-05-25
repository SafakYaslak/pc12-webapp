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
  ChartData
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
      <div className="flex items-center justify-center h-full">
        <p className="text-gray-500">No histogram data available</p>
      </div>
    );
  }

  // Configure chart data
  const chartData: ChartData<'bar'> = {
    labels: data.labels,
    datasets: [{
      label: data.yAxisLabel || 'Frequency',
      data: data.data,
      backgroundColor: data.backgroundColor || 'rgba(54, 162, 235, 0.6)',
      borderColor: data.borderColor || 'rgba(54, 162, 235, 1)',
      borderWidth: 1,
      borderRadius: 4, // Rounded corners for bars
      borderSkipped: false, // Apply border radius to all sides
    }]
  };

  // Configure chart options
  const options: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: {
        display: !!data.chartTitle,
        text: data.chartTitle,
        font: {
          size: 16,
          weight: 'bold'
        },
        padding: {
          top: 10,
          bottom: 20
        }
      },
      legend: {
        display: false
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
        padding: 12,
        callbacks: {
          title: (items) => {
            if (!items.length) return '';
            const item = items[0];
            return `Range: ${item.label}`;
          },
          label: (context) => {
            return `${data.yAxisLabel || 'Count'}: ${context.parsed.y}`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        },
        title: {
          display: true,
          text: data.yAxisLabel || 'Frequency',
          font: {
            size: 12,
            weight: 'bold'
          }
        },
        ticks: {
          font: {
            size: 10
          }
        }
      },
      x: {
        grid: {
          display: false
        },
        title: {
          display: true,
          text: data.xAxisLabel || 'Value',
          font: {
            size: 12,
            weight: 'bold'
          }
        },
        ticks: {
          font: {
            size: 10
          }
        }
      }
    },
    animation: {
      duration: 1000
    }
  };

  return (
    <div className="w-full h-full">
      <Bar 
        data={chartData} 
        options={options}
        height={400}
      />
    </div>
  );
};

export default HistogramChart;