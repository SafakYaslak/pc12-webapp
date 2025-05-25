import React from 'react';

interface Statistics {
  [key: string]: string | number;
}

interface StatisticsTableProps {
  statistics: Statistics;
}

const formatLabel = (key: string) => {
  // CamelCase veya snake_case etiketleri daha okunabilir hale getir
  return key
    // camelCase -> camel Case
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    // snake_case -> snake case
    .replace(/_/g, ' ')
    // İlk harfi büyük yap
    .replace(/^\w/, c => c.toUpperCase());
};

const StatisticsTable: React.FC<StatisticsTableProps> = ({ statistics }) => {
  const entries = Object.entries(statistics);

  if (entries.length === 0) {
    return (
      <div className="text-gray-500 p-4">
        İstatistik bulunamadı.
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-lg border border-gray-200">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th
              scope="col"
              className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
            >
              METRICS
            </th>
            <th
              scope="col"
              className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
            >
              VALUES
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {entries.map(([key, value]) => (
            <tr key={key}>
              <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                {formatLabel(key)}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                {value}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default StatisticsTable;
