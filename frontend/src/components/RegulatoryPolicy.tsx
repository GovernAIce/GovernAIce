import React, { useEffect, useState } from 'react';
import Card from './Card';

interface ChartItem {
  label: string;
  values: { [year: string]: number };
}

const yearColors: { [key: string]: string } = {
  2020: '#A78BFA',
  2021: '#F87171',
  2022: '#38BDF8',
  2023: '#FACC15',
  2024: '#60A5FA',
  2025: '#34D399',
};

const maxHeight = 100;

const RegulatoryPolicy = () => {
  const [chartData, setChartData] = useState<ChartItem[]>([]);

  useEffect(() => {
    fetch('http://localhost:5001/api/relevant-policies')  // adjust port if needed
      .then(res => res.json())
      .then(data => setChartData(data))
      .catch(err => console.error("Failed to fetch policy data", err));
  }, []);

  const barWidth = 10;
  const barGap = 6;
  const groupGap = 40;

  const years = chartData.length > 0 ? Object.keys(chartData[0].values) : [];

  return (
    <Card className="p-6 rounded-2xl border border-gray-200 shadow-md w-full max-w-3xl mx-auto">
      <h2 className="text-2xl font-bold text-blue-600 mb-1">Relevant Policies & Regulators</h2>
      <p className="text-sm text-gray-500 mb-4">(pop up relevant information)</p>

      <svg width="100%" height="200" viewBox="0 0 400 200" preserveAspectRatio="xMidYMid meet">
        <line x1="0" y1="180" x2="400" y2="180" stroke="#000" strokeWidth="1" />

        {chartData.map((data, groupIndex) => {
          const groupX = groupIndex * (years.length * (barWidth + barGap) + groupGap);
          return years.map((year, i) => {
            const value = data.values[year];
            const barHeight = (value / maxHeight) * 160;
            const x = groupX + i * (barWidth + barGap) + 50;
            const y = 180 - barHeight;
            return (
              <rect
                key={`${data.label}-${year}`}
                x={x}
                y={y}
                width={barWidth}
                height={barHeight}
                fill={yearColors[year]}
                rx="2"
              />
            );
          });
        })}

        {chartData.map((data, i) => {
          const groupX = i * (years.length * (barWidth + barGap) + groupGap);
          const center = groupX + 50 + (years.length * (barWidth + barGap)) / 2 - barGap;
          return (
            <text key={data.label} x={center} y={195} fontSize="10" textAnchor="middle" fill="#374151">
              {data.label}
            </text>
          );
        })}
      </svg>

      <div className="flex flex-wrap justify-center mt-4 gap-4 text-sm text-gray-700">
        {Object.entries(yearColors).map(([year, color]) => (
          <div key={year} className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
            {year}
          </div>
        ))}
      </div>
    </Card>
  );
};

export default RegulatoryPolicy;
