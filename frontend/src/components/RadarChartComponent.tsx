import React, { useState, useEffect } from 'react';
import Card from './Card';

type Year = 2020 | 2021 | 2022;

interface DataItem {
  category: string;
  2020: number;
  2021: number;
  2022: number;
}

const years: Year[] = [2020, 2021, 2022];

const colors: Record<Year, string> = {
  2020: '#3B82F6', // blue
  2021: '#EF4444', // red
  2022: '#06B6D4', // cyan
};

const RadarChartComponent = () => {
  const [data, setData] = useState<DataItem[]>([]);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; label: string; value: number } | null>(null);

  useEffect(() => {
            fetch('http://localhost:5002/api/radar-data') // ✅ Adjust port if needed
      .then((res) => res.json())
      .then((json) => setData(json))
      .catch((err) => console.error('Failed to fetch radar data', err));
  }, []);

  const centerX = 100;
  const centerY = 100;
  const radius = 60;

  const calculatePoints = (year: Year) => {
    return data.map((d, i) => {
      const angle = (i * 2 * Math.PI) / data.length - Math.PI / 2;
      const dist = (d[year] / 100) * radius;
      const x = centerX + dist * Math.cos(angle);
      const y = centerY + dist * Math.sin(angle);
      return { x, y };
    });
  };

  const axisLines = data.map((_, i) => {
    const angle = (i * 2 * Math.PI) / data.length - Math.PI / 2;
    const x = centerX + radius * Math.cos(angle);
    const y = centerY + radius * Math.sin(angle);
    return { x1: centerX, y1: centerY, x2: x, y2: y };
  });

  const overallScore =
    data.length > 0
      ? Math.round(data.reduce((acc, curr) => acc + curr[2022], 0) / data.length)
      : 0;

  return (
    <Card className="p-6 rounded-2xl border border-gray-200 shadow-md w-full max-w-5xl mx-auto">
      {/* Header */}
      <div className="mb-4">
        <h2 className="text-3xl font-bold text-blue-600">Overall Score: {overallScore}</h2>
        <p className="text-gray-600 mt-2">
          Compare use case to the target policies and give an overall score about the compliance level
        </p>
      </div>

      {/* Chart */}
      <div className="flex justify-center">
        <svg width={250} height={250} viewBox="0 0 200 200">
          {/* Background Circles */}
          {[15, 30, 45, 60].map((r) => (
            <circle
              key={r}
              cx={centerX}
              cy={centerY}
              r={r}
              fill="none"
              stroke="#e5e7eb"
              strokeWidth="1"
            />
          ))}

          {/* Axis Lines */}
          {axisLines.map((line, idx) => (
            <line key={idx} x1={line.x1} y1={line.y1} x2={line.x2} y2={line.y2} stroke="#d1d5db" />
          ))}

          {/* Polygons */}
          {years.map((year) => {
            const points = calculatePoints(year).map((p) => `${p.x},${p.y}`).join(' ');
            return (
              <polygon
                key={year}
                points={points}
                fill={colors[year]}
                fillOpacity="0.15"
                stroke={colors[year]}
                strokeWidth={2}
              />
            );
          })}

          {/* Data Points with Tooltips */}
          {years.map((year) =>
            calculatePoints(year).map((point, i) => (
              <circle
                key={`${year}-${i}`}
                cx={point.x}
                cy={point.y}
                r="5"
                fill={colors[year]}
                stroke="white"
                strokeWidth="1"
                className="cursor-pointer hover:r-7 transition-all"
                onMouseEnter={() =>
                  setTooltip({
                    x: point.x,
                    y: point.y,
                    label: data[i].category,
                    value: data[i][year],
                  })
                }
                onMouseLeave={() => setTooltip(null)}
              />
            ))
          )}

          {tooltip && (
            <foreignObject
              x={tooltip.x - 60}
              y={tooltip.y - 50}
              width={120}
              height={40}
              style={{ pointerEvents: 'none' }}
            >
              <div
                className="absolute bg-white border border-blue-200 rounded shadow p-2 z-10 text-center"
                style={{ fontSize: '9px', lineHeight: 1.2 }}
              >
                <strong>{tooltip.label}</strong>: {tooltip.value}%
              </div>
            </foreignObject>
          )}
        </svg>
      </div>

      {/* Legend */}
      <div className="flex justify-end mt-6 gap-6 text-sm text-gray-700">
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 bg-blue-500 rounded-full" /> 2020
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 bg-red-500 rounded-full" /> 2021
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 bg-cyan-500 rounded-full" /> 2022
        </div>
      </div>
    </Card>
  );
};

export default RadarChartComponent;
