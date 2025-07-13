import React from 'react';
import Card from './Card';

interface EURiskBarData {
  label: string;
  riskScore: number;
  mitigationScore: number;
}

const riskBarData: EURiskBarData[] = [
  {
    label: 'Level 1-Minimal / No Risk',
    riskScore: 5.5,
    mitigationScore: 6,
  },
  {
    label: 'Level 2-Limited Risk',
    riskScore: 3,
    mitigationScore: 5,
  },
  {
    label: 'Level 3-High Risk',
    riskScore: 2,
    mitigationScore: 2,
  },
  {
    label: 'Level 4-Unacceptable Risk',
    riskScore: 0,
    mitigationScore: 0,
  },
];

const maxScore = Math.max(...riskBarData.map(d => Math.max(d.riskScore, d.mitigationScore, 6)));

const EURiskLevelFrameworkWidget: React.FC = () => {
  // Check if we have input data (for now, always show data, but this can be connected to actual input)
  const hasInput = true; // This would be connected to actual document/country selection

  if (!hasInput) {
    return (
      <Card className="custom-border relative p-4 h-full">
        <img
          src="/icons/info.svg"
          alt="Info"
          className="absolute top-2 right-2 w-4 h-4 cursor-pointer"
        />
        <div className="flex flex-col items-center justify-center h-full text-center">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4">
            <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <h3 className="text-xl text-[#1975d4] font-bold mb-2">EU Risk Level Framework</h3>
          <p className="text-gray-600 text-sm">Upload a document and select a country to view EU risk level analysis</p>
        </div>
      </Card>
    );
  }

  return (
    <Card className="custom-border relative p-4 h-full flex flex-row items-stretch">
      {/* Left: Title and Legend */}
      <div className="flex flex-col justify-between w-1/3 min-w-[180px] pr-4">
        <div>
          <h3 className="text-xl text-[#1975d4] font-bold mb-4">EU Risk Level Framework</h3>
        </div>
        <div className="flex flex-col gap-2 mt-2">
          <div className="flex items-center gap-2">
            <span className="inline-block w-4 h-4 rounded-full" style={{ background: '#ffe082' }}></span>
            <span className="text-xs text-gray-700">Risk Score</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-block w-4 h-4 rounded-full" style={{ background: '#ff9800' }}></span>
            <span className="text-xs text-gray-700">Mitigation Score</span>
          </div>
        </div>
      </div>

      {/* Right: Bar Chart */}
      <div className="flex-1 flex flex-col justify-center">
        <svg viewBox={`0 0 ${maxScore * 40 + 120} 220`} className="w-full h-56">
          {/* Grid dots */}
          {Array.from({ length: 7 }).map((_, i) => (
            <g key={i}>
              <line
                x1={60 + i * 40}
                y1={20}
                x2={60 + i * 40}
                y2={200}
                stroke="#e5e7eb"
                strokeDasharray="2,4"
                strokeWidth={1}
              />
            </g>
          ))}
          {/* Y labels */}
          {riskBarData.map((d, i) => (
            <text
              key={d.label}
              x={0}
              y={55 + i * 40}
              fontSize={14}
              fill="#222"
              alignmentBaseline="middle"
            >
              {d.label}
            </text>
          ))}
          {/* Bars */}
          {riskBarData.map((d, i) => (
            <g key={d.label}>
              {/* Risk Score Bar */}
              <rect
                x={60}
                y={40 + i * 40}
                width={d.riskScore * 40}
                height={12}
                fill="#ffe082"
                rx={4}
              />
              {/* Mitigation Score Bar */}
              <rect
                x={60}
                y={54 + i * 40}
                width={d.mitigationScore * 40}
                height={12}
                fill="#ff9800"
                rx={4}
              />
            </g>
          ))}
          {/* X axis labels */}
          {Array.from({ length: 7 }).map((_, i) => (
            <text
              key={i}
              x={60 + i * 40}
              y={215}
              fontSize={12}
              fill="#888"
              textAnchor="middle"
            >
              {i}
            </text>
          ))}
        </svg>
      </div>
    </Card>
  );
};

export default EURiskLevelFrameworkWidget;
