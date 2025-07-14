import React, { useState } from 'react';
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

const maxScore = 10; // Now the chart max is 10
const chartWidth = 400; // width for the bar area (adjust as needed)
const xStep = chartWidth / maxScore;
const maxPossibleScore = 8;
const totalScore = Math.round(
  (riskBarData.reduce((sum, d) => sum + d.riskScore + d.mitigationScore, 0)) / maxPossibleScore
);

// Change vertical step from 40 to 48 for more spacing
const yStep = 48;
const leftMargin = 100; // Increased from 60 to 100 for more label space

const EURiskLevelFrameworkWidget: React.FC = () => {
  const [hoveredLegend, setHoveredLegend] = useState<string | null>(null);
  const hasInput = true;

  if (!hasInput) {
    return (
      <Card className="custom-border relative p-2 h-full">
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
    <Card className="custom-border flex flex-row items-center p-4 h-full">
      {/* Left: Title, Score, Description */}
      <div className="flex-1 flex flex-col justify-center min-w-[220px]">
        <h2 style={{ fontSize: '12', color: '#1975D4', fontWeight: 700 }}>
          EU Risk Level<br />Framework <span className="ml-2" style={{ color: '#1975D4', fontWeight: 700 }}>{totalScore}/8</span>
        </h2>
        <p className="text-base mt-2" style={{ color: '#000000' }}>
          Compare use case to the EU AI Risk and give overall scores about both of the risk and mitigation levels.
        </p>
      </div>
      {/* Right: Chart */}
      <div className="relative flex-1 flex flex-col justify-center items-center h-full">
        {/* Info icon */}
        <img
          src="/icons/info.svg"
          alt="Info"
          className="absolute top-2 right-2 w-6 h-6 opacity-70"
        />
        {/* Legend */}
        <div className="flex flex-row gap-4 mb-2 mt-2 items-center">
          <div className="flex items-center gap-1">
            <span className="inline-block w-4 h-4 rounded-full" style={{ background: '#ffe082' }}></span>
            <span className="text-xs" style={{ color: '#000000' }}>Risk Score</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="inline-block w-4 h-4 rounded-full" style={{ background: '#ff9800' }}></span>
            <span className="text-xs" style={{ color: '#000000' }}>Mitigation Score</span>
          </div>
        </div>
        {/* Bar Chart */}
        <svg viewBox={`0 0 ${chartWidth + leftMargin + 20} ${riskBarData.length * yStep + 60}`} className="w-full" style={{ height: `${riskBarData.length * yStep + 60}px` }}>
          {/* Dotted grid background */}
          {Array.from({ length: 11 }).map((_, i) => (
            <g key={i}>
              <line
                x1={leftMargin + i * xStep}
                y1={20}
                x2={leftMargin + i * xStep}
                y2={riskBarData.length * yStep + 20}
                stroke="#000000"
                strokeDasharray="2,4"
                strokeWidth={1}
              />
            </g>
          ))}
          {riskBarData.map((_, row) => (
            Array.from({ length: 11 }).map((_, col) => (
              <circle
                key={`dot-${row}-${col}`}
                cx={leftMargin + col * xStep}
                cy={46 + row * yStep}
                r={1.2}
                fill="#000000"
                opacity={0.5}
              />
            ))
          ))}
          {/* Y labels */}
          {riskBarData.map((d, i) => {
            const [main, ...rest] = d.label.split('-');
            return (
              <text
                key={d.label}
                x={0}
                y={55 + i * yStep}
                fontSize={12}
                fill="#000000"
                alignmentBaseline="middle"
              >
                {main}
                {rest.length > 0 && (
                  <tspan x={0} dy={14}>{rest.join('-')}</tspan>
                )}
              </text>
            );
          })}
          {/* Bars */}
          {riskBarData.map((d, i) => (
            <g key={d.label}>
              {/* Risk Score Bar */}
              <rect
                x={leftMargin}
                y={40 + i * yStep}
                width={d.riskScore * xStep}
                height={12}
                fill="#ffe082"
                rx={4}
              />
              {/* Mitigation Score Bar */}
              <rect
                x={leftMargin}
                y={54 + i * yStep}
                width={d.mitigationScore * xStep}
                height={12}
                fill="#ff9800"
                rx={4}
              />
            </g>
          ))}
          {/* X axis labels */}
          {Array.from({ length: 11 }).map((_, i) => (
            <text
              key={i}
              x={leftMargin + i * xStep}
              y={riskBarData.length * yStep + 30}
              fontSize={12}
              fill="#000000"
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
