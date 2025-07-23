import React, { useState, useEffect } from 'react';
import Card from './Card';

interface OECDScoreWidgetProps {
  hasInput?: boolean;
}

interface OECDPrinciple {
  name: string;
  score: number;
  color: string;
}

const OECDScoreWidget: React.FC<OECDScoreWidgetProps> = ({ hasInput = true }) => {
  const [selectedPrinciple, setSelectedPrinciple] = useState<OECDPrinciple | null>(null);
  const [hoveredLegend, setHoveredLegend] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; label: string; value: number } | null>(null);

  // OECD AI Principles with sample scores (these would come from analysis results)
 const [oecdPrinciples, setOecdPrinciples] = useState<OECDPrinciple[]>([]);

useEffect(() => {
  const fetchScores = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/oecd-scores');
      const data = await response.json();
      setOecdPrinciples(data);
    } catch (error) {
      console.error("Error fetching OECD scores:", error);
    }
  };

  fetchScores();
}, []);

  const overallScore = Math.round(
  oecdPrinciples.reduce((sum, p) => sum + p.score, 0) / oecdPrinciples.length
);

  // Calculate radar chart points
  const calculateRadarPoints = (scores: number[]) => {
    const centerX = 100;
    const centerY = 100;
    const radius = 60;
    const points: string[] = [];
    
    scores.forEach((score, index) => {
      const angle = (index * 2 * Math.PI) / scores.length - Math.PI / 2;
      const distance = (score / 100) * radius;
      const x = centerX + distance * Math.cos(angle);
      const y = centerY + distance * Math.sin(angle);
      points.push(`${x},${y}`);
    });
    
    return points.join(' ');
  };

  const radarPoints = calculateRadarPoints(oecdPrinciples.map(p => p.score));

  // Calculate axis lines
  const getAxisLines = () => {
    const lines: { x1: number; y1: number; x2: number; y2: number }[] = [];
    const centerX = 100;
    const centerY = 100;
    const radius = 60;
    
    oecdPrinciples.forEach((_, index) => {
      const angle = (index * 2 * Math.PI) / oecdPrinciples.length - Math.PI / 2;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      lines.push({ x1: centerX, y1: centerY, x2: x, y2: y });
    });
    
    return lines;
  };

  const axisLines = getAxisLines();

  if (!hasInput) {
    return (
      <Card className="custom-border relative p-4 h-full flex flex-col items-center justify-center">
        <img
          src="/icons/info.svg"
          alt="Info"
          className="w-8 h-8 mb-3 opacity-60"
        />
        <div className="text-center text-gray-500 text-sm">
          Upload a document and select countries to see OECD compliance analysis.
        </div>
      </Card>
    );
  }

  return (
    <Card className="custom-border flex flex-row items-center p-4 h-full">
      {/* Left: Score and description */}
      <div className="flex-1 flex flex-col justify-center">
        <h2 style={{ fontSize: '12', color: '#1975D4', fontWeight: 700 }}>OECD AI Principles: {overallScore}</h2>
        <p className="text-base mt-2">
          Compare use case to the OECD AI Principles and get an overall compliance score across all five principles.
        </p>
      </div>
      {/* Right: Chart */}
      <div className="relative" style={{ width: 250, height: 250 }}>
        <svg width={250} height={250} viewBox="0 0 200 200">
          {/* Background circles */}
          <circle cx="100" cy="100" r="60" fill="none" stroke="#e5e7eb" strokeWidth="1" />
          <circle cx="100" cy="100" r="45" fill="none" stroke="#e5e7eb" strokeWidth="1" />
          <circle cx="100" cy="100" r="30" fill="none" stroke="#e5e7eb" strokeWidth="1" />
          <circle cx="100" cy="100" r="15" fill="none" stroke="#e5e7eb" strokeWidth="1" />
          {/* Axis lines */}
          {axisLines.map((line, index) => (
            <line
              key={index}
              x1={line.x1}
              y1={line.y1}
              x2={line.x2}
              y2={line.y2}
              stroke="#d1d5db"
              strokeWidth="1"
            />
          ))}
          {/* Radar polygon */}
          <polygon
            points={radarPoints}
            fill="#1975d4"
            fillOpacity="0.3"
            stroke="#1975d4"
            strokeWidth="2"
          />
          {/* Data points */}
          {oecdPrinciples.map((principle, index) => {
            const angle = (index * 2 * Math.PI) / oecdPrinciples.length - Math.PI / 2;
            const distance = (principle.score / 100) * 60;
            const x = 100 + distance * Math.cos(angle);
            const y = 100 + distance * Math.sin(angle);
            return (
              <circle
                key={index}
                cx={x}
                cy={y}
                r="6"
                fill={principle.color}
                stroke="white"
                strokeWidth="1"
                className="cursor-pointer hover:r-8 transition-all"
                onMouseEnter={() => setTooltip({ x, y, label: principle.name, value: principle.score })}
                onMouseLeave={() => setTooltip(null)}
              />
            );
          })}
          {tooltip && (
            <foreignObject x={tooltip.x - 60} y={tooltip.y - 50} width={120} height={40} style={{ pointerEvents: 'none' }}>
              <div className="absolute bg-white border border-blue-200 rounded shadow p-2 z-10 text-center" style={{ fontSize: '9px', lineHeight: 1.2 }}>
                <strong>{tooltip.label}</strong>: {tooltip.value}%
              </div>
            </foreignObject>
          )}
        </svg>
        <img
          src="/icons/info.svg"
          alt="Info"
          className="absolute top-2 right-2 w-6 h-6 opacity-70"
        />
      </div>
    </Card>
  );
};

export default OECDScoreWidget; 