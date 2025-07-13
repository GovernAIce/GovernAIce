import React, { useState } from 'react';
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

  // OECD AI Principles with sample scores (these would come from analysis results)
  const oecdPrinciples: OECDPrinciple[] = [
    { name: 'Inclusive Growth & Sustainability', score: 85, color: '#4CAF50' },
    { name: 'Fairness & Privacy', score: 78, color: '#2196F3' },
    { name: 'Transparency & Explainability', score: 72, color: '#FF9800' },
    { name: 'Robustness, Security & Safety', score: 88, color: '#9C27B0' },
    { name: 'Accountability', score: 81, color: '#F44336' }
  ];

  const overallScore = Math.round(oecdPrinciples.reduce((sum, p) => sum + p.score, 0) / oecdPrinciples.length);

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
    <Card className="custom-border relative p-4 h-full">
      <img
        src="/icons/info.svg"
        alt="Info"
        className="absolute top-2 right-2 w-4 h-4 cursor-pointer"
      />
      <div className="flex flex-col h-full">
        <div className="text-center mb-4">
          <h3 className="text-xl text-[#1975d4] font-bold">OECD AI Principles</h3>
          <div className="flex items-center justify-center gap-2 mt-1">
            <span className="text-2xl font-bold text-[#1975d4]">{overallScore}</span>
            <span className="text-sm text-gray-600">/ 100</span>
          </div>
          <p className="text-xs text-gray-600 mt-1">Overall Compliance Score</p>
        </div>

        {/* Radar Chart */}
        <div className="flex-1 flex justify-center items-center">
          <div className="relative">
            <svg viewBox="0 0 200 200" className="w-48 h-48">
              {/* Background circles */}
              <circle cx="100" cy="100" r="60" fill="none" stroke="#e5e7eb" strokeWidth="1"/>
              <circle cx="100" cy="100" r="45" fill="none" stroke="#e5e7eb" strokeWidth="1"/>
              <circle cx="100" cy="100" r="30" fill="none" stroke="#e5e7eb" strokeWidth="1"/>
              <circle cx="100" cy="100" r="15" fill="none" stroke="#e5e7eb" strokeWidth="1"/>
              
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
                    r="3"
                    fill={principle.color}
                    stroke="white"
                    strokeWidth="1"
                    className="cursor-pointer hover:r-4 transition-all"
                    onMouseEnter={() => setSelectedPrinciple(principle)}
                    onMouseLeave={() => setSelectedPrinciple(null)}
                  />
                );
              })}
            </svg>
            
            {/* Center score */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="text-lg font-bold text-[#1975d4]">{overallScore}</div>
                <div className="text-xs text-gray-500">Score</div>
              </div>
            </div>
          </div>
        </div>

        {/* Principle Legend */}
        <div className="mt-4 space-y-2">
          {oecdPrinciples.map((principle, index) => (
            <div
              key={index}
              className={`flex items-center justify-between p-2 rounded text-xs transition-colors ${
                selectedPrinciple?.name === principle.name ? 'bg-blue-50' : 'hover:bg-gray-50'
              }`}
              onMouseEnter={() => setSelectedPrinciple(principle)}
              onMouseLeave={() => setSelectedPrinciple(null)}
            >
              <div className="flex items-center gap-2">
                <div 
                  className="w-3 h-3 rounded-full" 
                  style={{ backgroundColor: principle.color }}
                />
                <span className="font-medium">{principle.name}</span>
              </div>
              <span className="font-bold">{principle.score}%</span>
            </div>
          ))}
        </div>

        {/* Hover Details */}
        {selectedPrinciple && (
          <div className="mt-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
            <h4 className="text-sm font-bold text-blue-800 mb-1">{selectedPrinciple.name}</h4>
            <div className="flex items-center gap-2 mb-2">
              <div className="flex-1 bg-gray-200 rounded-full h-2">
                <div 
                  className="h-2 rounded-full transition-all duration-300"
                  style={{ 
                    width: `${selectedPrinciple.score}%`,
                    backgroundColor: selectedPrinciple.color 
                  }}
                />
              </div>
              <span className="text-xs font-bold">{selectedPrinciple.score}%</span>
            </div>
            <p className="text-xs text-blue-700">
              {selectedPrinciple.score >= 80 
                ? "Excellent compliance with this principle"
                : selectedPrinciple.score >= 60
                ? "Good compliance with room for improvement"
                : "Needs attention to meet compliance requirements"
              }
            </p>
          </div>
        )}
      </div>
    </Card>
  );
};

export default OECDScoreWidget; 
