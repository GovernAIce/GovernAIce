import React, { useState, useEffect } from 'react';
import Card from './Card';
import Button from './Button';

interface OverallScoreWidgetProps {
  overallScore?: number;
  uploadedFile: File | null;
  analysisComplete: boolean;
}

interface CompliancePrinciple {
  name: string;
  score: number;
  color: string;
}

const OverallScoreWidget: React.FC<OverallScoreWidgetProps> = ({ 
  overallScore, 
  uploadedFile, 
  analysisComplete 
}) => {
  const [selectedPrinciple, setSelectedPrinciple] = useState<CompliancePrinciple | null>(null);
  const [hoveredLegend, setHoveredLegend] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; label: string; value: number } | null>(null);

  // Generate compliance principles based on overall score
  const [compliancePrinciples, setCompliancePrinciples] = useState<CompliancePrinciple[]>([]);

  useEffect(() => {
    if (analysisComplete && overallScore !== undefined) {
      // Create 6 compliance areas based on the radar chart image
      const principles: CompliancePrinciple[] = [
        {
          name: "Inclusive and Sustainability",
          score: overallScore,
          color: overallScore >= 80 ? "#4CAF50" : overallScore >= 60 ? "#FF9800" : "#F44336"
        },
        {
          name: "Fairness and Privacy",
          score: Math.max(0, overallScore - 10),
          color: overallScore >= 80 ? "#2196F3" : overallScore >= 60 ? "#FF9800" : "#F44336"
        },
        {
          name: "Transparency and explainability",
          score: Math.max(0, overallScore + 5),
          color: overallScore >= 80 ? "#9C27B0" : overallScore >= 60 ? "#FF9800" : "#F44336"
        },
        {
          name: "Robustness, security, and safety",
          score: Math.max(0, overallScore - 15),
          color: overallScore >= 80 ? "#4CAF50" : overallScore >= 60 ? "#FF9800" : "#F44336"
        },
        {
          name: "AI",
          score: Math.max(0, overallScore - 5),
          color: overallScore >= 80 ? "#2196F3" : overallScore >= 60 ? "#FF9800" : "#F44336"
        },
        {
          name: "Accountability",
          score: Math.max(0, overallScore - 8),
          color: overallScore >= 80 ? "#4CAF50" : overallScore >= 60 ? "#FF9800" : "#F44336"
        }
      ];
      setCompliancePrinciples(principles);
    }
  }, [analysisComplete, overallScore]);

  const calculatedOverallScore = Math.round(
    compliancePrinciples.reduce((sum, p) => sum + p.score, 0) / compliancePrinciples.length
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

  const radarPoints = calculateRadarPoints(compliancePrinciples.map(p => p.score));

  // Calculate axis lines
  const getAxisLines = () => {
    const lines: { x1: number; y1: number; x2: number; y2: number }[] = [];
    const centerX = 100;
    const centerY = 100;
    const radius = 60;
    
    compliancePrinciples.forEach((_, index) => {
      const angle = (index * 2 * Math.PI) / compliancePrinciples.length - Math.PI / 2;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      lines.push({ x1: centerX, y1: centerY, x2: x, y2: y });
    });
    
    return lines;
  };

  const axisLines = getAxisLines();

  if (!uploadedFile) {
    return (
      <Card className="bg-white rounded-2xl shadow-lg p-5 h-full custom-border p-2">
        <div className="flex flex-row items-center p-2 h-full">
          {/* Left: Placeholder content */}
          <div className="flex-1 flex flex-col justify-center">
            <h2 style={{ fontSize: '12', color: '#1975D4', fontWeight: 700 }}>
              Overall Score
            </h2>
            <p className="text-base mt-1">
              Upload a document to view the overall compliance score and analysis.
            </p>
          </div>
          {/* Right: Placeholder chart area */}
          <div className="relative" style={{ width: 250, height: 250 }}>
            <div className="w-full h-full flex items-center justify-center">
              <div className="text-center text-gray-400">
                <div className="w-32 h-32 border-2 border-dashed border-gray-300 rounded-full flex items-center justify-center mb-2">
                  <span className="text-xs">Chart</span>
                </div>
                <p className="text-xs">Upload to see analysis</p>
              </div>
            </div>
            <img
              src="/icons/info.svg"
              alt="Info"
              className="absolute top-2 right-2 w-6 h-6 opacity-70"
            />
          </div>
        </div>
      </Card>
    );
  }

  if (!analysisComplete) {
    return (
      <Card className="bg-white rounded-2xl shadow-lg p-5 h-full custom-border p-2">
        <div className="flex flex-row items-center p-2 h-full">
          {/* Left: Loading content */}
          <div className="flex-1 flex flex-col justify-center">
            <h2 style={{ fontSize: '12', color: '#1975D4', fontWeight: 700 }}>
              Overall Score
            </h2>
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-4"></div>
            <div className="text-gray-500 text-sm">
              Analyzing compliance...
            </div>
          </div>
          {/* Right: Loading chart area */}
          <div className="relative" style={{ width: 250, height: 250 }}>
            <div className="w-full h-full flex items-center justify-center">
              <div className="text-center text-gray-400">
                <div className="w-32 h-32 border-2 border-dashed border-gray-300 rounded-full flex items-center justify-center mb-2 animate-pulse">
                  <span className="text-xs">Loading...</span>
                </div>
                <p className="text-xs">Processing data</p>
              </div>
            </div>
            <img
              src="/icons/info.svg"
              alt="Info"
              className="absolute top-2 right-2 w-6 h-6 opacity-70"
            />
          </div>
        </div>
      </Card>
    );
  }

  const score = overallScore || 0;
  const getScoreLevel = (score: number) => {
    if (score >= 80) return 'Excellent';
    if (score >= 60) return 'Good';
    if (score >= 40) return 'Fair';
    return 'Poor';
  };

  const scoreLevel = getScoreLevel(score);

  return (
    <Card className="bg-white rounded-2xl shadow-lg p-5 h-full custom-border p-2 h-full">
      <div className="flex flex-row items-center p-2 h-full">
        {/* Left: Score and description */}
        <div className="flex-1 flex flex-col justify-center">
          <h2 style={{ fontSize: '12', color: '#1975D4', fontWeight: 700 }}>
            Overall Score: {calculatedOverallScore}
          </h2>
          <p className="text-base mt-1">
          Compare use case to the target policies and give an overall score about the compliance level.
          </p>
          <div className="mt-1 text-sm">
            <span className={`font-medium ${
              score >= 80 ? 'text-green-600' : 
              score >= 60 ? 'text-yellow-600' : 'text-red-600'
            }`}>
              Level: {scoreLevel}
            </span>
          </div>
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
            {compliancePrinciples.map((principle, index) => {
              const angle = (index * 2 * Math.PI) / compliancePrinciples.length - Math.PI / 2;
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
      </div>
    </Card>
  );
};

export default OverallScoreWidget; 
