import React, { useState, useEffect } from 'react';
import Card from './Card';

interface OECDScoreWidgetProps {
  hasInput?: boolean;
  analysisResults?: any;
}

interface OECDPrinciple {
  name: string;
  score: number;
  color: string;
}

interface YearData {
  2020: number;
  2021: number;
  2022: number;
}

const OECDScoreWidget: React.FC<OECDScoreWidgetProps> = ({ hasInput = true, analysisResults }) => {
  const [selectedPrinciple, setSelectedPrinciple] = useState<OECDPrinciple | null>(null);
  const [hoveredLegend, setHoveredLegend] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; label: string; value: number; year: number } | null>(null);

  // OECD AI Principles with three-year data based on the radar chart image
  const [oecdPrinciples, setOecdPrinciples] = useState<OECDPrinciple[]>([]);
  const [yearData, setYearData] = useState<{ [key: string]: YearData }>({});

  useEffect(() => {
    if (hasInput && analysisResults) {
      // Use the actual analysis results instead of making API calls
      const insights = analysisResults.insights || [];
      
      // Extract OECD-related scores from the analysis insights
      const principles: OECDPrinciple[] = [
        {
          name: "Inclusive and Sustainability",
          score: insights.find((i: any) => i.policy?.toLowerCase().includes('inclusive') || i.policy?.toLowerCase().includes('sustainability'))?.compliance_score || 55,
          color: "#4CAF50"
        },
        {
          name: "Fairness and Privacy",
          score: insights.find((i: any) => i.policy?.toLowerCase().includes('fairness') || i.policy?.toLowerCase().includes('privacy'))?.compliance_score || 50,
          color: "#2196F3"
        },
        {
          name: "Transparency and explainability",
          score: insights.find((i: any) => i.policy?.toLowerCase().includes('transparency') || i.policy?.toLowerCase().includes('explainability'))?.compliance_score || 45,
          color: "#9C27B0"
        },
        {
          name: "Robustness, security, and safety",
          score: insights.find((i: any) => i.policy?.toLowerCase().includes('robustness') || i.policy?.toLowerCase().includes('security') || i.policy?.toLowerCase().includes('safety'))?.compliance_score || 35,
          color: "#FF9800"
        },
        {
          name: "AI",
          score: insights.find((i: any) => i.policy?.toLowerCase().includes('ai'))?.compliance_score || 30,
          color: "#F44336"
        },
        {
          name: "Accountability",
          score: insights.find((i: any) => i.policy?.toLowerCase().includes('accountability'))?.compliance_score || 45,
          color: "#4CAF50"
        }
      ];
      
      setOecdPrinciples(principles);
      
      // Generate three-year data based on current analysis scores
      const yearComparison: { [key: string]: YearData } = {};
      principles.forEach(principle => {
        const baseScore = principle.score;
        yearComparison[principle.name] = {
          2020: Math.max(0, baseScore - 15),
          2021: Math.max(0, baseScore - 5),
          2022: baseScore
        };
      });
      
      setYearData(yearComparison);
    }
  }, [hasInput, analysisResults]);

  const overallScore = Math.round(
    oecdPrinciples.reduce((sum, p) => sum + p.score, 0) / oecdPrinciples.length
  );

  // Calculate radar chart points for each year
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

  // Get scores for a specific year
  const getYearScores = (year: number) => {
    return oecdPrinciples.map(principle => yearData[principle.name]?.[year as keyof YearData] || principle.score);
  };

  const radarPoints2020 = calculateRadarPoints(getYearScores(2020));
  const radarPoints2021 = calculateRadarPoints(getYearScores(2021));
  const radarPoints2022 = calculateRadarPoints(getYearScores(2022));

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
      <Card className="bg-white rounded-2xl shadow-lg p-5 h-full custom-border p-2">
        <div className="flex flex-row items-center p-2 h-full">
          {/* Left: Placeholder content */}
          <div className="flex-1 flex flex-col justify-center">
            <h2 style={{ fontSize: '12', color: '#1975D4', fontWeight: 700 }}>
              OECD Values-Based Principles
            </h2>
            <p className="text-base mt-1">
              Upload a document to view OECD compliance analysis across three years.
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

  return (
    <Card className="bg-white rounded-2xl shadow-lg p-5 h-full custom-border p-2 h-full">
      <div className="flex flex-row items-center p-2 h-full">
        {/* Left: Score and description */}
        <div className="flex-1 flex flex-col justify-center">
          <h2 style={{ fontSize: '12', color: '#1975D4', fontWeight: 700 }}>
            OECD Values-Based Principles: {overallScore}
          </h2>
          <p className="text-base mt-1">
            Compare use case to the OECD AI Risk Framework and give an overall score about the compliance level across three years.
          </p>
          <div className="mt-2 text-xs text-gray-600">
            <div className="flex items-center gap-2 mb-1">
              <span className="w-3 h-3 bg-blue-500 rounded-full"></span>
              <span>2020</span>
            </div>
            <div className="flex items-center gap-2 mb-1">
              <span className="w-3 h-3 bg-red-500 rounded-full"></span>
              <span>2021</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 bg-cyan-500 rounded-full"></span>
              <span>2022</span>
            </div>
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
            {/* Radar polygons for each year */}
            <polygon
              points={radarPoints2020}
              fill="#3B82F6"
              fillOpacity="0.15"
              stroke="#3B82F6"
              strokeWidth="2"
            />
            <polygon
              points={radarPoints2021}
              fill="#EF4444"
              fillOpacity="0.15"
              stroke="#EF4444"
              strokeWidth="2"
            />
            <polygon
              points={radarPoints2022}
              fill="#06B6D4"
              fillOpacity="0.15"
              stroke="#06B6D4"
              strokeWidth="2"
            />
            {/* Data points for each year */}
            {[2020, 2021, 2022].map((year) => {
              const yearColor = year === 2020 ? "#3B82F6" : year === 2021 ? "#EF4444" : "#06B6D4";
              return oecdPrinciples.map((principle, index) => {
                const angle = (index * 2 * Math.PI) / oecdPrinciples.length - Math.PI / 2;
                const score = yearData[principle.name]?.[year as keyof YearData] || principle.score;
                const distance = (score / 100) * 60;
                const x = 100 + distance * Math.cos(angle);
                const y = 100 + distance * Math.sin(angle);
                return (
                  <circle
                    key={`${year}-${index}`}
                    cx={x}
                    cy={y}
                    r="3"
                    fill={yearColor}
                    stroke="white"
                    strokeWidth="1"
                    className="cursor-pointer hover:r-5 transition-all"
                    onMouseEnter={() => setTooltip({ x, y, label: principle.name, value: score, year })}
                    onMouseLeave={() => setTooltip(null)}
                  />
                );
              });
            })}
            {tooltip && (
              <foreignObject x={tooltip.x - 60} y={tooltip.y - 50} width={120} height={40} style={{ pointerEvents: 'none' }}>
                <div className="absolute bg-white border border-blue-200 rounded shadow p-2 z-10 text-center" style={{ fontSize: '9px', lineHeight: 1.2 }}>
                  <strong>{tooltip.label}</strong><br/>
                  {tooltip.year}: {tooltip.value}%
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

export default OECDScoreWidget; 
