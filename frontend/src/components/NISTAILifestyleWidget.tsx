import React, { useState } from 'react';
import Card from './Card';

interface NISTLifecycleStage {
  name: string;
  riskLevel: number;
  mitigabilityLevel: number;
  color: string;
  description: string;
}

const NISTAILifestyleWidget: React.FC = () => {
  const [selectedStage, setSelectedStage] = useState<NISTLifecycleStage | null>(null);

  // NIST AI Lifecycle stages with risk and mitigability scores
  const nistStages: NISTLifecycleStage[] = [
    {
      name: 'Govern & Map',
      riskLevel: 65,
      mitigabilityLevel: 80,
      color: '#4CAF50',
      description: 'Organizational governance and risk mapping'
    },
    {
      name: 'Measure & Manage',
      riskLevel: 72,
      mitigabilityLevel: 75,
      color: '#2196F3',
      description: 'Risk measurement and management processes'
    },
    {
      name: 'Design & Develop',
      riskLevel: 58,
      mitigabilityLevel: 85,
      color: '#FF9800',
      description: 'AI system design and development'
    },
    {
      name: 'Verify & Validate',
      riskLevel: 70,
      mitigabilityLevel: 78,
      color: '#9C27B0',
      description: 'Testing, verification and validation'
    },
    {
      name: 'Deploy & Operate',
      riskLevel: 68,
      mitigabilityLevel: 82,
      color: '#F44336',
      description: 'Deployment and operational monitoring'
    },
    {
      name: 'Monitor & Maintain',
      riskLevel: 75,
      mitigabilityLevel: 70,
      color: '#607D8B',
      description: 'Ongoing monitoring and maintenance'
    }
  ];

  const overallRisk = Math.round(nistStages.reduce((sum, s) => sum + s.riskLevel, 0) / nistStages.length);
  const overallMitigability = Math.round(nistStages.reduce((sum, s) => sum + s.mitigabilityLevel, 0) / nistStages.length);

  // Calculate dual-axis chart points
  const calculateChartPoints = (stages: NISTLifecycleStage[]) => {
    const centerX = 100;
    const centerY = 100;
    const radius = 60;
    const riskPoints: string[] = [];
    const mitigabilityPoints: string[] = [];
    
    stages.forEach((stage, index) => {
      const angle = (index * 2 * Math.PI) / stages.length - Math.PI / 2;
      
      // Risk points (inner polygon)
      const riskDistance = (stage.riskLevel / 100) * radius;
      const riskX = centerX + riskDistance * Math.cos(angle);
      const riskY = centerY + riskDistance * Math.sin(angle);
      riskPoints.push(`${riskX},${riskY}`);
      
      // Mitigability points (outer polygon)
      const mitigabilityDistance = (stage.mitigabilityLevel / 100) * radius;
      const mitigabilityX = centerX + mitigabilityDistance * Math.cos(angle);
      const mitigabilityY = centerY + mitigabilityDistance * Math.sin(angle);
      mitigabilityPoints.push(`${mitigabilityX},${mitigabilityY}`);
    });
    
    return {
      risk: riskPoints.join(' '),
      mitigability: mitigabilityPoints.join(' ')
    };
  };

  const chartPoints = calculateChartPoints(nistStages);

  // Calculate axis lines
  const getAxisLines = () => {
    const lines: { x1: number; y1: number; x2: number; y2: number }[] = [];
    const centerX = 100;
    const centerY = 100;
    const radius = 60;
    
    nistStages.forEach((_, index) => {
      const angle = (index * 2 * Math.PI) / nistStages.length - Math.PI / 2;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      lines.push({ x1: centerX, y1: centerY, x2: x, y2: y });
    });
    
    return lines;
  };

  const axisLines = getAxisLines();

  const getRiskColor = (risk: number) => {
    if (risk >= 80) return '#F44336'; // High risk - Red
    if (risk >= 60) return '#FF9800'; // Medium risk - Orange
    return '#4CAF50'; // Low risk - Green
  };

  const getMitigabilityColor = (mitigability: number) => {
    if (mitigability >= 80) return '#4CAF50'; // High mitigability - Green
    if (mitigability >= 60) return '#2196F3'; // Medium mitigability - Blue
    return '#FF9800'; // Low mitigability - Orange
  };

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
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h3 className="text-xl text-[#1975d4] font-bold mb-2">NIST AI Lifecycle</h3>
          <p className="text-gray-600 text-sm">Upload a document and select a country to view NIST AI lifecycle analysis</p>
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
          <h3 className="text-xl text-[#1975d4] font-bold">NIST AI Lifecycle</h3>
          <div className="flex items-center justify-center gap-4 mt-2">
            <div className="text-center">
              <span className="text-lg font-bold text-red-600">{overallRisk}</span>
              <span className="text-xs text-gray-600">/ 100</span>
              <p className="text-xs text-gray-600">Risk Level</p>
            </div>
            <div className="text-center">
              <span className="text-lg font-bold text-green-600">{overallMitigability}</span>
              <span className="text-xs text-gray-600">/ 100</span>
              <p className="text-xs text-gray-600">Mitigability</p>
            </div>
          </div>
        </div>

        {/* Dual-Axis Chart */}
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
              
              {/* Mitigability polygon (outer) */}
              <polygon
                points={chartPoints.mitigability}
                fill="#2196F3"
                fillOpacity="0.2"
                stroke="#2196F3"
                strokeWidth="2"
                strokeDasharray="5,5"
              />
              
              {/* Risk polygon (inner) */}
              <polygon
                points={chartPoints.risk}
                fill="#F44336"
                fillOpacity="0.3"
                stroke="#F44336"
                strokeWidth="2"
              />
              
              {/* Data points for risk */}
              {nistStages.map((stage, index) => {
                const angle = (index * 2 * Math.PI) / nistStages.length - Math.PI / 2;
                const riskDistance = (stage.riskLevel / 100) * 60;
                const riskX = 100 + riskDistance * Math.cos(angle);
                const riskY = 100 + riskDistance * Math.sin(angle);
                
                return (
                  <circle
                    key={`risk-${index}`}
                    cx={riskX}
                    cy={riskY}
                    r="3"
                    fill={getRiskColor(stage.riskLevel)}
                    stroke="white"
                    strokeWidth="1"
                    className="cursor-pointer hover:r-4 transition-all"
                    onMouseEnter={() => setSelectedStage(stage)}
                    onMouseLeave={() => setSelectedStage(null)}
                  />
                );
              })}
              
              {/* Data points for mitigability */}
              {nistStages.map((stage, index) => {
                const angle = (index * 2 * Math.PI) / nistStages.length - Math.PI / 2;
                const mitigabilityDistance = (stage.mitigabilityLevel / 100) * 60;
                const mitigabilityX = 100 + mitigabilityDistance * Math.cos(angle);
                const mitigabilityY = 100 + mitigabilityDistance * Math.sin(angle);
                
                return (
                  <circle
                    key={`mitigability-${index}`}
                    cx={mitigabilityX}
                    cy={mitigabilityY}
                    r="2"
                    fill={getMitigabilityColor(stage.mitigabilityLevel)}
                    stroke="white"
                    strokeWidth="1"
                    className="cursor-pointer hover:r-3 transition-all"
                    onMouseEnter={() => setSelectedStage(stage)}
                    onMouseLeave={() => setSelectedStage(null)}
                  />
                );
              })}
            </svg>
            
            {/* Center legend */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center bg-white rounded-lg p-2 shadow-sm">
                <div className="flex items-center gap-2 text-xs">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <span>Risk</span>
                  <div className="w-3 h-3 bg-blue-500 rounded-full border border-dashed"></div>
                  <span>Mitigability</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Stage Legend */}
        <div className="mt-4 space-y-2 max-h-32 overflow-y-auto">
          {nistStages.map((stage, index) => (
            <div
              key={index}
              className={`flex items-center justify-between p-2 rounded text-xs transition-colors ${
                selectedStage?.name === stage.name ? 'bg-blue-50' : 'hover:bg-gray-50'
              }`}
              onMouseEnter={() => setSelectedStage(stage)}
              onMouseLeave={() => setSelectedStage(null)}
            >
              <div className="flex items-center gap-2">
                <div 
                  className="w-3 h-3 rounded-full" 
                  style={{ backgroundColor: stage.color }}
                />
                <span className="font-medium">{stage.name}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-red-600 font-bold">{stage.riskLevel}%</span>
                <span className="text-blue-600 font-bold">{stage.mitigabilityLevel}%</span>
              </div>
            </div>
          ))}
        </div>

        {/* Hover Details */}
        {selectedStage && (
          <div className="mt-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
            <h4 className="text-sm font-bold text-blue-800 mb-1">{selectedStage.name}</h4>
            <p className="text-xs text-blue-700 mb-2">{selectedStage.description}</p>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-xs text-red-600 font-medium">Risk:</span>
                <div className="flex-1 bg-gray-200 rounded-full h-2">
                  <div 
                    className="h-2 rounded-full transition-all duration-300"
                    style={{ 
                      width: `${selectedStage.riskLevel}%`,
                      backgroundColor: getRiskColor(selectedStage.riskLevel)
                    }}
                  />
                </div>
                <span className="text-xs font-bold text-red-600">{selectedStage.riskLevel}%</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-blue-600 font-medium">Mitigability:</span>
                <div className="flex-1 bg-gray-200 rounded-full h-2">
                  <div 
                    className="h-2 rounded-full transition-all duration-300"
                    style={{ 
                      width: `${selectedStage.mitigabilityLevel}%`,
                      backgroundColor: getMitigabilityColor(selectedStage.mitigabilityLevel)
                    }}
                  />
                </div>
                <span className="text-xs font-bold text-blue-600">{selectedStage.mitigabilityLevel}%</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default NISTAILifestyleWidget;
