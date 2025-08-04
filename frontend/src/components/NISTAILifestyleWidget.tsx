import React, { useState, useEffect } from 'react';
import Card from './Card';

interface NISTLifecycleStage {
  name: string;
  riskScore: number;
  mitigationScore: number;
}

const NISTAILifestyleWidget: React.FC<{ hasInput?: boolean; analysisResults?: any }> = ({ hasInput = true, analysisResults }) => {
  const [nistStages, setNistStages] = useState<NISTLifecycleStage[]>([]);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; label: string; riskScore: number; mitigationScore: number } | null>(null);

  useEffect(() => {
    if (hasInput && analysisResults) {
      // Use the actual analysis results instead of making API calls
      const insights = analysisResults.insights || [];
      
      // Map analysis insights to NIST lifecycle stages
      const stages: NISTLifecycleStage[] = [
        {
          name: "Plan & Design",
          riskScore: Math.round((insights.find((i: any) => i.policy?.toLowerCase().includes('plan') || i.policy?.toLowerCase().includes('design'))?.compliance_score || 70) / 10),
          mitigationScore: Math.max(1, Math.round((insights.find((i: any) => i.policy?.toLowerCase().includes('plan') || i.policy?.toLowerCase().includes('design'))?.compliance_score || 70) / 10) - 2)
        },
        {
          name: "Collect & Process Data",
          riskScore: Math.round((insights.find((i: any) => i.policy?.toLowerCase().includes('data') || i.policy?.toLowerCase().includes('collect') || i.policy?.toLowerCase().includes('process'))?.compliance_score || 80) / 10),
          mitigationScore: Math.max(1, Math.round((insights.find((i: any) => i.policy?.toLowerCase().includes('data') || i.policy?.toLowerCase().includes('collect') || i.policy?.toLowerCase().includes('process'))?.compliance_score || 80) / 10) - 3)
        },
        {
          name: "Build & Use Model",
          riskScore: Math.round((insights.find((i: any) => i.policy?.toLowerCase().includes('model') || i.policy?.toLowerCase().includes('build') || i.policy?.toLowerCase().includes('use'))?.compliance_score || 90) / 10),
          mitigationScore: Math.max(1, Math.round((insights.find((i: any) => i.policy?.toLowerCase().includes('model') || i.policy?.toLowerCase().includes('build') || i.policy?.toLowerCase().includes('use'))?.compliance_score || 90) / 10) - 5)
        },
        {
          name: "Verify & Validate",
          riskScore: Math.round((insights.find((i: any) => i.policy?.toLowerCase().includes('verify') || i.policy?.toLowerCase().includes('validate'))?.compliance_score || 60) / 10),
          mitigationScore: Math.max(1, Math.round((insights.find((i: any) => i.policy?.toLowerCase().includes('verify') || i.policy?.toLowerCase().includes('validate'))?.compliance_score || 60) / 10) - 4)
        },
        {
          name: "Deploy and Use",
          riskScore: Math.round((insights.find((i: any) => i.policy?.toLowerCase().includes('deploy') || i.policy?.toLowerCase().includes('use'))?.compliance_score || 40) / 10),
          mitigationScore: Math.max(1, Math.round((insights.find((i: any) => i.policy?.toLowerCase().includes('deploy') || i.policy?.toLowerCase().includes('use'))?.compliance_score || 40) / 10) - 3)
        },
        {
          name: "Operate & Monitor",
          riskScore: Math.round((insights.find((i: any) => i.policy?.toLowerCase().includes('operate') || i.policy?.toLowerCase().includes('monitor'))?.compliance_score || 60) / 10),
          mitigationScore: Math.max(1, Math.round((insights.find((i: any) => i.policy?.toLowerCase().includes('operate') || i.policy?.toLowerCase().includes('monitor'))?.compliance_score || 60) / 10) - 3)
        }
      ];
      
      setNistStages(stages);
    }
  }, [hasInput, analysisResults]);

  const overallRiskScore = Math.round(
    nistStages.reduce((sum, s) => sum + s.riskScore, 0) / nistStages.length
  );

  const overallMitigationScore = Math.round(
    nistStages.reduce((sum, s) => sum + s.mitigationScore, 0) / nistStages.length
  );

  // Calculate radar chart points for each series
  const calculateRadarPoints = (scores: number[]) => {
    const centerX = 100;
    const centerY = 100;
    const radius = 60;
    const points: string[] = [];
    
    scores.forEach((score, index) => {
      const angle = (index * 2 * Math.PI) / scores.length - Math.PI / 2;
      const distance = (score / 10) * radius; // Scale from 0-10 to 0-60
      const x = centerX + distance * Math.cos(angle);
      const y = centerY + distance * Math.sin(angle);
      points.push(`${x},${y}`);
    });
    
    return points.join(' ');
  };

  const riskPoints = calculateRadarPoints(nistStages.map(s => s.riskScore));
  const mitigationPoints = calculateRadarPoints(nistStages.map(s => s.mitigationScore));

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

  if (nistStages.length === 0) {
    return (
      <Card className="bg-white rounded-2xl shadow-lg p-5 h-full custom-border p-2">
        <div className="flex flex-row items-center p-2 h-full">
          {/* Left: Placeholder content */}
          <div className="flex-1 flex flex-col justify-center">
            <h2 style={{ fontSize: '12', color: '#1975D4', fontWeight: 700 }}>
              NIST AI Lifecycle Framework
            </h2>
            <p className="text-base mt-1">
              Upload a document to view NIST AI lifecycle analysis with risk and mitigation scores.
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
            NIST AI Lifecycle Framework: {overallRiskScore}/{overallMitigationScore}
          </h2>
          <p className="text-base mt-1">
            Compare use case to the NIST AI Lifecycle framework and give overall scores about both of the risk and mitigation levels.
          </p>
          <div className="mt-2 text-xs text-gray-600">
            <div className="flex items-center gap-2 mb-1">
              <span className="w-3 h-3 bg-cyan-400 rounded-full"></span>
              <span>Risk Score</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 bg-blue-600 rounded-full"></span>
              <span>Mitigation Score</span>
            </div>
          </div>
        </div>
        {/* Right: Chart */}
        <div className="relative" style={{ width: 250, height: 250 }}>
          <svg width={250} height={250} viewBox="0 0 200 200">
            {/* Background circles for 0-10 scale */}
            <circle cx="100" cy="100" r="60" fill="none" stroke="#e5e7eb" strokeWidth="1" />
            <circle cx="100" cy="100" r="48" fill="none" stroke="#e5e7eb" strokeWidth="1" />
            <circle cx="100" cy="100" r="36" fill="none" stroke="#e5e7eb" strokeWidth="1" />
            <circle cx="100" cy="100" r="24" fill="none" stroke="#e5e7eb" strokeWidth="1" />
            <circle cx="100" cy="100" r="12" fill="none" stroke="#e5e7eb" strokeWidth="1" />
            
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
            
            {/* Radar polygons for each series */}
            <polygon
              points={riskPoints}
              fill="#4fd1c5"
              fillOpacity="0.15"
              stroke="#4fd1c5"
              strokeWidth="2"
            />
            <polygon
              points={mitigationPoints}
              fill="#3b82f6"
              fillOpacity="0.15"
              stroke="#3b82f6"
              strokeWidth="2"
            />
            
            {/* Data points for each series */}
            {nistStages.map((stage, index) => {
              const angle = (index * 2 * Math.PI) / nistStages.length - Math.PI / 2;
              
              // Risk score point
              const riskDistance = (stage.riskScore / 10) * 60;
              const riskX = 100 + riskDistance * Math.cos(angle);
              const riskY = 100 + riskDistance * Math.sin(angle);
              
              // Mitigation score point
              const mitigationDistance = (stage.mitigationScore / 10) * 60;
              const mitigationX = 100 + mitigationDistance * Math.cos(angle);
              const mitigationY = 100 + mitigationDistance * Math.sin(angle);
              
              return (
                <g key={index}>
                  <circle
                    cx={riskX}
                    cy={riskY}
                    r="3"
                    fill="#4fd1c5"
                    stroke="white"
                    strokeWidth="1"
                    className="cursor-pointer hover:r-5 transition-all"
                    onMouseEnter={() => setTooltip({ x: riskX, y: riskY, label: stage.name, riskScore: stage.riskScore, mitigationScore: stage.mitigationScore })}
                    onMouseLeave={() => setTooltip(null)}
                  />
                  <circle
                    cx={mitigationX}
                    cy={mitigationY}
                    r="3"
                    fill="#3b82f6"
                    stroke="white"
                    strokeWidth="1"
                    className="cursor-pointer hover:r-5 transition-all"
                    onMouseEnter={() => setTooltip({ x: mitigationX, y: mitigationY, label: stage.name, riskScore: stage.riskScore, mitigationScore: stage.mitigationScore })}
                    onMouseLeave={() => setTooltip(null)}
                  />
                </g>
              );
            })}
            
            {tooltip && (
              <foreignObject x={tooltip.x - 60} y={tooltip.y - 50} width={120} height={40} style={{ pointerEvents: 'none' }}>
                <div className="absolute bg-white border border-blue-200 rounded shadow p-2 z-10 text-center" style={{ fontSize: '9px', lineHeight: 1.2 }}>
                  <strong>{tooltip.label}</strong><br/>
                  Risk: {tooltip.riskScore}/10<br/>
                  Mitigation: {tooltip.mitigationScore}/10
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

export default NISTAILifestyleWidget;
