import React from 'react';
import Card from './Card';

interface NISTLifecycleStage {
  name: string;
  riskLevel: number;
  mitigabilityLevel: number;
}

const NISTAILifestyleWidget: React.FC = () => {
  // Example dynamic data (replace with props or context as needed)
  const nistStages: NISTLifecycleStage[] = [
    { name: 'Plan & Design', riskLevel: 7, mitigabilityLevel: 5 },
    { name: 'Collect & Process Data', riskLevel: 10, mitigabilityLevel: 8 },
    { name: 'Build & Use Model', riskLevel: 9, mitigabilityLevel: 7 },
    { name: 'Verify & Validate', riskLevel: 6, mitigabilityLevel: 4 },
    { name: 'Deploy and Use', riskLevel: 3, mitigabilityLevel: 2 },
    { name: 'Operate & Monitor', riskLevel: 5, mitigabilityLevel: 6 },
  ];

  const size = 250; // px (figure size)
  const center = size / 2;
  const radius = 90;
  const levels = 5;
  const maxValue = 10;
  const angleStep = (2 * Math.PI) / nistStages.length;

  // Dotted background
  const dots = [];
  for (let x = 0; x < size; x += 20) {
    for (let y = 0; y < size; y += 20) {
      dots.push(<circle key={`dot-${x}-${y}`} cx={x} cy={y} r={1} fill="#e5e7eb" />);
    }
  }

  // Grid lines (cube effect)
  const gridLines = [];
  for (let l = 1; l <= levels; l++) {
    const r = (radius * l) / levels;
    const points: [number, number][] = [];
    for (let i = 0; i < nistStages.length; i++) {
      const angle = i * angleStep - Math.PI / 2;
      points.push([
        center + r * Math.cos(angle),
        center + r * Math.sin(angle),
      ]);
    }
    // Connect points
    for (let i = 0; i < points.length; i++) {
      const [x1, y1] = points[i];
      const [x2, y2] = points[(i + 1) % points.length];
      gridLines.push(
        <line
          key={`grid-${l}-${i}`}
          x1={x1}
          y1={y1}
          x2={x2}
          y2={y2}
          stroke="#bdbdbd"
          strokeDasharray="6,4"
          strokeWidth={1}
        />
      );
    }
  }

  // Axes
  const axes = nistStages.map((stage, i) => {
    const angle = i * angleStep - Math.PI / 2;
    return (
      <line
        key={`axis-${i}`}
        x1={center}
        y1={center}
        x2={center + radius * Math.cos(angle)}
        y2={center + radius * Math.sin(angle)}
        stroke="#bdbdbd"
        strokeWidth={1}
      />
    );
  });

  // Labels
  const labelRadius = radius + 12;
  const labels = nistStages.map((stage, i) => {
    const angle = i * angleStep - Math.PI / 2;
    const x = center + labelRadius * Math.cos(angle);
    const y = center + labelRadius * Math.sin(angle);
    let anchor = 'middle';
    if (angle < -Math.PI / 2 || angle > Math.PI / 2) anchor = 'end';
    if (angle > -Math.PI / 2 && angle < Math.PI / 2) anchor = 'start';
    return (
      <text
        key={`label-${i}`}
        x={x}
        y={y}
        fontSize={12}
        fill="#222"
        textAnchor={anchor}
        alignmentBaseline="middle"
        fontFamily="inherit"
      >
        {stage.name}
      </text>
    );
  });

  // Polygons
  const getPolygonPoints = (levels: number[]) =>
    levels
      .map((val, i) => {
        const angle = i * angleStep - Math.PI / 2;
        const r = (val / maxValue) * radius;
        return `${center + r * Math.cos(angle)},${center + r * Math.sin(angle)}`;
      })
      .join(' ');

  const riskPoints = getPolygonPoints(nistStages.map(s => s.riskLevel));
  const mitigabilityPoints = getPolygonPoints(nistStages.map(s => s.mitigabilityLevel));

  // Calculate overall score (example)
  const overallScore = Math.round(
    nistStages.reduce((sum, s) => sum + s.riskLevel, 0) / nistStages.length * 10
  );

  return (
    <Card className="custom-border flex flex-row items-center p-4 h-full">
      {/* Left: Score and description */}
      <div className="flex-1 flex flex-col justify-center">
        <h2 style={{ fontSize: '12', color: '#1975D4', fontWeight: 700 }}>NIST AI Lifecycle: {overallScore}</h2>
        <p className="text-base mt-2">
          Compare use case rto the NIST AI Lifecycle framework and give overall scores about both of the risk and mitigation levels
        </p>
      </div>
      {/* Right: Chart */}
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size}>
          {/* Dotted background */}
          {dots}
          {/* Grid lines */}
          {gridLines}
          {/* Axes */}
          {axes}
          {/* Polygons */}
          <polygon
            points={riskPoints}
            fill="#4fd1c5"
            fillOpacity={0.5}
            stroke="#4fd1c5"
            strokeWidth={3}
          />
          <polygon
            points={mitigabilityPoints}
            fill="#63b3ed"
            fillOpacity={0.5}
            stroke="#63b3ed"
            strokeWidth={3}
          />
          {/* Labels */}
          {labels}
          {/* Grid value labels */}
          {[...Array(levels + 1)].map((_, l) => (
            l > 0 && (
              <text
                key={`value-label-${l}`}
                x={center}
                y={center - (radius * l) / levels - 4}
                fontSize={12}
                fill="#222"
                textAnchor="middle"
                fontFamily="inherit"
              >
                {l * (maxValue / levels)}
              </text>
            )
          ))}
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

export default NISTAILifestyleWidget;
