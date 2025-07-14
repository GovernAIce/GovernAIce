import React, { useState } from 'react';
import Card from './Card';

interface ExcellenciesMajorGapsWidgetProps {
  uploadedFile: File | null;
  selectedCountries: string[];
  insights: any[];
}

const MAX_POINTS = 5;
const MAX_POLICIES = 5;

const ExcellenciesMajorGapsWidget: React.FC<ExcellenciesMajorGapsWidgetProps> = ({ uploadedFile, selectedCountries, insights }) => {
  const hasResults = insights && insights.length > 0 && uploadedFile && selectedCountries.length > 0;
  const [expandedExcellencies, setExpandedExcellencies] = useState<{ [key: number]: boolean }>({});
  const [expandedGaps, setExpandedGaps] = useState<{ [key: number]: boolean }>({});

  const toggleExcellencies = (idx: number) => {
    setExpandedExcellencies(prev => ({ ...prev, [idx]: !prev[idx] }));
  };
  const toggleGaps = (idx: number) => {
    setExpandedGaps(prev => ({ ...prev, [idx]: !prev[idx] }));
  };

  // Only show up to MAX_POLICIES policies
  const displayedInsights = insights.slice(0, MAX_POLICIES);

  return (
    <Card className="custom-border flex flex-col p-4 h-full min-h-[300px]">
      <img
        src="/icons/info.svg"
        alt="Info"
        className="absolute top-3 right-3 w-4 h-4 cursor-pointer"
      />
      {!hasResults ? (
        <div className="flex flex-col items-center justify-center h-full text-center">
          <h3 className="text-[#1975d4] text-xl font-bold mb-2">Excellencies & Major Gaps</h3>
          <p className="text-black text-base leading-relaxed">
            Upload a document and select countries to view excellencies and major gaps based on your analysis.
          </p>
        </div>
      ) : (
        <div className="flex flex-row gap-10 mt-4">
          <div className="flex-1">
            <h3 className="text-[#1975d4] text-xl font-bold mb-2">Excellencies</h3>
            {displayedInsights.map((insight, idx) => {
              const points = insight.excellent_points || [];
              const isExpanded = expandedExcellencies[idx];
              const showToggle = points.length > MAX_POINTS;
              const displayPoints = isExpanded ? points : points.slice(0, MAX_POINTS);
              return (
                <div key={idx} className="mb-2">
                  <div className="font-semibold text-sm mb-1">{insight.policy}</div>
                  <div className={isExpanded ? "max-h-32 overflow-y-auto pr-1" : ""}>
                    <ul className="list-disc list-inside text-black text-xs">
                      {points.length > 0 ? (
                        displayPoints.map((pt: string, i: number) => <li key={i}>{pt}</li>)
                      ) : (
                        <li>No excellencies found.</li>
                      )}
                    </ul>
                  </div>
                  {showToggle && (
                    <button
                      className="text-xs text-blue-600 underline mt-1 focus:outline-none"
                      onClick={() => toggleExcellencies(idx)}
                    >
                      {isExpanded ? 'Show less' : `Show more (${points.length - MAX_POINTS} more)`}
                    </button>
                  )}
                </div>
              );
            })}
          </div>
          <div className="flex-1">
            <h3 className="text-[#1975d4] text-xl font-bold mb-2">Major Gaps</h3>
            {displayedInsights.map((insight, idx) => {
              const points = insight.major_gaps || [];
              const isExpanded = expandedGaps[idx];
              const showToggle = points.length > MAX_POINTS;
              const displayPoints = isExpanded ? points : points.slice(0, MAX_POINTS);
              return (
                <div key={idx} className="mb-2">
                  <div className="font-semibold text-sm mb-1">{insight.policy}</div>
                  <div className={isExpanded ? "max-h-32 overflow-y-auto pr-1" : ""}>
                    <ul className="list-disc list-inside text-black text-xs">
                      {points.length > 0 ? (
                        displayPoints.map((pt: string, i: number) => <li key={i}>{pt}</li>)
                      ) : (
                        <li>No major gaps found.</li>
                      )}
                    </ul>
                  </div>
                  {showToggle && (
                    <button
                      className="text-xs text-blue-600 underline mt-1 focus:outline-none"
                      onClick={() => toggleGaps(idx)}
                    >
                      {isExpanded ? 'Show less' : `Show more (${points.length - MAX_POINTS} more)`}
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </Card>
  );
};

export default ExcellenciesMajorGapsWidget; 
