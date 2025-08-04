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
    <Card className="bg-white rounded-2xl shadow-lg p-5 h-full custom-border flex flex-col p-2 h-full">
      <img
        src="/icons/info.svg"
        alt="Info"
        className="absolute top-2 right-2 w-4 h-4 cursor-pointer"
      />
      {!hasResults ? (
        <div className="flex flex-row gap-4 h-full">
          <div className="flex-1 overflow-y-auto">
            <h3 className="text-[#1975d4] text-lg font-bold mb-2 sticky top-0 bg-white pb-1">Excellencies</h3>
            <div className="space-y-2">
              {Array.from({ length: 3 }).map((_, idx) => (
                <div key={idx} className="bg-gray-50 rounded-lg p-2 border border-dashed border-gray-300">
                  <div className="font-semibold text-sm mb-1 text-gray-400 italic">Policy {idx + 1}</div>
                  <ul className="list-disc list-inside text-gray-400 text-xs space-y-0.5">
                    <li>Upload document to see excellencies</li>
                  </ul>
                </div>
              ))}
            </div>
          </div>
          <div className="flex-1 overflow-y-auto">
            <h3 className="text-[#1975d4] text-lg font-bold mb-2 sticky top-0 bg-white pb-1">Major Gaps</h3>
            <div className="space-y-2">
              {Array.from({ length: 3 }).map((_, idx) => (
                <div key={idx} className="bg-red-50 rounded-lg p-2 border border-dashed border-red-300">
                  <div className="font-semibold text-sm mb-1 text-gray-400 italic">Policy {idx + 1}</div>
                  <ul className="list-disc list-inside text-gray-400 text-xs space-y-0.5">
                    <li>Upload document to see gaps</li>
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="flex flex-row gap-4 h-full">
          <div className="flex-1 overflow-y-auto">
            <h3 className="text-[#1975d4] text-lg font-bold mb-2 sticky top-0 bg-white pb-1">Excellencies</h3>
            <div className="space-y-2">
              {displayedInsights.map((insight, idx) => {
                const points = insight.excellent_points || [];
                const isExpanded = expandedExcellencies[idx];
                const showToggle = points.length > MAX_POINTS;
                const displayPoints = isExpanded ? points : points.slice(0, MAX_POINTS);
                return (
                  <div key={idx} className="bg-gray-50 rounded-lg p-2">
                    <div className="font-semibold text-sm mb-1 text-gray-800">{insight.policy}</div>
                    <div className={isExpanded ? "max-h-40 overflow-y-auto pr-1" : ""}>
                      <ul className="list-disc list-inside text-black text-xs space-y-0.5">
                        {points.length > 0 ? (
                          displayPoints.map((pt: string, i: number) => <li key={i}>{pt}</li>)
                        ) : (
                          <li>No excellencies found.</li>
                        )}
                      </ul>
                    </div>
                    {showToggle && (
                      <button
                        className="text-xs text-blue-600 underline mt-1 focus:outline-none hover:text-blue-800"
                        onClick={() => toggleExcellencies(idx)}
                      >
                        {isExpanded ? 'Show less' : `Show more (${points.length - MAX_POINTS} more)`}
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
          <div className="flex-1 overflow-y-auto">
            <h3 className="text-[#1975d4] text-lg font-bold mb-2 sticky top-0 bg-white pb-1">Major Gaps</h3>
            <div className="space-y-2">
              {displayedInsights.map((insight, idx) => {
                const points = insight.major_gaps || [];
                const isExpanded = expandedGaps[idx];
                const showToggle = points.length > MAX_POINTS;
                const displayPoints = isExpanded ? points : points.slice(0, MAX_POINTS);
                return (
                  <div key={idx} className="bg-red-50 rounded-lg p-2">
                    <div className="font-semibold text-sm mb-1 text-gray-800">{insight.policy}</div>
                    <div className={isExpanded ? "max-h-40 overflow-y-auto pr-1" : ""}>
                      <ul className="list-disc list-inside text-black text-xs space-y-0.5">
                        {points.length > 0 ? (
                          displayPoints.map((pt: string, i: number) => <li key={i}>{pt}</li>)
                        ) : (
                          <li>No major gaps found.</li>
                        )}
                      </ul>
                    </div>
                    {showToggle && (
                      <button
                        className="text-xs text-red-600 underline mt-1 focus:outline-none hover:text-red-800"
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
        </div>
      )}
    </Card>
  );
};

export default ExcellenciesMajorGapsWidget; 
