import React, { useState } from 'react';
import Card from './Card';

interface PolicyInsight {
  policy: string;
  compliance_score: number;
  policy_details: string;
  excellent_points: string[];
  major_gaps: string[];
}

interface PolicyAnalysisResultsProps {
  insights: PolicyInsight[];
  onClose?: () => void;
}

const PolicyAnalysisResults: React.FC<PolicyAnalysisResultsProps> = ({ insights, onClose }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [expandedPolicy, setExpandedPolicy] = useState<string | null>(null);

  const handlePrevious = () => {
    setCurrentIndex(prev => prev > 0 ? prev - 1 : insights.length - 1);
    setExpandedPolicy(null);
  };

  const handleNext = () => {
    setCurrentIndex(prev => prev < insights.length - 1 ? prev + 1 : 0);
    setExpandedPolicy(null);
  };

  const toggleExpanded = (policy: string) => {
    setExpandedPolicy(expandedPolicy === policy ? null : policy);
  };

  if (!insights || insights.length === 0) {
    return null;
  }

  const currentInsight = insights[currentIndex];

  return (
    <Card className="custom-border p-4 h-auto min-h-[300px]">
      <div className="flex flex-col h-full">
        {/* Header with navigation */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg text-[#1975d4] font-bold">Analysis Results</h3>
          <div className="flex items-center gap-2">
            <button
              onClick={handlePrevious}
              className="p-1 rounded-full hover:bg-gray-100 transition-colors"
              disabled={insights.length <= 1}
            >
              <img src="/icons/Arrow10.svg" alt="Previous" className="w-4 h-4 rotate-180" />
            </button>
            <span className="text-xs text-gray-600">
              {currentIndex + 1} of {insights.length}
            </span>
            <button
              onClick={handleNext}
              className="p-1 rounded-full hover:bg-gray-100 transition-colors"
              disabled={insights.length <= 1}
            >
              <img src="/icons/Arrow10.svg" alt="Next" className="w-4 h-4" />
            </button>
            {onClose && (
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-gray-600 text-lg font-bold"
              >
                ×
              </button>
            )}
          </div>
        </div>

        {/* Policy Card */}
        <div className="flex-1 bg-white rounded-lg border border-gray-200 p-4">
          <div className="mb-3">
            <h4 className="font-semibold text-gray-800 mb-2">{currentInsight.policy}</h4>
            
            {/* Compliance Score */}
            <div className="flex items-center gap-2 mb-3">
              <span className="text-sm text-gray-600">Compliance Score:</span>
              <div className="flex items-center gap-1">
                <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    className={`h-full rounded-full transition-all duration-300 ${
                      currentInsight.compliance_score >= 80 ? 'bg-green-500' :
                      currentInsight.compliance_score >= 60 ? 'bg-yellow-500' :
                      'bg-red-500'
                    }`}
                    style={{ width: `${currentInsight.compliance_score}%` }}
                  ></div>
                </div>
                <span className="text-sm font-medium">
                  {currentInsight.compliance_score}%
                </span>
              </div>
            </div>

            {/* Policy Details */}
            <div className="mb-3">
              <h5 className="text-sm font-medium text-gray-700 mb-1">Policy Overview</h5>
              <p className="text-xs text-gray-600 leading-relaxed">
                {currentInsight.policy_details}
              </p>
            </div>

            {/* Expandable Sections */}
            <div className="space-y-2">
              {/* Excellent Points */}
              {currentInsight.excellent_points && currentInsight.excellent_points.length > 0 && (
                <div>
                  <button
                    onClick={() => toggleExpanded('excellent')}
                    className="flex items-center justify-between w-full text-left text-xs font-medium text-green-700 hover:text-green-800"
                  >
                    <span>✅ Excellent Points ({currentInsight.excellent_points.length})</span>
                    <span className="text-lg">{expandedPolicy === 'excellent' ? '−' : '+'}</span>
                  </button>
                  {expandedPolicy === 'excellent' && (
                    <div className="mt-2 pl-4 border-l-2 border-green-200">
                      <ul className="text-xs text-gray-600 space-y-1">
                        {currentInsight.excellent_points.map((point, index) => (
                          <li key={index} className="list-disc list-inside">{point}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {/* Major Gaps */}
              {currentInsight.major_gaps && currentInsight.major_gaps.length > 0 && (
                <div>
                  <button
                    onClick={() => toggleExpanded('gaps')}
                    className="flex items-center justify-between w-full text-left text-xs font-medium text-red-700 hover:text-red-800"
                  >
                    <span>⚠️ Major Gaps ({currentInsight.major_gaps.length})</span>
                    <span className="text-lg">{expandedPolicy === 'gaps' ? '−' : '+'}</span>
                  </button>
                  {expandedPolicy === 'gaps' && (
                    <div className="mt-2 pl-4 border-l-2 border-red-200">
                      <ul className="text-xs text-gray-600 space-y-1">
                        {currentInsight.major_gaps.map((gap, index) => (
                          <li key={index} className="list-disc list-inside">{gap}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Navigation Dots */}
        {insights.length > 1 && (
          <div className="flex justify-center gap-1 mt-3">
            {insights.map((_, index) => (
              <button
                key={index}
                onClick={() => {
                  setCurrentIndex(index);
                  setExpandedPolicy(null);
                }}
                className={`w-2 h-2 rounded-full transition-colors ${
                  index === currentIndex ? 'bg-blue-500' : 'bg-gray-300'
                }`}
              />
            ))}
          </div>
        )}
      </div>
    </Card>
  );
};

export default PolicyAnalysisResults; 
