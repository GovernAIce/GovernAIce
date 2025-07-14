import React, { useState } from 'react';
import Card from './Card';

interface RegulatoryInsight {
  score: number;
  details: string;
  comparison: string;
}

interface RegulatoryAnalysisResultsProps {
  insights: {
    OECD: RegulatoryInsight;
    NIST_AI: RegulatoryInsight;
    EU_AI: RegulatoryInsight;
  };
  onClose?: () => void;
}

const RegulatoryAnalysisResults: React.FC<RegulatoryAnalysisResultsProps> = ({ insights, onClose }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [expandedSection, setExpandedSection] = useState<string | null>(null);

  const frameworks = [
    { key: 'OECD', name: 'OECD AI Principles', color: 'blue' },
    { key: 'NIST_AI', name: 'NIST AI Risk Management Framework', color: 'green' },
    { key: 'EU_AI', name: 'EU AI Act', color: 'purple' }
  ];

  const handlePrevious = () => {
    setCurrentIndex(prev => prev > 0 ? prev - 1 : frameworks.length - 1);
    setExpandedSection(null);
  };

  const handleNext = () => {
    setCurrentIndex(prev => prev < frameworks.length - 1 ? prev + 1 : 0);
    setExpandedSection(null);
  };

  const toggleExpanded = (section: string) => {
    setExpandedSection(expandedSection === section ? null : section);
  };

  const currentFramework = frameworks[currentIndex];
  const currentInsight = insights[currentFramework.key as keyof typeof insights];

  const getColorClasses = (color: string) => {
    switch (color) {
      case 'blue': return 'bg-blue-500 border-blue-200 text-blue-700';
      case 'green': return 'bg-green-500 border-green-200 text-green-700';
      case 'purple': return 'bg-purple-500 border-purple-200 text-purple-700';
      default: return 'bg-blue-500 border-blue-200 text-blue-700';
    }
  };

  return (
    <Card className="custom-border p-4 h-auto min-h-[300px]">
      <div className="flex flex-col h-full">
        {/* Header with navigation */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg text-[#1975d4] font-bold">Regulatory Analysis Results</h3>
          <div className="flex items-center gap-2">
            <button
              onClick={handlePrevious}
              className="p-1 rounded-full hover:bg-gray-100 transition-colors"
              disabled={frameworks.length <= 1}
            >
              <img src="/icons/Arrow10.svg" alt="Previous" className="w-4 h-4 rotate-180" />
            </button>
            <span className="text-xs text-gray-600">
              {currentIndex + 1} of {frameworks.length}
            </span>
            <button
              onClick={handleNext}
              className="p-1 rounded-full hover:bg-gray-100 transition-colors"
              disabled={frameworks.length <= 1}
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

        {/* Framework Card */}
        <div className="flex-1 bg-white rounded-lg border border-gray-200 p-4">
          <div className="mb-3">
            <h4 className="font-semibold text-gray-800 mb-2">{currentFramework.name}</h4>
            
            {/* Compliance Score */}
            <div className="flex items-center gap-2 mb-3">
              <span className="text-sm text-gray-600">Compliance Score:</span>
              <div className="flex items-center gap-1">
                <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    className={`h-full rounded-full transition-all duration-300 ${getColorClasses(currentFramework.color)}`}
                    style={{ width: `${currentInsight.score}%` }}
                  ></div>
                </div>
                <span className="text-sm font-medium">
                  {currentInsight.score}%
                </span>
              </div>
            </div>

            {/* Framework Details */}
            <div className="mb-3">
              <h5 className="text-sm font-medium text-gray-700 mb-1">Framework Overview</h5>
              <p className="text-xs text-gray-600 leading-relaxed">
                {currentInsight.details}
              </p>
            </div>

            {/* Expandable Sections */}
            <div className="space-y-2">
              {/* Comparison */}
              {currentInsight.comparison && (
                <div>
                  <button
                    onClick={() => toggleExpanded('comparison')}
                    className="flex items-center justify-between w-full text-left text-xs font-medium text-gray-700 hover:text-gray-800"
                  >
                    <span>📊 Comparative Analysis</span>
                    <span className="text-lg">{expandedSection === 'comparison' ? '−' : '+'}</span>
                  </button>
                  {expandedSection === 'comparison' && (
                    <div className="mt-2 pl-4 border-l-2 border-gray-200">
                      <p className="text-xs text-gray-600 leading-relaxed">
                        {currentInsight.comparison}
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Navigation Dots */}
        {frameworks.length > 1 && (
          <div className="flex justify-center gap-1 mt-3">
            {frameworks.map((_, index) => (
              <button
                key={index}
                onClick={() => {
                  setCurrentIndex(index);
                  setExpandedSection(null);
                }}
                className={`w-2 h-2 rounded-full transition-colors ${
                  index === currentIndex ? 'bg-blue-500' : 'bg-gray-300'
                }`}
              />
            ))}
          </div>
        )}

        {/* Summary Scores */}
        <div className="mt-3 p-2 bg-gray-50 rounded">
          <h5 className="text-xs font-medium text-gray-700 mb-2">Summary Scores</h5>
          <div className="flex justify-between text-xs">
            {frameworks.map((framework) => (
              <div key={framework.key} className="text-center">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white text-xs font-bold ${getColorClasses(framework.color)}`}>
                  {insights[framework.key as keyof typeof insights].score}
                </div>
                <p className="text-gray-600 mt-1">{framework.name.split(' ')[0]}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Card>
  );
};

export default RegulatoryAnalysisResults; 
