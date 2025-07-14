import React, { useState } from 'react';
import Card from './Card';
import Button from './Button';

interface PolicyInsight {
  policy: string;
  compliance_score: number;
  policy_details: string;
  excellent_points: string[];
  major_gaps: string[];
}

interface RegulatoryInsight {
  score: number;
  details: string;
  comparison: string;
}

interface RelevantPoliciesWidgetProps {
  analysisResults?: {
    insights: PolicyInsight[] | {
      OECD: RegulatoryInsight;
      NIST_AI: RegulatoryInsight;
      EU_AI: RegulatoryInsight;
    };
    analysisType?: 'general' | 'regulatory';
  };
}

const RelevantPoliciesWidget: React.FC<RelevantPoliciesWidgetProps> = ({ analysisResults }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [expandedSection, setExpandedSection] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'compliance' | 'risk' | 'strategy'>('compliance');

  const isRegulatory = analysisResults?.analysisType === 'regulatory';
  const insights = analysisResults?.insights;

  const handlePrevious = () => {
    if (isRegulatory) {
      setCurrentIndex(prev => prev > 0 ? prev - 1 : 2);
    } else {
      const policyInsights = insights as PolicyInsight[];
      setCurrentIndex(prev => prev > 0 ? prev - 1 : policyInsights.length - 1);
    }
    setExpandedSection(null);
  };

  const handleNext = () => {
    if (isRegulatory) {
      setCurrentIndex(prev => prev < 2 ? prev + 1 : 0);
    } else {
      const policyInsights = insights as PolicyInsight[];
      setCurrentIndex(prev => prev < policyInsights.length - 1 ? prev + 1 : 0);
    }
    setExpandedSection(null);
  };

  const toggleExpanded = (section: string) => {
    setExpandedSection(expandedSection === section ? null : section);
  };

  const getCurrentInsight = () => {
    if (!insights) return null;

    if (isRegulatory) {
      const regulatoryInsights = insights as {
        OECD: RegulatoryInsight;
        NIST_AI: RegulatoryInsight;
        EU_AI: RegulatoryInsight;
      };
      const frameworks = ['OECD', 'NIST_AI', 'EU_AI'];
      const currentFramework = frameworks[currentIndex];
      return {
        name: currentFramework === 'OECD' ? 'OECD AI Principles' :
              currentFramework === 'NIST_AI' ? 'NIST AI Risk Management Framework' :
              'EU AI Act',
        insight: regulatoryInsights[currentFramework as keyof typeof regulatoryInsights],
        type: 'regulatory' as const
      };
    } else {
      const policyInsights = insights as PolicyInsight[];
      return {
        name: policyInsights[currentIndex]?.policy || '',
        insight: policyInsights[currentIndex],
        type: 'policy' as const
      };
    }
  };

  const currentInsight = getCurrentInsight();

  const getRiskLevel = (score: number) => {
    if (score >= 80) return { level: 'Low', color: 'text-green-600', bgColor: 'bg-green-100' };
    if (score >= 60) return { level: 'Medium', color: 'text-yellow-600', bgColor: 'bg-yellow-100' };
    return { level: 'High', color: 'text-red-600', bgColor: 'bg-red-100' };
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreBarColor = (score: number) => {
    if (score >= 80) return 'bg-green-500';
    if (score >= 60) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  // Get the score based on insight type
  const getScore = () => {
    if (!currentInsight) return 0;
    if (currentInsight.type === 'policy') {
      return (currentInsight.insight as PolicyInsight).compliance_score;
    } else {
      return (currentInsight.insight as RegulatoryInsight).score;
    }
  };

  // Get the details based on insight type
  const getDetails = () => {
    if (!currentInsight) return '';
    if (currentInsight.type === 'policy') {
      return (currentInsight.insight as PolicyInsight).policy_details;
    } else {
      return (currentInsight.insight as RegulatoryInsight).details;
    }
  };

  const score = getScore();
  const details = getDetails();
  const riskLevel = getRiskLevel(score);

  if (!analysisResults || !currentInsight) {
    return (
      <Card className="custom-border p-4 h-auto min-h-[200px]">
        <img
          src="/icons/info.svg"
          alt="Info"
          className="absolute top-2 right-2 w-4 h-4 cursor-pointer"
        />
        <div className="flex flex-col h-full gap-5">
          <div className="flex justify-between items-center">
            <h3 className="text-xl text-[#1975d4] font-bold">Compliance Analysis</h3>
          </div>
          <p className="text-sm text-black">
            Upload a document to see compliance comparison and risk assessment...
          </p>
          <div className="flex justify-center items-center gap-4 mt-auto">
            <button disabled>
              <img
                src="/icons/Arrow11.svg"
                alt="Left Arrow"
                style={{ width: '24px', height: '24px', cursor: 'pointer', opacity: 0.5 }}
              />
            </button>
            <Button className="w-72 text-white text-sm" disabled>View Full Analysis</Button>
            <button disabled>
              <img
                src="/icons/Arrow10.svg"
                alt="Right Arrow"
                style={{ width: '24px', height: '24px', cursor: 'pointer', opacity: 0.5 }}
              />
            </button>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card className="custom-border p-4 h-auto min-h-[200px]">
      <img
        src="/icons/info.svg"
        alt="Info"
        className="absolute top-2 right-2 w-4 h-4 cursor-pointer"
      />
      <div className="flex flex-col h-full gap-3">
        {/* Header with View Mode Tabs */}
        <div className="flex justify-between items-center flex-shrink-0">
          <h3 className="text-xl text-[#1975d4] font-bold">
            {isRegulatory ? 'Regulatory Analysis' : 'Compliance Analysis'}
          </h3>
          <div className="flex gap-1">
            <button
              onClick={() => setViewMode('compliance')}
              className={`px-2 py-1 text-xs rounded ${
                viewMode === 'compliance' 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-200 text-gray-600'
              }`}
            >
              Compliance
            </button>
            <button
              onClick={() => setViewMode('risk')}
              className={`px-2 py-1 text-xs rounded ${
                viewMode === 'risk' 
                  ? 'bg-red-500 text-white' 
                  : 'bg-gray-200 text-gray-600'
              }`}
            >
              Risk
            </button>
            <button
              onClick={() => setViewMode('strategy')}
              className={`px-2 py-1 text-xs rounded ${
                viewMode === 'strategy' 
                  ? 'bg-green-500 text-white' 
                  : 'bg-gray-200 text-gray-600'
              }`}
            >
              Strategy
            </button>
          </div>
        </div>

        {/* Content Area - Scrollable */}
        <div className="flex-1 bg-gray-50 rounded-lg p-3 overflow-y-auto min-h-0">
          <h4 className="font-semibold text-gray-800 mb-2 text-sm">{currentInsight.name}</h4>
          
          {viewMode === 'compliance' && (
            <>
              {/* Compliance Score */}
              <div className="flex items-center gap-2 mb-3">
                <span className="text-xs text-gray-600">Compliance Score:</span>
                <div className="flex items-center gap-1">
                  <div className="w-12 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                    <div 
                      className={`h-full rounded-full transition-all duration-300 ${getScoreBarColor(score)}`}
                      style={{ width: `${score}%` }}
                    ></div>
                  </div>
                  <span className={`text-xs font-medium ${getScoreColor(score)}`}>
                    {score}%
                  </span>
                </div>
              </div>

              {/* Policy Details */}
              <div className="mb-3">
                <p className="text-xs text-gray-600 leading-relaxed">
                  {details}
                </p>
              </div>

              {/* Expandable Sections for Policy Analysis */}
              {currentInsight.type === 'policy' && (
                <div className="space-y-1">
                  {/* Excellent Points */}
                  {(currentInsight.insight as PolicyInsight).excellent_points && (currentInsight.insight as PolicyInsight).excellent_points.length > 0 && (
                    <div>
                      <button
                        onClick={() => toggleExpanded('excellent')}
                        className="flex items-center justify-between w-full text-left text-xs font-medium text-green-700 hover:text-green-800"
                      >
                        <span>✅ Excellent Points ({(currentInsight.insight as PolicyInsight).excellent_points.length})</span>
                        <span className="text-sm">{expandedSection === 'excellent' ? '−' : '+'}</span>
                      </button>
                      {expandedSection === 'excellent' && (
                        <div className="mt-1 pl-3 border-l-2 border-green-200">
                          <ul className="text-xs text-gray-600 space-y-0.5">
                            {(currentInsight.insight as PolicyInsight).excellent_points.map((point, index) => (
                              <li key={index} className="list-disc list-inside">{point}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Major Gaps */}
                  {(currentInsight.insight as PolicyInsight).major_gaps && (currentInsight.insight as PolicyInsight).major_gaps.length > 0 && (
                    <div>
                      <button
                        onClick={() => toggleExpanded('gaps')}
                        className="flex items-center justify-between w-full text-left text-xs font-medium text-red-700 hover:text-red-800"
                      >
                        <span>⚠️ Major Gaps ({(currentInsight.insight as PolicyInsight).major_gaps.length})</span>
                        <span className="text-sm">{expandedSection === 'gaps' ? '−' : '+'}</span>
                      </button>
                      {expandedSection === 'gaps' && (
                        <div className="mt-1 pl-3 border-l-2 border-red-200">
                          <ul className="text-xs text-gray-600 space-y-0.5">
                            {(currentInsight.insight as PolicyInsight).major_gaps.map((gap, index) => (
                              <li key={index} className="list-disc list-inside">{gap}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Comparison for Regulatory Analysis */}
              {currentInsight.type === 'regulatory' && (currentInsight.insight as RegulatoryInsight).comparison && (
                <div>
                  <button
                    onClick={() => toggleExpanded('comparison')}
                    className="flex items-center justify-between w-full text-left text-xs font-medium text-gray-700 hover:text-gray-800"
                  >
                    <span>📊 Comparative Analysis</span>
                    <span className="text-sm">{expandedSection === 'comparison' ? '−' : '+'}</span>
                  </button>
                  {expandedSection === 'comparison' && (
                    <div className="mt-1 pl-3 border-l-2 border-gray-200">
                      <p className="text-xs text-gray-600 leading-relaxed">
                        {(currentInsight.insight as RegulatoryInsight).comparison}
                      </p>
                    </div>
                  )}
                </div>
              )}
            </>
          )}

          {viewMode === 'risk' && (
            <div className="space-y-3">
              {/* Risk Level Indicator */}
              <div className={`p-2 rounded ${riskLevel.bgColor}`}>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium">Risk Level:</span>
                  <span className={`text-xs font-bold ${riskLevel.color}`}>
                    {riskLevel.level} Risk
                  </span>
                </div>
              </div>

              {/* Risk Assessment Details */}
              <div>
                <h5 className="text-xs font-medium text-gray-700 mb-1">Risk Assessment</h5>
                <p className="text-xs text-gray-600 leading-relaxed">
                  {score < 60 
                    ? "High compliance risk detected. Immediate attention required for regulatory gaps."
                    : score < 80 
                    ? "Medium compliance risk. Some areas need improvement to meet full requirements."
                    : "Low compliance risk. Strong alignment with regulatory requirements."
                  }
                </p>
              </div>

              {/* Risk Heat Map Placeholder */}
              <div className="bg-white rounded p-2 border">
                <h6 className="text-xs font-medium text-gray-700 mb-1">Regional Risk Heat Map</h6>
                <div className="grid grid-cols-3 gap-1 text-xs">
                  <div className="text-center p-1 bg-red-100 rounded">USA<br/>High</div>
                  <div className="text-center p-1 bg-yellow-100 rounded">UK<br/>Medium</div>
                  <div className="text-center p-1 bg-green-100 rounded">EU<br/>Low</div>
                </div>
              </div>
            </div>
          )}

          {viewMode === 'strategy' && (
            <div className="space-y-3">
              {/* Strategic Recommendations */}
              <div>
                <h5 className="text-xs font-medium text-gray-700 mb-1">Strategic Recommendations</h5>
                <div className="space-y-2">
                  {score < 60 ? (
                    <>
                      <div className="p-2 bg-red-50 rounded border-l-2 border-red-400">
                        <p className="text-xs text-red-700 font-medium">Immediate Actions Required</p>
                        <p className="text-xs text-red-600">Address compliance gaps before market entry</p>
                      </div>
                      <div className="p-2 bg-yellow-50 rounded border-l-2 border-yellow-400">
                        <p className="text-xs text-yellow-700 font-medium">Consider Alternative Markets</p>
                        <p className="text-xs text-yellow-600">Evaluate lower-risk regions for initial launch</p>
                      </div>
                    </>
                  ) : score < 80 ? (
                    <>
                      <div className="p-2 bg-yellow-50 rounded border-l-2 border-yellow-400">
                        <p className="text-xs text-yellow-700 font-medium">Moderate Risk Strategy</p>
                        <p className="text-xs text-yellow-600">Implement improvements before full deployment</p>
                      </div>
                      <div className="p-2 bg-blue-50 rounded border-l-2 border-blue-400">
                        <p className="text-xs text-blue-700 font-medium">Phased Rollout</p>
                        <p className="text-xs text-blue-600">Consider gradual market entry approach</p>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="p-2 bg-green-50 rounded border-l-2 border-green-400">
                        <p className="text-xs text-green-700 font-medium">Market Ready</p>
                        <p className="text-xs text-green-600">Proceed with confidence in target markets</p>
                      </div>
                      <div className="p-2 bg-blue-50 rounded border-l-2 border-blue-400">
                        <p className="text-xs text-blue-700 font-medium">Expand Opportunities</p>
                        <p className="text-xs text-blue-600">Consider additional regional markets</p>
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Business Impact */}
              <div>
                <h6 className="text-xs font-medium text-gray-700 mb-1">Business Impact</h6>
                <p className="text-xs text-gray-600">
                  {score < 60 
                    ? "High regulatory risk may delay market entry and increase compliance costs."
                    : score < 80 
                    ? "Moderate risk allows for controlled market entry with manageable compliance costs."
                    : "Low risk enables rapid market entry with minimal compliance overhead."
                  }
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Navigation - Fixed at bottom */}
        <div className="flex flex-col gap-2 flex-shrink-0">
          <div className="flex justify-center items-center gap-4">
            <button
              onClick={handlePrevious}
              className="p-1 rounded-full hover:bg-gray-100 transition-colors"
            >
              <img
                src="/icons/Arrow11.svg"
                alt="Left Arrow"
                style={{ width: '24px', height: '24px', cursor: 'pointer' }}
              />
            </button>
            <Button className="w-72 text-white text-sm">
              {viewMode === 'compliance' ? 'View Full Analysis' : 
               viewMode === 'risk' ? 'View Risk Report' : 'View Strategy Report'}
            </Button>
            <button
              onClick={handleNext}
              className="p-1 rounded-full hover:bg-gray-100 transition-colors"
            >
              <img
                src="/icons/Arrow10.svg"
                alt="Right Arrow"
                style={{ width: '24px', height: '24px', cursor: 'pointer' }}
              />
            </button>
          </div>

          {/* Minimalistic Position Indicator */}
          <div className="flex justify-center">
            <div className="bg-gray-100 rounded-full px-3 py-1">
              <span className="text-xs text-gray-600 font-medium">
                {currentIndex + 1} of {isRegulatory ? 3 : (insights as PolicyInsight[]).length}
              </span>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default RelevantPoliciesWidget; 
