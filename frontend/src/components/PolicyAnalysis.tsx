import React, { useState } from 'react';
import UploadProjectWidget from './UploadProjectWidget';
import ExplorePolicyWidget from './ExplorePolicyWidget';
import ComplianceAnalysisWidget from './ComplianceAnalysisWidget';
import OECDScoreWidget from './OECDScoreWidget';
import NISTAILifestyleWidget from './NISTAILifestyleWidget';
import ExcellenciesMajorGapsWidget from './ExcellenciesMajorGapsWidget';
import ChatWithMeWidget from './ChatWithMeWidget';

const PolicyAnalysis: React.FC = () => {
  const [analysisResults, setAnalysisResults] = useState<any>(null);

  const handleAnalysisComplete = (results: any) => {
    setAnalysisResults(results);
  };

  return (
    <div className="flex-1 min-h-0 h-full">
      <div className="grid grid-cols-3 gap-4 h-full">
        <UploadProjectWidget onAnalysisComplete={handleAnalysisComplete} />
        <ExplorePolicyWidget />
        <ComplianceAnalysisWidget />
        <OECDScoreWidget />
        <NISTAILifestyleWidget />
        <div className="w-full row-span-3">
          <ChatWithMeWidget />
        </div>
        <div className="col-span-2 w-full row-span-3">
          <ExcellenciesMajorGapsWidget />
        </div>
      </div>
    </div>
  );
};

export default PolicyAnalysis; 
