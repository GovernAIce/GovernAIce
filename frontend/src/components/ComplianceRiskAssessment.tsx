import React, { useState } from 'react';
import UploadProjectWidget from './UploadProjectWidget';
import ExplorePolicyWidget from './ExplorePolicyWidget';
import OECDScoreWidget from './OECDScoreWidget';
import NISTAILifestyleWidget from './NISTAILifestyleWidget';
import EURiskLevelFrameworkWidget from './EURiskLevelFrameworkWidget';
import ChatWithMeWidget from './ChatWithMeWidget';
import RelevantPoliciesWidget from './RelevantPoliciesWidget';
import ComplianceResults from './PolicyAnalysisResults';

const ComplianceRiskAssessment: React.FC = () => {
    const [analysisResults, setAnalysisResults] = useState<any>(null);
  
    const handleAnalysisComplete = (results: any) => {
      setAnalysisResults(results);
    };
  
    return (
      <div className="flex-1 min-h-0 h-full">
        <div className="grid grid-cols-3 gap-4 gap-y-1 h-full">
          <UploadProjectWidget onAnalysisComplete={handleAnalysisComplete} />
          <ExplorePolicyWidget />
          <RelevantPoliciesWidget analysisResults={analysisResults} />
          <OECDScoreWidget />
          <NISTAILifestyleWidget />
          <div className="w-full row-span-3">
            <ChatWithMeWidget />
          </div>
          <div className="col-span-2 w-full row-span-2 py-2">
            <EURiskLevelFrameworkWidget />
          </div>
        </div>
    
      </div>
  );
};

export default ComplianceRiskAssessment; 
