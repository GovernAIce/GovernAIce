import React, { useState } from 'react';
import UploadProjectWidget from './UploadProjectWidget';
import ExplorePolicyWidget from './ExplorePolicyWidget';
import OECDScoreWidget from './OECDScoreWidget';
import NISTAILifestyleWidget from './NISTAILifestyleWidget';
import ExcellenciesMajorGapsWidget from './ExcellenciesMajorGapsWidget';
import ChatWithMeWidget from './ChatWithMeWidget';
import ComplianceAnalysisWidget from './ComplianceAnalysisWidget';

const PolicyAnalysis: React.FC = () => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [insights, setInsights] = useState<any[]>([]);

  const handleAnalysisComplete = (results: any) => {
    setUploadedFile(results.uploadedFile || null);
    setInsights(results.insights || []);
  };

  return (
    <div className="flex-1 min-h-0 h-full">
      <div className="grid grid-cols-3 gap-4 h-full">
        <UploadProjectWidget onFileUpload={setUploadedFile} onAnalysisComplete={handleAnalysisComplete} />
        <ExplorePolicyWidget />
        <ComplianceAnalysisWidget
          uploadedFile={uploadedFile}
          policies={insights.length > 0 ? insights.map(i => ({ ...i, title: i.policy })) : []}
        />        
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
