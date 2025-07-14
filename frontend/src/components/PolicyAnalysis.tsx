import React, { useState } from 'react';
import UploadProjectWidget from './UploadProjectWidget';
import ExplorePolicyWidget from './ExplorePolicyWidget';
import ComplianceAnalysisWidget from './ComplianceAnalysisWidget';
import OECDScoreWidget from './OECDScoreWidget';
import NISTAILifestyleWidget from './NISTAILifestyleWidget';
import ExcellenciesMajorGapsWidget from './ExcellenciesMajorGapsWidget';
import ChatWithMeWidget from './ChatWithMeWidget';

const PolicyAnalysis: React.FC = () => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  return (
    <div className="flex-1 min-h-0 h-full">
      <div className="grid grid-cols-3 gap-4 h-full">
        <UploadProjectWidget onFileUpload={setUploadedFile} />
        <ExplorePolicyWidget />
        <ComplianceAnalysisWidget uploadedFile={uploadedFile} />
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
