import React, { useState } from 'react';
import UploadProjectWidget from './UploadProjectWidget';
import ExplorePolicyWidget from './ExplorePolicyWidget';
import RelevantPoliciesWidget from './RelevantPoliciesWidget';
import OECDScoreWidget from './OECDScoreWidget';
import NISTAILifestyleWidget from './NISTAILifestyleWidget';
import EURiskLevelFrameworkWidget from './EURiskLevelFrameworkWidget';
import ChatWithMeWidget from './ChatWithMeWidget';

const ComplianceRiskAssessment:  React.FC = () => {
    const [domain, setDomain] = useState<string | undefined>(undefined);
    const [searchQuery, setSearchQuery] = useState<string | undefined>(undefined);

  return (
    <div className="flex-1 min-h-0 h-full">
      <div className="grid grid-cols-3 gap-4 h-full">
        <UploadProjectWidget />
        <ExplorePolicyWidget />
        <RelevantPoliciesWidget />
        <OECDScoreWidget />
        <NISTAILifestyleWidget />
        <div className="w-full row-span-3">
          <ChatWithMeWidget />
        </div>
        <div className="col-span-2 w-full row-span-3">
          <EURiskLevelFrameworkWidget />
        </div>
      </div>
    </div>
  );
};

export default ComplianceRiskAssessment; 
