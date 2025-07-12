import React from 'react';
import UploadProjectWidget from './UploadProjectWidget';
import ExplorePolicyWidget from './ExplorePolicyWidget';
import RelevantPoliciesWidget from './RelevantPoliciesWidget';
import OECDScoreWidget from './OECDScoreWidget';
import NISTAILifestyleWidget from './NISTAILifestyleWidget';
import EURiskLevelFrameworkWidget from './EURiskLevelFrameworkWidget';
import ChatWithMeWidget from './ChatWithMeWidget';
import ExcellenciesMajorGapsWidget from './ExcellenciesMajorGapsWidget';

const Dashboard: React.FC = () => (
  <div className="flex-1 min-h-0 h-full">
    <div className="grid grid-cols-3 gap-8 h-full">
      <UploadProjectWidget />
      <ExplorePolicyWidget />
      <RelevantPoliciesWidget />
      <OECDScoreWidget />
      <NISTAILifestyleWidget />
      <EURiskLevelFrameworkWidget />
      <div className="col-span-2 w-full row-span-2">
        <ExcellenciesMajorGapsWidget />
      </div>
      <ChatWithMeWidget className="row-span-2" />
    </div>
  </div>
);

export default Dashboard; 
