import React from 'react';
import UploadProjectWidget from './UploadProjectWidget';
import ExplorePolicyWidget from './ExplorePolicyWidget';
import RelevantPoliciesWidget from './RelevantPoliciesWidget';
import OverallScoreWidget from './OverallScoreWidget';
import PoliciesRegulatorsWidget from './PoliciesRegulatorsWidget';
import ChatWithMeWidget from './ChatWithMeWidget';
import ExcellenciesMajorGapsWidget from './ExcellenciesMajorGapsWidget';

const Dashboard: React.FC = () => (
  <div className="flex-1 min-h-0 h-full">
    <div className="grid grid-cols-3 gap-8 h-full">
      <UploadProjectWidget />
      <ExplorePolicyWidget />
      <RelevantPoliciesWidget />
      <OverallScoreWidget />
      <PoliciesRegulatorsWidget />
      <ChatWithMeWidget className="row-span-2" />
      <div className="col-span-2 w-full">
        <ExcellenciesMajorGapsWidget />
      </div>
    </div>
  </div>
);

export default Dashboard; 
