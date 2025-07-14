import React from 'react';
import Card from './Card';

const ExcellenciesMajorGapsWidget: React.FC = () => (
  <Card className="relative p-6 border-2 border-[#1975d4] rounded-xl">
    <img
      src="/icons/info.svg"
      alt="Info"
      className="absolute top-3 right-3 w-4 h-4 cursor-pointer"
    />
    <div className="flex flex-row gap-10">
      <div className="flex-1">
        <h3 className="text-[#1975d4] text-xl font-bold mb-2">Excellencies</h3>
        <p className="text-black text-base leading-relaxed">
          Identify and list which parts are well-written in the use case,<br />
          compared to the selected policies,<br />
          such as measures to risks.
        </p>
      </div>
      <div className="flex-1">
        <h3 className="text-[#1975d4] text-xl font-bold mb-2">Major Gaps</h3>
        <p className="text-black text-base leading-relaxed">
          Identify and list what are missed in the use case,<br />
          compared to the selected policies.
        </p>
      </div>
    </div>
  </Card>
);

export default ExcellenciesMajorGapsWidget; 
