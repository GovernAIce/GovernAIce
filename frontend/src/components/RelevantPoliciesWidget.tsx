import React from 'react';
import Card from './Card';
import Button from './Button';

const RelevantPoliciesWidget: React.FC = () => (
  <Card className="custom-border relative p-3 h-[130px]">
    <img
      src="/icons/info.svg"
      alt="Info"
      className="absolute top-2 right-2 w-4 h-4 cursor-pointer"
    />
    <div className="flex flex-col h-full gap-5">
      <div className="flex justify-between items-center">
        <h3 className="text-xl text-[#1975d4] font-bold">Relevant Policies & Regulators</h3>
      </div>
      <p className="text-sm text-black">
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua...
      </p>
      <div className="flex justify-center items-center gap-4 mt-auto">
        <button>
          <img
            src="/icons/arrow11.svg"
            alt="Left Arrow"
            style={{ width: '24px', height: '24px', cursor: 'pointer' }}
          />
        </button>
        <Button className="w-72 text-white text-sm">View Full Policy Document</Button>
        <button>
          <img
            src="/icons/arrow11.svg"
            alt="Right Arrow"
            style={{ width: '24px', height: '24px', cursor: 'pointer', transform: 'rotate(180deg)' }}
          />
        </button>
      </div>
    </div>
  </Card>
);

export default RelevantPoliciesWidget; 
