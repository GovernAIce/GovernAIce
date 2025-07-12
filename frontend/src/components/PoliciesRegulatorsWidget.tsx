import React from 'react';
import Card from './Card';

const PoliciesRegulatorsWidget: React.FC = () => (
  <Card className="custom-border relative p-4 h-full">
    <img
      src="/icons/info.svg"
      alt="Info"
      className="absolute top-2 right-2 w-4 h-4 cursor-pointer"
    />
    <div className="flex flex-col justify-center items-center h-full w-full">
      <h3 className="text-xl text-[#1975d4] font-bold">Relevant Policies & Regulators</h3>
      <p className="text-sm text-black">(pop up relevant information)</p>
      <div className="w-full flex flex-col items-center justify-center">
        <div className="h-48 flex items-end justify-center gap-2 mb-4">
          <div className="flex flex-col items-center gap-1">
            <div className="flex gap-1">
              <div className="w-4 h-8 bg-[#ff928a] rounded-t"></div>
              <div className="w-4 h-6 bg-[#ffae4c] rounded-t"></div>
              <div className="w-4 h-10 bg-[#6fd195] rounded-t"></div>
            </div>
            <span className="text-xs text-[#9ea2ae]">2020</span>
          </div>
          <div className="flex flex-col items-center gap-1">
            <div className="flex gap-1">
              <div className="w-4 h-16 bg-[#ff928a] rounded-t"></div>
              <div className="w-4 h-8 bg-[#ffae4c] rounded-t"></div>
              <div className="w-4 h-12 bg-[#6fd195] rounded-t"></div>
            </div>
            <span className="text-xs text-[#9ea2ae]">2021</span>
          </div>
          <div className="flex flex-col items-center gap-1">
            <div className="flex gap-1">
              <div className="w-4 h-20 bg-[#1975d4] rounded-t"></div>
            </div>
            <span className="text-xs text-[#9ea2ae]">2022</span>
          </div>
          <div className="flex flex-col items-center gap-1">
            <div className="flex gap-1">
              <div className="w-4 h-14 bg-[#1975d4] rounded-t"></div>
              <div className="w-4 h-16 bg-[#3cc3df] rounded-t"></div>
            </div>
            <span className="text-xs text-[#9ea2ae]">2023</span>
          </div>
          <div className="flex flex-col items-center gap-1">
            <div className="flex gap-1">
              <div className="w-4 h-18 bg-[#1975d4] rounded-t"></div>
              <div className="w-4 h-10 bg-[#6fd195] rounded-t"></div>
            </div>
            <span className="text-xs text-[#9ea2ae]">2024</span>
          </div>
        </div>
        <div className="flex justify-center gap-4 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-[#ff928a] rounded"></div>
            <span className="text-[#9ea2ae]">Figma</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-[#1975d4] rounded"></div>
            <span className="text-[#9ea2ae]">AI</span>
          </div>
        </div>
      </div>
    </div>
  </Card>
);

export default PoliciesRegulatorsWidget; 
