import React from 'react';
import Card from './Card';

const UploadProjectWidget: React.FC = () => (
  <Card className="custom-border relative p-3 h-[130px]">
    <img
      src="/icons/info.svg"
      alt="Info"
      className="absolute top-2 right-2 w-4 h-4 cursor-pointer"
    />
    <div className="flex flex-col items-center gap-2.5">
      <div className="w-28 h-32 custom-bg rounded-lg flex items-center justify-center">
        <img
          src="/icons/Upload.svg"
          alt="Upload Icon"
        />
      </div>
      <h3 className="text-xl text-[#1975d4] font-bold">Upload Project</h3>
      <p className="text-sm text-black text-center">
        Please upload your product illustration: doc, pdf, jpg...
      </p>
    </div>
  </Card>
);

export default UploadProjectWidget; 
