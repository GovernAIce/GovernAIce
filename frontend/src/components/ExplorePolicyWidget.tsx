import React, { useState } from 'react';
import Card from './Card';
import Input from './Input';
import Button from './Button';

const ExplorePolicyWidget: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCountry, setSelectedCountry] = useState('Country');
  const [selectedDomain, setSelectedDomain] = useState('Domain');

  return (
    <Card className="custom-border relative p-3 h-[130px]">
      <img
        src="/icons/info.svg"
        alt="Info"
        className="absolute top-2 right-2 w-4 h-4 cursor-pointer"
      />
      <div className="flex flex-col h-full gap-2.5">
        <h3 className="text-xl text-[#1975d4] font-bold">Explore Policy</h3>
        <div className="relative">
          {/* Replace with icon import if needed */}
        </div>
        <Input
          placeholder="Search"
          className="pl-10 pr-10 text-sm text-gray-800 bg-transparent outline-none"
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
        />
        <select
          className="custom-border rounded-lg p-2 text-sm w-full appearance-none bg-transparent"
          value={selectedCountry}
          onChange={e => setSelectedCountry(e.target.value)}
        >
          <option>Country</option>
          <option>New Country</option>
        </select>
        <select
          className="custom-border rounded-lg p-2 text-sm w-full appearance-none bg-transparent"
          value={selectedDomain}
          onChange={e => setSelectedDomain(e.target.value)}
        >
          <option>Domain</option>
          <option>New Domain</option>
        </select>
        <Button className="w-full text-white text-sm mt-auto">Compare</Button>
      </div>
    </Card>
  );
};

export default ExplorePolicyWidget; 
