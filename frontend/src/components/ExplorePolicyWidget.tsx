import React, { useState } from 'react';
import Card from './Card';
import Input from './Input';
import Button from './Button';
import CountryDropdown from './CountryDropdown';
import { useCountryContext } from '../contexts/CountryContext';

const ExplorePolicyWidget: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDomain, setSelectedDomain] = useState('Domain');
  const { selectedCountries, setSelectedCountries, hasCountries } = useCountryContext();

  const handleCompare = () => {
    if (!hasCountries) {
      alert('Please select at least one country to compare');
      return;
    }
    
    console.log('Comparing policies for countries:', selectedCountries);
    console.log('Search query:', searchQuery);
    console.log('Domain:', selectedDomain);
    
    // TODO: Implement the comparison logic here
    // You can make API calls to fetch policies for each selected country
    // and then display the comparison results
  };

  return (
    <Card className="custom-border relative p-2 h-auto min-h-[200px]">
      <img
        src="/icons/info.svg"
        alt="Info"
        className="absolute top-1 right-1 w-3 h-3 cursor-pointer"
      />
      
      <div className="flex flex-col h-full gap-1.5">
        <h3 className="text-lg text-[#1975d4] font-bold">Explore Policy</h3>
        
        <Input
          placeholder="Search policies..."
          className="pl-8 pr-8 text-xs text-gray-800 bg-transparent outline-none"
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
        />
        
        <CountryDropdown
          value={selectedCountries}
          onChange={(value: string | string[]) => setSelectedCountries(Array.isArray(value) ? value : [])}
          multiple={true}
          placeholder="Select countries to compare"
          className="custom-border rounded-lg p-1 text-xs w-full bg-transparent flex-1"
        />
        
        <select
          className="custom-border rounded-lg p-1 text-xs w-full appearance-none bg-transparent"
          value={selectedDomain}
          onChange={e => setSelectedDomain(e.target.value)}
        >
          <option>Domain</option>
          <option>New Domain</option>
        </select>
        
        <Button 
          className="w-full text-white text-xs mt-auto"
          onClick={handleCompare}
          disabled={!hasCountries}
        >
          {!hasCountries 
            ? 'Compare' 
            : `Compare (${selectedCountries.length})`
          }
        </Button>
      </div>
    </Card>
  );
};

export default ExplorePolicyWidget; 
