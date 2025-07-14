import React, { useState } from 'react';
import Card from './Card';
import Input from './Input';
import Button from './Button';
import CountryDropdown from './CountryDropdown';
import { useCountryContext } from '../contexts/CountryContext';

const DOMAINS = [
  "Healthcare",
  "Entertainment",
  "Business",
  "Beauty",
  "Education",
  "Finance",
  "Transportation",
  "Retail",
  "Government",
  "Legal"
];

const ExplorePolicyWidget: React.FC<{ onDomainChange?: (domains: string[]) => void }> = ({ onDomainChange }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDomains, setSelectedDomains] = useState<string[]>([]);
  const { selectedCountries, setSelectedCountries, hasCountries } = useCountryContext();

  const handleCompare = () => {
    if (!hasCountries) {
      alert('Please select at least one country to compare');
      return;
    }
    // Optionally notify parent of selected domains
    if (onDomainChange) onDomainChange(selectedDomains);
    // TODO: Implement the comparison logic here
  };


  const handleDomainChange = (domain: string) => {
    setSelectedDomains(prev =>
      prev.includes(domain)
        ? prev.filter(d => d !== domain)
        : [...prev, domain]
    );
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
        {/* Domain Dropdown */}
        <div className="custom-border rounded-lg p-1 text-xs w-full bg-transparent flex-1 mt-1">
          <div className="font-medium mb-1">Select Domains</div>
          <div className="flex flex-wrap gap-2">
            {DOMAINS.map(domain => (
              <label key={domain} className="flex items-center gap-1 cursor-pointer">
                <input
                  type="checkbox"
                  checked={selectedDomains.includes(domain)}
                  onChange={() => handleDomainChange(domain)}
                  className="rounded"
                />
                <span className="text-xs">{domain}</span>
              </label>
            ))}
          </div>
        </div>
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
