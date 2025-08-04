import React, { useState } from 'react';
import Card from './Card';
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

// DomainDropdown Component
const DomainDropdown: React.FC<{
  value: string[];
  onChange: (value: string[]) => void;
  className?: string;
}> = ({ value, onChange, className = "" }) => {
  const handleDomainChange = (domain: string) => {
    const newValues = value.includes(domain)
      ? value.filter(d => d !== domain)
      : [...value, domain];
    onChange(newValues);
  };

  return (
    <div className={className + " max-h-32 overflow-y-auto bg-white rounded-lg p-2 border-2 border-blue-400"}>
      <div className="flex flex-wrap gap-2">
        {DOMAINS.map(domain => (
          <label key={domain} className="flex items-center gap-1 cursor-pointer">
            <input
              type="checkbox"
              checked={value.includes(domain)}
              onChange={() => handleDomainChange(domain)}
              className="rounded"
            />
            <span className="text-xs">{domain}</span>
          </label>
        ))}
      </div>
    </div>
  );
};

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

  const clearSearch = () => {
    setSearchQuery('');
  };

  return (
    <Card className="bg-white rounded-2xl shadow-lg p-5 h-full custom-border p-4 h-full">
      <div className="flex flex-col h-full gap-4">
        <h3 className="text-lg text-[#1975d4] font-bold">Explore Policy</h3>
        
        {/* Search Input */}
        <div className="relative">
          <input
            type="text"
            placeholder="Search"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            className="w-full px-3 py-2 border border-blue-300 rounded-lg text-sm focus:outline-none focus:border-blue-500"
          />
          {searchQuery && (
            <button
              onClick={clearSearch}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
            >
              ×
            </button>
          )}
        </div>

        {/* Country and Domain Dropdowns */}
        <div className="flex gap-3">
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-700 mb-1">Country</label>
            <CountryDropdown
              value={selectedCountries}
              onChange={setSelectedCountries}
              multiple={true}
              placeholder="Select Countries"
              className="w-full"
            />
          </div>

          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-700 mb-1">Domain</label>
            <DomainDropdown
              value={selectedDomains}
              onChange={setSelectedDomains}
              className="w-full"
            />
          </div>
        </div>

        {/* Compare Button */}
        {/* } <button
          onClick={handleCompare}
          disabled={!hasCountries}
          className="w-full bg-gradient-to-r from-[#9FD8FF] to-[#2196f3] text-white py-3 px-4 rounded-[50px] font-medium text-sm opacity-100 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
        >
          Compare
        </button>
        */}
      </div>
    </Card>
  );
};

export default ExplorePolicyWidget;
