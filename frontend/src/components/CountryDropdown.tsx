import React, { useEffect, useState } from "react";
import { fetchCountries } from "../api/metadata";

interface CountryDropdownProps {
  value: string | string[];
  onChange: (value: string | string[]) => void;
  className?: string;
  multiple?: boolean;
  placeholder?: string;
}

const CountryDropdown: React.FC<CountryDropdownProps> = ({ 
  value, 
  onChange, 
  className = "",
  multiple = false,
  placeholder = "Select a country"
}) => {
  const [countries, setCountries] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    fetchCountries()
      .then((res: any) => {
        if (res.data && res.data.countries) {
          setCountries((res.data.countries || []).sort((a: string, b: string) => a.localeCompare(b)));
          setError(null);
        } else {
          setError("Invalid response format");
        }
      })
      .catch((err: any) => {
        setError(`Failed to load countries: ${err.message || 'Unknown error'}`);
      })
      .finally(() => setLoading(false));
  }, []);

  const handleMultipleChange = (country: string) => {
    const currentValues = Array.isArray(value) ? value : [];
    const newValues = currentValues.includes(country)
      ? currentValues.filter(c => c !== country)
      : [...currentValues, country];
    onChange(newValues);
  };

  if (loading) {
    return (
      <div className={className}>
        <div className="p-2 text-gray-500">Loading countries...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={className}>
        <div className="p-2 text-red-500">{error}</div>
      </div>
    );
  }

  // Always show as multi-checkbox list
  const selectedCountries = Array.isArray(value) ? value : [];
  return (
    <div className={className + " max-h-32 overflow-y-auto bg-white rounded-lg p-2 border-2 border-blue-400"}>
      <div className="flex flex-wrap gap-2">
        {countries.map(country => (
          <label key={country} className="flex items-center gap-1 cursor-pointer">
            <input
              type="checkbox"
              checked={selectedCountries.includes(country)}
              onChange={() => handleMultipleChange(country)}
              className="rounded"
            />
            <span className="text-xs">{country}</span>
          </label>
        ))}
      </div>
    </div>
  );
};

export default CountryDropdown; 
