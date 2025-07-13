import React, { useEffect, useState, useRef } from "react";
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
  const [open, setOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

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

  // Close dropdown on outside click
  useEffect(() => {
    if (!open) return;
    const handleClick = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

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

  if (multiple) {
    const selectedCountries = Array.isArray(value) ? value : [];
    return (
      <div className={className + " relative"} ref={dropdownRef}>
        <button
          type="button"
          className="px-3 py-2 border rounded bg-white shadow-sm text-sm text-gray-700 hover:bg-gray-50 w-full text-left focus:outline-none focus:ring-0"
          onClick={() => setOpen((v) => !v)}
        >
          {selectedCountries.length === 0
            ? "Select Countries to Compare"
            : `${selectedCountries.length} countr${selectedCountries.length === 1 ? 'y' : 'ies'} selected`}
        </button>
        {open && (
          <div className="absolute z-10 mt-2 w-64 max-h-56 overflow-y-auto bg-white border border-gray-200 rounded shadow-lg p-2" style={{ minWidth: 200 }}>
            {countries.map(country => (
              <label key={country} className="flex items-center space-x-2 p-1 hover:bg-gray-50 cursor-pointer">
                <input
                  type="checkbox"
                  checked={selectedCountries.includes(country)}
                  onChange={() => handleMultipleChange(country)}
                  className="rounded"
                />
                <span className="text-sm">{country}</span>
              </label>
            ))}
            <div className="mt-2 flex justify-end">
              <button
                type="button"
                className="text-xs text-blue-600 hover:underline px-2 py-1"
                onClick={() => setOpen(false)}
              >
                Done
              </button>
            </div>
          </div>
        )}
      </div>
    );
  }

  // Single select fallback
  return (
    <select
      className={className}
      value={Array.isArray(value) ? "" : value}
      onChange={e => onChange(e.target.value)}
    >
      <option value="">{placeholder}</option>
      {countries.map(country => (
        <option key={country} value={country}>{country}</option>
      ))}
    </select>
  );
};

export default CountryDropdown; 
