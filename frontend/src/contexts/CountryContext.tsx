import React, { createContext, useContext, useState, ReactNode } from 'react';

interface CountryContextType {
  selectedCountries: string[];
  setSelectedCountries: (countries: string[]) => void;
  addCountry: (country: string) => void;
  removeCountry: (country: string) => void;
  clearCountries: () => void;
  hasCountries: boolean;
}

const CountryContext = createContext<CountryContextType | undefined>(undefined);

interface CountryProviderProps {
  children: ReactNode;
}

export const CountryProvider: React.FC<CountryProviderProps> = ({ children }) => {
  const [selectedCountries, setSelectedCountries] = useState<string[]>([]);

  const addCountry = (country: string) => {
    if (!selectedCountries.includes(country)) {
      setSelectedCountries([...selectedCountries, country]);
    }
  };

  const removeCountry = (country: string) => {
    setSelectedCountries(selectedCountries.filter(c => c !== country));
  };

  const clearCountries = () => {
    setSelectedCountries([]);
  };

  const hasCountries = selectedCountries.length > 0;

  const value: CountryContextType = {
    selectedCountries,
    setSelectedCountries,
    addCountry,
    removeCountry,
    clearCountries,
    hasCountries,
  };

  return (
    <CountryContext.Provider value={value}>
      {children}
    </CountryContext.Provider>
  );
};

export const useCountryContext = (): CountryContextType => {
  const context = useContext(CountryContext);
  if (context === undefined) {
    throw new Error('useCountryContext must be used within a CountryProvider');
  }
  return context;
}; 
