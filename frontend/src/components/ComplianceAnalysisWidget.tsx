import React, { useEffect, useState } from 'react';
import Card from './Card';
import { useCountryContext } from '../contexts/CountryContext';

interface Policy {
  title: string;
  source?: string;
  text?: string;
}

interface ComplianceAnalysisWidgetProps {
  domain?: string;
  searchQuery?: string;
}

const ComplianceAnalysisWidget: React.FC<ComplianceAnalysisWidgetProps> = ({ domain, searchQuery }) => {
  const { selectedCountries } = useCountryContext();
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedCountries.length) return;
    setLoading(true);
    setError(null);
    // Example: Adjust endpoint and payload as needed
    fetch('http://localhost:5001/api/policies/relevant', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        countries: selectedCountries,
        domain,
        search: searchQuery
      })
    })
      .then(res => res.json())
      .then(data => {
        setPolicies(data.policies || []);
        setLoading(false);
      })
      .catch(err => {
        setError('Failed to fetch policies.');
        setLoading(false);
      });
  }, [selectedCountries, domain, searchQuery]);

  return (
    <Card className="custom-border p-4 h-auto min-h-[200px]">
      <h3 className="text-lg text-[#1975d4] font-bold mb-2">Relevant Policies</h3>
      {loading && <div className="text-gray-500">Loading...</div>}
      {error && <div className="text-red-500">{error}</div>}
      {!loading && !error && policies.length === 0 && (
        <div className="text-gray-500">No relevant policies found for your selection.</div>
      )}
      {!loading && !error && policies.length > 0 && (
        <ul className="space-y-2">
          {policies.map((policy, idx) => (
            <li key={idx} className="border-b pb-2">
              <div className="font-semibold">{policy.title}</div>
              {policy.source && (
                <a href={policy.source} target="_blank" rel="noopener noreferrer" className="text-xs text-blue-600 underline">Source</a>
              )}
              {policy.text && <div className="text-xs text-gray-700 mt-1">{policy.text}</div>}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
};

export default ComplianceAnalysisWidget; 
