import React, { useEffect, useState } from 'react';
import Card from './Card';

interface Policy {
  title: string;
  regulator: string;
  country: string;
}

const RegulatoryPolicy = () => {
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPolicies = async () => {
      try {
        setLoading(true);
        const response = await fetch('http://localhost:5001/api/policies/relevant', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            countries: ['India', 'Germany'],
            user_input: 'My startup deals with AI-based financial data analytics for children and young adults.',
            domain: 'AI regulation',
            search: 'financial data minors',
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        console.log('API Response:', data); // Debug log
        setPolicies(data.policies || []);
      } catch (err: any) {
        console.error('Fetch error:', err);
        setError(err.message || 'Failed to fetch policies');
      } finally {
        setLoading(false);
      }
    };

    fetchPolicies();
  }, []);

  return (
    <Card className="p-6 rounded-2xl border border-gray-200 shadow-md w-full max-w-3xl mx-auto">
      <h2 className="text-2xl font-bold text-blue-600 mb-1">Relevant Policies & Regulators</h2>
      <p className="text-sm text-gray-500 mb-4">(pop up relevant information)</p>

      {loading && <p className="text-sm text-gray-500">Loading policies...</p>}
      {error && <p className="text-sm text-red-500">Error: {error}</p>}
      {!loading && !error && policies.length > 0 ? (
        <div className="space-y-4">
          {policies.map((policy, index) => (
            <div key={index} className="border-b border-gray-200 pb-2">
              <h3 className="text-lg font-medium text-gray-800">{policy.title}</h3>
              <p className="text-sm text-gray-600">Regulator: {policy.regulator}</p>
              <p className="text-sm text-gray-600">Country: {policy.country}</p>
            </div>
          ))}
        </div>
      ) : (
        !loading && !error && <p className="text-sm text-gray-500">No policies found.</p>
      )}
    </Card>
  );
};

export default RegulatoryPolicy;