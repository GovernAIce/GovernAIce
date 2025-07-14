import React, { useEffect, useState } from 'react';
import Card from './Card';
import { useCountryContext } from '../contexts/CountryContext';
import { policyAPI } from '../api';

interface Policy {
  title: string;
  source?: string;
  text?: string;
  country?: string;
  domain?: string;
}

interface ComplianceAnalysisWidgetProps {
  domain?: string;
  searchQuery?: string;
  uploadedFile: File | null;
}

const ComplianceAnalysisWidget: React.FC<ComplianceAnalysisWidgetProps> = ({ 
  domain, 
  searchQuery, 
  uploadedFile
}) => {
  const { selectedCountries, hasCountries } = useCountryContext();
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastSearch, setLastSearch] = useState<string>('');

  // Enhanced search query generation based on document analysis and countries
  const generateSearchQuery = () => {
    let query = searchQuery || '';
    if (domain) {
      query += ` ${domain}`;
    }
    const countryKeywords = selectedCountries.map(country => {
      const keywords = {
        'USA': 'privacy data protection consumer rights',
        'EU': 'GDPR data protection privacy rights',
        'UK': 'data protection privacy consumer rights',
        'Canada': 'PIPEDA privacy data protection',
        'Australia': 'privacy data protection consumer',
        'Japan': 'privacy data protection consumer',
        'China': 'data security privacy protection',
        'India': 'data protection privacy consumer',
        'Brazil': 'LGPD privacy data protection',
        'Singapore': 'privacy data protection consumer',
        'South Korea': 'privacy data protection consumer',
        'Saudi Arabia': 'data protection privacy consumer',
        'UAE': 'data protection privacy consumer',
        'Taiwan': 'privacy data protection consumer'
      };
      return keywords[country as keyof typeof keywords] || 'privacy data protection';
    }).join(' ');
    query += ` ${countryKeywords}`;
    if (uploadedFile?.name) {
      const fileName = uploadedFile.name.toLowerCase();
      if (fileName.includes('privacy') || fileName.includes('data')) {
        query += ' privacy data protection';
      }
      if (fileName.includes('security') || fileName.includes('cyber')) {
        query += ' security cybersecurity';
      }
      if (fileName.includes('ai') || fileName.includes('artificial')) {
        query += ' artificial intelligence AI';
      }
      if (fileName.includes('health') || fileName.includes('medical')) {
        query += ' healthcare medical privacy';
      }
      if (fileName.includes('financial') || fileName.includes('bank')) {
        query += ' financial banking compliance';
      }
    }
    return query.trim();
  };

  const canFetchPolicies = () => {
    return hasCountries && uploadedFile;
  };

  const getStatusMessage = () => {
    if (!hasCountries && !uploadedFile) {
      return {
        type: 'warning',
        message: 'Please select countries and upload a document to view relevant policies.',
        icon: '⚠️'
      };
    }
    if (!hasCountries) {
      return {
        type: 'warning',
        message: 'Please select countries in the Explore Policy section to view relevant policies.',
        icon: '🌍'
      };
    }
    if (!uploadedFile) {
      return {
        type: 'info',
        message: 'Please upload a document to view relevant policies for your selected countries.',
        icon: '📄'
      };
    }
    return null;
  };

  useEffect(() => {
    if (!canFetchPolicies()) {
      setPolicies([]);
      setError(null);
      return;
    }
    const currentSearch = generateSearchQuery();
    const searchKey = `${selectedCountries.join(',')}-${currentSearch}-${uploadedFile?.name}`;
    if (searchKey === lastSearch) {
      return;
    }
    setLoading(true);
    setError(null);
    setLastSearch(searchKey);
    policyAPI.getRelevantPolicies(selectedCountries, domain, currentSearch)
      .then(response => {
        const data = response.data;
        setPolicies(data.policies || []);
        setLoading(false);
      })
      .catch(err => {
        setError('Failed to fetch relevant policies. Please try again.');
        setLoading(false);
      });
  }, [selectedCountries, domain, searchQuery, uploadedFile, hasCountries]);

  const getCountryBadge = (country: string) => {
    const colors = {
      'USA': 'bg-blue-100 text-blue-800',
      'EU': 'bg-purple-100 text-purple-800',
      'UK': 'bg-red-100 text-red-800',
      'Canada': 'bg-red-100 text-red-800',
      'Australia': 'bg-green-100 text-green-800',
      'Japan': 'bg-red-100 text-red-800',
      'China': 'bg-red-100 text-red-800',
      'India': 'bg-orange-100 text-orange-800',
      'Brazil': 'bg-green-100 text-green-800',
      'Singapore': 'bg-red-100 text-red-800',
      'South Korea': 'bg-blue-100 text-blue-800',
      'Saudi Arabia': 'bg-green-100 text-green-800',
      'UAE': 'bg-green-100 text-green-800',
      'Taiwan': 'bg-blue-100 text-blue-800'
    };
    const colorClass = colors[country as keyof typeof colors] || 'bg-gray-100 text-gray-800';
    return (
      <span className={`inline-block px-2 py-1 text-xs rounded-full ${colorClass}`}>
        {country}
      </span>
    );
  };

  const statusMessage = getStatusMessage();

  return (
    <Card className="custom-border p-4 h-auto min-h-[200px]">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg text-[#1975d4] font-bold">Relevant Policies</h3>
        <div className="flex items-center gap-2">
          {canFetchPolicies() && (
            <div className="text-xs text-gray-500">
              {policies.length}/5 policies found
            </div>
          )}
        </div>
      </div>
      {statusMessage && (
        <div className={`text-sm p-3 rounded-lg mb-3 ${
          statusMessage.type === 'warning' 
            ? 'bg-amber-50 text-amber-700 border border-amber-200' 
            : 'bg-blue-50 text-blue-700 border border-blue-200'
        }`}>
          <div className="flex items-center gap-2">
            <span className="text-lg">{statusMessage.icon}</span>
            <span>{statusMessage.message}</span>
          </div>
        </div>
      )}
      {loading && (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-gray-600">Analyzing document and finding relevant policies...</span>
        </div>
      )}
      {error && (
        <div className="text-red-500 text-sm p-3 bg-red-50 rounded-lg">
          ❌ {error}
        </div>
      )}
      {!loading && !error && policies.length === 0 && (
        <div className="text-gray-500 text-sm p-4 text-center">
          <div className="mb-2">📋</div>
          No relevant policies found for the selected criteria.
          <br />
          <span className="text-xs">Try selecting different countries or uploading a different document.</span>
        </div>
      )}
      {!loading && !error && canFetchPolicies() && policies.length > 0 && (
        <div className="space-y-3">
          {policies.map((policy, idx) => (
            <div key={idx} className="border border-gray-200 rounded-lg p-3 hover:bg-gray-50 transition-colors">
              <div className="flex items-center gap-2 mb-1">
                <h4 className="font-semibold text-sm text-gray-900 flex-1">
                  {policy.title}
                </h4>
                {policy.country && getCountryBadge(policy.country)}
              </div>
              <div className="flex items-center justify-between">
                {policy.source && (
                  <a 
                    href={policy.source} 
                    target="_blank" 
                    rel="noopener noreferrer" 
                    className="text-xs text-blue-600 hover:text-blue-800 underline"
                  >
                    View Source
                  </a>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
};

export default ComplianceAnalysisWidget; 
