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
  policies?: Policy[];
}

const ComplianceAnalysisWidget: React.FC<ComplianceAnalysisWidgetProps> = ({
  domain,
  searchQuery,
  uploadedFile,
  policies: externalPolicies,
}) => {
  const { selectedCountries, hasCountries } = useCountryContext();
  const [policies, setPolicies] = useState<Policy[]>(externalPolicies || []);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastSearch, setLastSearch] = useState<string>('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [showFullText, setShowFullText] = useState(false);
  const [fullPolicyText, setFullPolicyText] = useState<string | null>(null); // Store full policy text
  const [fullPolicyLoading, setFullPolicyLoading] = useState(false); // Loading state for full policy
  const [fullPolicyError, setFullPolicyError] = useState<string | null>(null); // Error state for full policy

  const generateSearchQuery = () => {
    let query = searchQuery || '';
    if (domain) query += ` ${domain}`;
    const countryKeywords = selectedCountries.map((country) => {
      const keywords = {
        USA: 'privacy data protection consumer rights',
        EU: 'GDPR data protection privacy rights',
        UK: 'data protection privacy consumer rights',
        Canada: 'PIPEDA privacy data protection',
        Australia: 'privacy data protection consumer',
        Japan: 'privacy data protection consumer',
        China: 'data security privacy protection',
        India: 'data protection privacy consumer',
        Brazil: 'LGPD privacy data protection',
        Singapore: 'privacy data protection consumer',
        'South Korea': 'privacy data protection consumer',
        'Saudi Arabia': 'data protection privacy consumer',
        UAE: 'data protection privacy consumer',
        Taiwan: 'privacy data protection consumer',
      };
      return keywords[country as keyof typeof keywords] || 'privacy data protection';
    }).join(' ');
    query += ` ${countryKeywords}`;

    if (uploadedFile?.name) {
      const name = uploadedFile.name.toLowerCase();
      if (name.includes('privacy') || name.includes('data')) query += ' privacy data protection';
      if (name.includes('security') || name.includes('cyber')) query += ' security cybersecurity';
      if (name.includes('ai') || name.includes('artificial')) query += ' artificial intelligence AI';
      if (name.includes('health') || name.includes('medical')) query += ' healthcare medical privacy';
      if (name.includes('financial') || name.includes('bank')) query += ' financial banking compliance';
    }
    return query.trim();
  };

  const canFetchPolicies = () => hasCountries && uploadedFile;

  const getStatusMessage = () => {
    if (!hasCountries && !uploadedFile)
      return { type: 'warning', message: 'Please select countries and upload a document.', icon: '⚠️' };
    if (!hasCountries)
      return { type: 'warning', message: 'Please select countries in Explore Policy.', icon: '🌍' };
    if (!uploadedFile)
      return { type: 'info', message: 'Please upload a document to view relevant policies.', icon: '📄' };
    return null;
  };

  // Fetch relevant policies
  useEffect(() => {
    if (externalPolicies) {
      setPolicies(externalPolicies);
      return;
    }
    if (!canFetchPolicies()) {
      setPolicies([]);
      setError(null);
      return;
    }

    const currentSearch = generateSearchQuery();
    const searchKey = `${selectedCountries.join(',')}-${currentSearch}-${uploadedFile?.name}`;
    if (searchKey === lastSearch) return;

    setLoading(true);
    setError(null);
    setLastSearch(searchKey);

    policyAPI
      .use_case_one(selectedCountries, domain, currentSearch)
      .then((response) => {
        const data = response.data;
        setPolicies(data.policies || []);
        setCurrentIndex(0);
        setShowFullText(false);
        setFullPolicyText(null); // Reset full policy text
        setFullPolicyError(null); // Reset full policy error
        setLoading(false);
      })
      .catch(() => {
        setError('Failed to fetch relevant policies. Please try again.');
        setLoading(false);
      });
  }, [selectedCountries, domain, searchQuery, uploadedFile, hasCountries, externalPolicies]);

  // Fetch full policy document
  const fetchFullPolicy = async () => {
    if (!policies[currentIndex]?.country || !policies[currentIndex]?.title) return;

    setFullPolicyLoading(true);
    setFullPolicyError(null);
    setFullPolicyText(null);

    try {
      const response = await policyAPI.get_policy_document(
        policies[currentIndex].title, // Uses parent_title from /api/policies/relevant
        policies[currentIndex].country!
      );
      setFullPolicyText(response.data.text || 'No content available.');
    } catch (err) {
      setFullPolicyError('Failed to fetch full policy document. Please try again.');
    } finally {
      setFullPolicyLoading(false);
    }
  };

  // Handle View Full Policy button click
  const handleViewFullPolicy = () => {
    if (showFullText) {
      setShowFullText(false);
      setFullPolicyText(null); // Clear full policy text when hiding
      setFullPolicyError(null); // Clear errors
    } else {
      setShowFullText(true);
      fetchFullPolicy(); // Fetch full policy when showing
    }
  };

  const statusMessage = getStatusMessage();

  return (
    <Card className="custom-border p-4 h-auto min-h-[200px]">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg text-[#1975d4] rubik-bold">Relevant Policies</h3>
        <div className="text-xs text-gray-500 rubik-regular">
          {policies.length > 0 && `${currentIndex + 1}/${policies.length} policies`}
        </div>
      </div>

      {statusMessage && (
        <div
          className={`text-sm p-3 rounded-lg mb-3 ${
            statusMessage.type === 'warning'
              ? 'bg-amber-50 text-amber-700 border border-amber-200'
              : 'bg-blue-50 text-blue-700 border border-blue-200'
          }`}
        >
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
        <div className="text-red-500 text-sm p-3 bg-red-50 rounded-lg rubik-regular">❌ {error}</div>
      )}

      {!loading && !error && policies.length === 0 && (
        <div className="text-gray-500 text-sm p-4 text-center rubik-regular">
          <div className="mb-2">📋</div>
          No relevant policies found.
          <br />
          <span className="text-xs">Try different countries or another document.</span>
        </div>
      )}

      {!loading && !error && policies.length > 0 && (
        <div className="space-y-2">
          <div className="border border-gray-200 rounded px-4 py-2 bg-white text-sm">
            <div className="font-bold text-[#1975d4] text-base mb-1 rubik-bold">
              {policies[currentIndex].title}
            </div>
            <div className="text-gray-700 text-sm mb-2 whitespace-pre-line rubik-regular">
              {showFullText ? (
                fullPolicyLoading ? (
                  <div className="flex items-center justify-center py-4">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                    <span className="ml-2 text-gray-600">Loading full policy document...</span>
                  </div>
                ) : fullPolicyError ? (
                  <div className="text-red-500 text-sm rubik-regular">❌ {fullPolicyError}</div>
                ) : (
                  fullPolicyText || 'No content available.'
                )
              ) : (
                (policies[currentIndex].text?.slice(0, 300) || 'No content available.') + '...'
              )}
            </div>

            {/* View Full Policy Button */}
            <div className="flex justify-center mb-3">
              <button
                onClick={handleViewFullPolicy}
                className="bg-gradient-to-r from-[#2196f3] to-[#21cbf3] text-white px-4 py-2 rounded-full text-sm font-medium shadow hover:opacity-90 rubik-medium"
                disabled={fullPolicyLoading}
              >
                {showFullText ? 'Hide Full Policy' : 'View Full Policy Document'}
              </button>
            </div>

            {/* Navigation */}
            <div className="flex justify-between items-center mt-4">
              <button
                disabled={currentIndex === 0}
                onClick={() => {
                  setCurrentIndex((prev) => prev - 1);
                  setShowFullText(false);
                  setFullPolicyText(null); // Reset full policy text
                  setFullPolicyError(null); // Reset full policy error
                }}
                className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded disabled:opacity-50 rubik-medium"
              >
                ⬅️ Previous
              </button>
              <span className="text-xs text-gray-500 rubik-regular">
                {currentIndex + 1} / {policies.length}
              </span>
              <button
                disabled={currentIndex === policies.length - 1}
                onClick={() => {
                  setCurrentIndex((prev) => prev + 1);
                  setShowFullText(false);
                  setFullPolicyText(null); // Reset full policy text
                  setFullPolicyError(null); // Reset full policy error
                }}
                className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded disabled:opacity-50 rubik-medium"
              >
                Next ➡️
              </button>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
};

export default ComplianceAnalysisWidget;