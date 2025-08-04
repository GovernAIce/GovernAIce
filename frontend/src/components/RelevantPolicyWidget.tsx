import React, { useEffect, useState } from 'react';
import Card from './Card';
import { policyAPI } from '../api';
import leftArrow from '../assets/left_arrow.svg';
import rightArrow from '../assets/right_arrow.svg';

interface Policy {
  title: string;
  source?: string;
  text?: string;
  country?: string;
  domain?: string;
  regulator?: string;
}

interface RelevantPolicyWidgetProps {
  uploadedFile: File | null;
  policies?: Policy[];
}

const RelevantPolicyWidget: React.FC<RelevantPolicyWidgetProps> = ({
  uploadedFile,
  policies: externalPolicies,
}) => {
  const [policies, setPolicies] = useState<Policy[]>(externalPolicies || []);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [currentPolicyText, setCurrentPolicyText] = useState<string>('');

  // When external policies are provided (from upload analysis), use them
  useEffect(() => {
    if (externalPolicies && externalPolicies.length > 0) {
      setPolicies(externalPolicies);
      setCurrentIndex(0);
      setError(null);
      return;
    }

    // If no external policies and no file uploaded, clear policies
    if (!uploadedFile) {
      setPolicies([]);
      setError(null);
      return;
    }

    // If we have a file but no external policies, show a message
    if (uploadedFile && (!externalPolicies || externalPolicies.length === 0)) {
      setPolicies([]);
      setError('Please complete the analysis to view relevant policies.');
      return;
    }
  }, [externalPolicies, uploadedFile]);

  // Fetch full policy text when current policy changes
  useEffect(() => {
    if (policies.length > 0 && currentIndex < policies.length) {
      const currentPolicy = policies[currentIndex];
      if (currentPolicy.title && currentPolicy.country) {
        fetchPolicyText(currentPolicy.title, currentPolicy.country);
      }
    }
  }, [currentIndex, policies]);

  const fetchPolicyText = async (title: string, country: string) => {
    try {
      setLoading(true);
      const response = await policyAPI.view_policy_document(title, country);
      setCurrentPolicyText(response.data.text || 'No content available.');
    } catch (error) {
      console.error('Error fetching policy text:', error);
      setCurrentPolicyText('Unable to load policy content.');
    } finally {
      setLoading(false);
    }
  };

  const downloadPolicyDocument = () => {
    if (!currentPolicyText || currentPolicyText === 'No content available.' || currentPolicyText === 'Unable to load policy content.') {
      alert('No policy content available to download.');
      return;
    }

    const currentPolicy = policies[currentIndex];
    const fileName = `${currentPolicy.title.replace(/[^a-zA-Z0-9]/g, '_')}_${currentPolicy.country || 'Unknown'}.txt`;
    
    const blob = new Blob([currentPolicyText], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  };

  const getStatusMessage = () => {
    if (!uploadedFile)
      return { type: 'info', message: 'Please upload a document to view relevant policies.', icon: '📄' };
    if (!externalPolicies && uploadedFile)
      return { type: 'warning', message: 'Please complete the analysis to view relevant policies.', icon: '⚙️' };
    if (policies.length === 0 && uploadedFile && externalPolicies)
      return { type: 'info', message: 'No relevant policies found for your document.', icon: '🔍' };
    return null;
  };

  const statusMessage = getStatusMessage();

  return (
    <Card className="bg-white rounded-2xl shadow-lg p-5 h-full custom-border p-2 h-full">
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-base text-[#1975d4] font-bold">Relevant Policies</h3>
        <div className="text-xs text-gray-500">
          {policies.length > 0 && `${currentIndex + 1}/${policies.length} policies`}
        </div>
      </div>

      <div className="flex-1 flex flex-col">
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
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mx-auto mb-2"></div>
              <span className="text-gray-600 text-sm">Loading policy content...</span>
            </div>
          </div>
        )}

        {error && !statusMessage && (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-red-500 text-sm p-3 bg-red-50 rounded-lg text-center">
              ❌ {error}
            </div>
          </div>
        )}

        {!loading && !error && policies.length === 0 && !statusMessage && (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-gray-500 text-sm p-4 text-center">
              <div className="mb-2">📋</div>
              No relevant policies found.
              <br />
              <span className="text-xs">Try uploading a different document.</span>
            </div>
          </div>
        )}

        {!loading && !error && policies.length > 0 && (
          <div className="flex-1 flex flex-col min-h-0">
            <div className="rounded px-3 py-2 bg-white text-sm flex-1 overflow-y-auto">
              <div className="font-bold text-[#1975d4] text-sm mb-1 truncate" title={policies[currentIndex].title}>
                {policies[currentIndex].title}
              </div>
              {policies[currentIndex].regulator && (
                <div className="text-xs text-gray-600 mb-1 truncate" title={policies[currentIndex].regulator}>
                  Regulator: {policies[currentIndex].regulator}
                </div>
              )}
              {policies[currentIndex].country && (
                <div className="text-xs text-gray-600 mb-1">
                  Country: {policies[currentIndex].country}
                </div>
              )}
              {policies[currentIndex].source && (
                <div className="text-xs text-gray-600 mb-1 truncate" title={policies[currentIndex].source}>
                  Source: <a href={policies[currentIndex].source} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                    {policies[currentIndex].source.length > 50 ? policies[currentIndex].source.substring(0, 50) + '...' : policies[currentIndex].source}
                  </a>
                </div>
              )}
              <div className="text-gray-700 text-xs mb-2 whitespace-pre-line max-h-20 overflow-y-auto">
                {(currentPolicyText.slice(0, 200) || 'No content available.') + '...'}
                <br />
                <span className="text-xs text-blue-600">Click "Download Full Policy Document" to save the complete policy as TXT</span>
              </div>

              {/* View Full Policy Button */}
              <div className="flex justify-center items-center gap-2 mb-3">
                <button
                  onClick={downloadPolicyDocument}
                  className="bg-gradient-to-r from-[#9FD8FF] to-[#2196f3] text-white px-3 py-1 rounded-[50px] text-sm font-medium shadow opacity-100 transition-opacity"
                >
                  View Full Policy Document
                </button>
              </div>
            </div>

            {/* Navigation - Fixed at bottom with proper spacing */}
            <div className="flex justify-between items-center mt-3 pt-2 border-t border-gray-200">
              <button
                disabled={currentIndex === 0}
                onClick={() => {
                  setCurrentIndex((prev) => prev - 1);
                }}
                className="p-2 rounded-full hover:bg-gray-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <img
                  src="/icons/Arrow11.svg"
                  alt="Previous"
                  className="w-5 h-5"
                  style={{ filter: currentIndex === 0 ? 'opacity(0.3)' : 'none' }}
                />
              </button>
        
              <button
                disabled={currentIndex === policies.length - 1}
                onClick={() => {
                  setCurrentIndex((prev) => prev + 1);
                }}
                className="p-2 rounded-full hover:bg-gray-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <img
                  src="/icons/Arrow10.svg"
                  alt="Next"
                  className="w-5 h-5"
                  style={{ filter: currentIndex === policies.length - 1 ? 'opacity(0.3)' : 'none' }}
                />
              </button>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default RelevantPolicyWidget;
