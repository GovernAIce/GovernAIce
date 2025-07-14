import React, { useState } from 'react';
import Card from './Card';
// import { uploadFile } from './UploadProjectWidget'; // Assuming this is where uploadFile is defined
import { useCountryContext } from '../contexts/CountryContext';
import { policyAPI } from '../api';

interface Policy {
    title: string;
    source?: string;
    text?: string;
    country?: string;
    domain?: string;
    excellent_points: string[];
    major_gaps: string[];
    compliance_score: number;
    policy?: string;
}
  
  interface ComplianceAnalysisWidgetProps {
    domain?: string;
    searchQuery?: string;
    uploadedFile?: File | null;
    insights: Policy[];
    onClose?: () => void;
}



const ComplianceResults: React.FC<ComplianceAnalysisWidgetProps> =  ({ 
    domain, 
    searchQuery, 
    uploadedFile,
    insights,
    onClose
}) => {
    const { selectedCountries, hasCountries } = useCountryContext();
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastSearch, setLastSearch] = useState<string>('');

  // Enhanced search query generation based on document analysis and countries
  const generateSearchQuery = () => {
    let query = searchQuery || '';
    
    // Add domain to search if available
    if (domain) {
      query += ` ${domain}`;
    }
    
    // Add country-specific keywords based on selected countries
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
    

    
    // Add file-specific keywords based on file name
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

  // Check if we have all required conditions to fetch policies
  const canFetchPolicies = () => {
    return hasCountries && uploadedFile;
  };

//   // Get the current status message
//   const getStatusMessage = () => {
//     if (!hasCountries && !uploadedFile) {
//       return {
//         type: 'warning',
//         message: 'Please select countries and upload a document to view relevant policies.',
//         icon: '⚠️'
//       };
//     }
    
//     if (!hasCountries) {
//       return {
//         type: 'warning',
//         message: 'Please select countries in the Explore Policy section to view relevant policies.',
//         icon: '🌍'
//       };
//     }
    
//     if (!uploadedFile) {
//       return {
//         type: 'info',
//         message: 'Please upload a document to view relevant policies for your selected countries.',
//         icon: '📄'
//       };
//     }
    
//     return null;
//   };

// //   // Test function to manually trigger API call
// //   const testApiCall = async () => {
// //     console.log('Testing API call manually...');
// //     setLoading(true);
// //     setError(null);
    
// //     try {
// //       const response = await policyAPI.getRelevantPolicies(['USA'], 'AI', 'privacy');
// //       console.log('Manual API test response:', response.data);
// //       setPolicies(response.data.policies || []);
// //     } catch (err) {
// //       console.error('Manual API test error:', err);
// //       setError('Manual test failed: ' + (err as any).message);
// //     } finally {
// //       setLoading(false);
// //     }
// //   };

// //   useEffect(() => {
// //     // Debug logging
// //     console.log('ComplianceAnalysisWidget useEffect triggered:', {
// //       hasCountries,
// //       uploadedFile: uploadedFile?.name,
// //       selectedCountries,
// //       domain,
// //       searchQuery,
// //       analysisResults: !!analysisResults
// //     });

//     // Only fetch policies if we have both countries and uploaded file
//     if (!canFetchPolicies()) {
//       console.log('Cannot fetch policies - missing conditions:', {
//         hasCountries,
//         hasUploadedFile: !!uploadedFile
//       });
//       setPolicies([]);
//       setError(null);
//       return;
//     }

//     const currentSearchQuery = generateSearchQuery();
//     const searchKey = `${selectedCountries.join(',')}-${currentSearchQuery}-${uploadedFile?.name}`;
  

//     console.log('Generated search query:', currentSearchQuery);
//     console.log('Search key:', searchKey);
//     console.log('Last search key:', lastSearch);
    
//     // Only fetch if search criteria changed
//     if (searchKey === lastSearch) {
//       console.log('Search criteria unchanged, skipping API call');
//       return;
//     }
    
//     console.log('Making API call to fetch policies...');
//     setLoading(true);
//     setError(null);
//     setLastSearch(searchKey);

//     policyAPI.getRelevantPolicies(selectedCountries, domain, currentSearchQuery)
//       .then(response => {
//         console.log('API response received:', response.data);
//         const data = response.data;
//         setPolicies(data.policies || []);
//         setLoading(false);
//       })
//       .catch(err => {
//         console.error('Error fetching policies:', err);
//         setError('Failed to fetch relevant policies. Please try again.');
//         setLoading(false);
//       });
//   }, [selectedCountries, domain, searchQuery, uploadedFile, analysisResults, hasCountries]);

//   const getCountryBadge = (country: string) => {
//     const colors = {
//       'USA': 'bg-blue-100 text-blue-800',
//       'EU': 'bg-purple-100 text-purple-800',
//       'UK': 'bg-red-100 text-red-800',
//       'Canada': 'bg-red-100 text-red-800',
//       'Australia': 'bg-green-100 text-green-800',
//       'Japan': 'bg-red-100 text-red-800',
//       'China': 'bg-red-100 text-red-800',
//       'India': 'bg-orange-100 text-orange-800',
//       'Brazil': 'bg-green-100 text-green-800',
//       'Singapore': 'bg-red-100 text-red-800',
//       'South Korea': 'bg-blue-100 text-blue-800',
//       'Saudi Arabia': 'bg-green-100 text-green-800',
//       'UAE': 'bg-green-100 text-green-800',
//       'Taiwan': 'bg-blue-100 text-blue-800'
//     };
    
//     const colorClass = colors[country as keyof typeof colors] || 'bg-gray-100 text-gray-800';
    
//     return (
//       <span className={`inline-block px-2 py-1 text-xs rounded-full ${colorClass}`}>
//         {country}
//       </span>
//     );
//   };

//   const statusMessage = getStatusMessage();

//   // Debug render logging
//   console.log('ComplianceAnalysisWidget render:', {
//     policies: policies.length,
//     loading,
//     error,
//     canFetchPolicies: canFetchPolicies(),
//     statusMessage: !!statusMessage
//   });

  const [currentIndex, setCurrentIndex] = useState(0);
  const [expandedPolicy, setExpandedPolicy] = useState<string | null>(null);

  const handlePrevious = () => {
    setCurrentIndex(prev => prev > 0 ? prev - 1 : insights.length - 1);
    setExpandedPolicy(null);
  };

  const handleNext = () => {
    setCurrentIndex(prev => prev < insights.length - 1 ? prev + 1 : 0);
    setExpandedPolicy(null);
  };

  const toggleExpanded = (policy: string) => {
    setExpandedPolicy(expandedPolicy === policy ? null : policy);
  };

  if (!insights || insights.length === 0) {
    return null;
  }

  const currentInsight = insights[currentIndex];

  return (
    <Card className="custom-border p-4 h-auto min-h-[300px]">
      <div className="flex flex-col h-full">
        {/* Header with navigation */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg text-[#1975d4] font-bold">Analysis Results</h3>
          <div className="flex items-center gap-2">
            <button
              onClick={handlePrevious}
              className="p-1 rounded-full hover:bg-gray-100 transition-colors"
              disabled={insights.length <= 1}
            >
              <img src="/icons/Arrow10.svg" alt="Previous" className="w-4 h-4 rotate-180" />
            </button>
            <span className="text-xs text-gray-600">
              {currentIndex + 1} of {insights.length}
            </span>
            <button
              onClick={handleNext}
              className="p-1 rounded-full hover:bg-gray-100 transition-colors"
              disabled={insights.length <= 1}
            >
              <img src="/icons/Arrow10.svg" alt="Next" className="w-4 h-4" />
            </button>
            {onClose && (
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-gray-600 text-lg font-bold"
              >
                ×
              </button>
            )}
          </div>
        </div>


        {/* Policy Card */}
        <div className="flex-1 bg-white rounded-lg border border-gray-200 p-4">
          <div className="mb-3">
            <h4 className="font-semibold text-gray-800 mb-2">{currentInsight.policy}</h4>
            
            {/* Compliance Score */}
            <div className="flex items-center gap-2 mb-3">
              <span className="text-sm text-gray-600">Compliance Score:</span>
              <div className="flex items-center gap-1">
                <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    className={`h-full rounded-full transition-all duration-300 ${
                      currentInsight.compliance_score >= 80 ? 'bg-green-500' :
                      currentInsight.compliance_score >= 60 ? 'bg-yellow-500' :
                      'bg-red-500'
                    }`}
                    style={{ width: `${currentInsight.compliance_score}%` }}
                  ></div>
                </div>
                <span className="text-sm font-medium">
                  {currentInsight.compliance_score}%
                </span>
              </div>
            </div>

            {/* Policy Details */}
            <div className="mb-3">
              <h5 className="text-sm font-medium text-gray-700 mb-1">Policy Overview</h5>
              <p className="text-xs text-gray-600 leading-relaxed">
                {currentInsight.policy}
              </p>
            </div>

            {/* Expandable Sections */}
            <div className="space-y-2">
              {/* Excellent Points */}
              {currentInsight.excellent_points && currentInsight.excellent_points.length > 0 && (
                <div>
                  <button
                    onClick={() => toggleExpanded('excellent')}
                    className="flex items-center justify-between w-full text-left text-xs font-medium text-green-700 hover:text-green-800"
                  >
                    <span>✅ Excellent Points ({currentInsight.excellent_points.length})</span>
                    <span className="text-lg">{expandedPolicy === 'excellent' ? '−' : '+'}</span>
                  </button>
                  {expandedPolicy === 'excellent' && (
                    <div className="mt-2 pl-4 border-l-2 border-green-200">
                      <ul className="text-xs text-gray-600 space-y-1">
                        {currentInsight.excellent_points.map((point, index) => (
                          <li key={index} className="list-disc list-inside">{point}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {/* Major Gaps */}
              {currentInsight.major_gaps && currentInsight.major_gaps.length > 0 && (
                <div>
                  <button
                    onClick={() => toggleExpanded('gaps')}
                    className="flex items-center justify-between w-full text-left text-xs font-medium text-red-700 hover:text-red-800"
                  >
                    <span>⚠️ Major Gaps ({currentInsight.major_gaps.length})</span>
                    <span className="text-lg">{expandedPolicy === 'gaps' ? '−' : '+'}</span>
                  </button>
                  {expandedPolicy === 'gaps' && (
                    <div className="mt-2 pl-4 border-l-2 border-red-200">
                      <ul className="text-xs text-gray-600 space-y-1">
                        {currentInsight.major_gaps.map((gap, index) => (
                          <li key={index} className="list-disc list-inside">{gap}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Navigation Dots */}
        {insights.length > 1 && (
          <div className="flex justify-center gap-1 mt-3">
            {insights.map((_, index) => (
              <button
                key={index}
                onClick={() => {
                  setCurrentIndex(index);
                  setExpandedPolicy(null);
                }}
                className={`w-2 h-2 rounded-full transition-colors ${
                  index === currentIndex ? 'bg-blue-500' : 'bg-gray-300'
                }`}
              />
            ))}
          </div>
        )}
      </div>
    </Card>
  );
};

export default ComplianceResults; 
