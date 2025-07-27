import React, { useState, useEffect } from 'react';
import { mlAPI, MLStatus } from '../api/ml';

interface MLTestWidgetProps {
  className?: string;
}

const MLTestWidget: React.FC<MLTestWidgetProps> = ({ className = '' }) => {
  const [mlStatus, setMLStatus] = useState<MLStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [testResult, setTestResult] = useState<any>(null);
  const [documentText, setDocumentText] = useState('');
  const [selectedCountry, setSelectedCountry] = useState('USA');

  const countries = ['USA', 'EU', 'UK', 'Canada', 'Australia', 'Singapore'];

  useEffect(() => {
    checkMLStatus();
  }, []);

  const checkMLStatus = async () => {
    try {
      setLoading(true);
      const response = await mlAPI.getMLStatus();
      setMLStatus(response.data);
    } catch (error) {
      console.error('Error checking ML status:', error);
    } finally {
      setLoading(false);
    }
  };

  const testComplianceAnalysis = async () => {
    if (!documentText.trim()) {
      alert('Please enter some document text');
      return;
    }

    try {
      setLoading(true);
      const response = await mlAPI.analyzeCompliance(documentText, selectedCountry);
      setTestResult(response.data);
    } catch (error) {
      console.error('Error testing compliance analysis:', error);
      alert('Error testing compliance analysis');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: boolean) => {
    return status ? 'text-green-600' : 'text-red-600';
  };

  const getStatusIcon = (status: boolean) => {
    return status ? '✅' : '❌';
  };

  return (
    <div className={`p-6 bg-white rounded-lg shadow-md ${className}`}>
      <h2 className="text-2xl font-bold mb-4">ML Integration Test</h2>
      
      {/* ML Status */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-3">ML Model Status</h3>
        {loading ? (
          <p>Loading ML status...</p>
        ) : mlStatus ? (
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className={`flex items-center ${getStatusColor(mlStatus.ml_models_available)}`}>
              <span className="mr-2">{getStatusIcon(mlStatus.ml_models_available)}</span>
              ML Models Available
            </div>
            <div className={`flex items-center ${getStatusColor(mlStatus.compliance_checker)}`}>
              <span className="mr-2">{getStatusIcon(mlStatus.compliance_checker)}</span>
              Compliance Checker
            </div>
            <div className={`flex items-center ${getStatusColor(mlStatus.policy_comparator)}`}>
              <span className="mr-2">{getStatusIcon(mlStatus.policy_comparator)}</span>
              Policy Comparator
            </div>
            <div className={`flex items-center ${getStatusColor(mlStatus.principle_assessor)}`}>
              <span className="mr-2">{getStatusIcon(mlStatus.principle_assessor)}</span>
              Principle Assessor
            </div>
            <div className={`flex items-center ${getStatusColor(mlStatus.together_api_key)}`}>
              <span className="mr-2">{getStatusIcon(mlStatus.together_api_key)}</span>
              Together AI Key
            </div>
          </div>
        ) : (
          <p className="text-red-600">Failed to load ML status</p>
        )}
      </div>

      {/* Test Compliance Analysis */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-3">Test Compliance Analysis</h3>
        
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Country:</label>
          <select
            value={selectedCountry}
            onChange={(e) => setSelectedCountry(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md"
          >
            {countries.map(country => (
              <option key={country} value={country}>{country}</option>
            ))}
          </select>
        </div>

        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Document Text:</label>
          <textarea
            value={documentText}
            onChange={(e) => setDocumentText(e.target.value)}
            placeholder="Enter document text to analyze..."
            className="w-full p-2 border border-gray-300 rounded-md h-32"
          />
        </div>

        <button
          onClick={testComplianceAnalysis}
          disabled={loading}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? 'Testing...' : 'Test Compliance Analysis'}
        </button>
      </div>

      {/* Test Results */}
      {testResult && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-3">Test Results</h3>
          <div className="bg-gray-50 p-4 rounded-md">
            <div className="mb-2">
              <strong>Document ID:</strong> {testResult.doc_id}
            </div>
            <div className="mb-2">
              <strong>Country:</strong> {testResult.country}
            </div>
            <div className="mb-2">
              <strong>Analysis Type:</strong> {testResult.analysis_type}
            </div>
            <div className="mb-2">
              <strong>Overall Score:</strong> {testResult.results.overall_score}/100
            </div>
            <div className="mb-2">
              <strong>Major Gaps:</strong>
              <ul className="list-disc list-inside ml-4">
                {testResult.results.major_gaps.map((gap: string, index: number) => (
                  <li key={index}>{gap}</li>
                ))}
              </ul>
            </div>
            <div className="mb-2">
              <strong>Excellencies:</strong>
              <ul className="list-disc list-inside ml-4">
                {testResult.results.excellencies.map((excellency: string, index: number) => (
                  <li key={index}>{excellency}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Refresh Button */}
      <div>
        <button
          onClick={checkMLStatus}
          disabled={loading}
          className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 disabled:opacity-50"
        >
          Refresh ML Status
        </button>
      </div>
    </div>
  );
};

export default MLTestWidget; 
