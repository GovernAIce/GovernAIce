import React, { useState } from 'react';
import UploadProjectWidget from './UploadProjectWidget';
import ExplorePolicyWidget from './ExplorePolicyWidget';
import ExcellenciesMajorGapsWidget from './ExcellenciesMajorGapsWidget';
import ChatWithMeWidget from './ChatWithMeWidget';
import RelevantPolicyWidget from './RelevantPolicyWidget';
import OverallScoreWidget from './OverallScoreWidget';
import { useCountryContext } from '../contexts/CountryContext';
import RadarChartComponent from './RadarChartComponent'
import RegulatoryPolicy from './RegulatoryPolicy'

interface AnalysisResult {
  doc_id: string;
  filename: string;
  overall_score: number;
  insights: any[];
  policies: any[];
  risk_assessments: any;
  total_count: number;
  countries_searched: string[];
  domain: string;
  search_query: string;
  analysisType: string;
  uploadedFile: File;
  relevant_policies: any[];
}

const PolicyAnalysis: React.FC = () => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult | null>(null);
  const { selectedCountries } = useCountryContext();

  const handleAnalysisComplete = (results: AnalysisResult) => {
    setUploadedFile(results.uploadedFile || null);
    setAnalysisResults(results);
  };

  const handleFileUpload = (file: File) => {
    console.log('File uploaded:', file.name);
    setUploadedFile(file);
  };

  return (
    <div className="flex-1 h-full overflow-hidden">
      <div className="grid grid-cols-3 gap-2 gap-y-2 h-full overflow-hidden p-2">
        {/* Row 1 */}
        <div className="overflow-hidden">
          <UploadProjectWidget 
            onFileUpload={handleFileUpload} 
            onAnalysisComplete={handleAnalysisComplete} 
          />
        </div>
        <div className="overflow-hidden">
          <ExplorePolicyWidget />
        </div>
        <div className="overflow-hidden">
          <RelevantPolicyWidget
            uploadedFile={uploadedFile}
            policies={analysisResults?.relevant_policies || analysisResults?.policies || []}
          />
        </div>

        {/* Row 2 */}
        <div className="overflow-hidden">
          <OverallScoreWidget
            overallScore={analysisResults?.overall_score}
            uploadedFile={uploadedFile}
            analysisComplete={!!analysisResults}
          />
        </div>
        <div className="overflow-hidden">
          <RegulatoryPolicy 
            policies={analysisResults?.relevant_policies || analysisResults?.policies || []}
            uploadedFile={uploadedFile}
          />
        </div>
        <div className="row-span-2 overflow-hidden">
          <ChatWithMeWidget />
        </div>

        {/* Row 3 - Widget spanning 2 columns */}
        <div className="col-span-2 overflow-hidden">
          <ExcellenciesMajorGapsWidget 
            uploadedFile={uploadedFile}
            selectedCountries={selectedCountries}
            insights={analysisResults?.insights?.map(i => ({
              policy: i.policy,
              excellent_points: i.excellent_points || [],
              major_gaps: i.major_gaps || []
            })) || []}
          />
        </div>
      </div>
    </div>
  );
};

export default PolicyAnalysis; 
