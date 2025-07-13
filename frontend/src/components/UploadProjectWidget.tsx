import React, { useState, useRef } from 'react';
import Card from './Card';
import Button from './Button';
import { uploadAPI } from '../api';
import { useCountryContext } from '../contexts/CountryContext';

interface UploadProjectWidgetProps {
  onAnalysisComplete?: (results: any) => void;
}

const UploadProjectWidget: React.FC<UploadProjectWidgetProps> = ({ onAnalysisComplete }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [analysisType, setAnalysisType] = useState<'general' | 'regulatory'>('general');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { selectedCountries, hasCountries } = useCountryContext();

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Validate file type
      const allowedTypes = ['application/pdf', 'text/plain', 'application/msword', 
                           'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
      if (!allowedTypes.includes(file.type)) {
        setError('Please select a valid file type (PDF, DOC, DOCX, or TXT)');
        return;
      }
      
      // Validate file size (10MB limit)
      if (file.size > 10 * 1024 * 1024) {
        setError('File size must be less than 10MB');
        return;
      }
      
      setSelectedFile(file);
      setError(null);
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
      const allowedTypes = ['application/pdf', 'text/plain', 'application/msword', 
                           'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
      if (!allowedTypes.includes(file.type)) {
        setError('Please select a valid file type (PDF, DOC, DOCX, or TXT)');
        return;
      }
      
      if (file.size > 10 * 1024 * 1024) {
        setError('File size must be less than 10MB');
        return;
      }
      
      setSelectedFile(file);
      setError(null);
    }
  };

  const simulateProgress = () => {
    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 15 + 5; // Random increment between 5-20%
      if (progress >= 90) {
        progress = 90; // Stop at 90% until actual completion
        clearInterval(interval);
      }
      setUploadProgress(Math.min(progress, 90));
    }, 200);
    return interval;
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file to upload');
      return;
    }

    if (analysisType === 'general' && !hasCountries) {
      setError('Please select at least one country for analysis in the Explore Policy section');
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);
    setError(null);

    // Start progress simulation
    const progressInterval = simulateProgress();

    try {
      let response;
      
      if (analysisType === 'general') {
        response = await uploadAPI.uploadAndAnalyze(selectedFile, selectedCountries);
      } else {
        response = await uploadAPI.uploadRegulatoryCompliance(selectedFile);
      }

      // Complete progress
      clearInterval(progressInterval);
      setUploadProgress(100);
      
      // Pass results to parent component
      if (onAnalysisComplete) {
        onAnalysisComplete({
          ...response.data,
          analysisType
        });
      }
      
      // Reset form after successful upload
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      
    } catch (err: any) {
      clearInterval(progressInterval);
      setError(err.response?.data?.error || 'Upload failed. Please try again.');
    } finally {
      setIsUploading(false);
      // Keep progress at 100% for a moment before resetting
      setTimeout(() => setUploadProgress(0), 1000);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  return (
    <Card className="custom-border relative p-2 h-auto min-h-[280px]">
      <img
        src="/icons/info.svg"
        alt="Info"
        className="absolute top-1 right-1 w-3 h-3 cursor-pointer"
      />
      
      <div className="flex flex-col h-full gap-2">
        <h3 className="text-lg text-[#1975d4] font-bold">Upload Project</h3>
        
        {/* Analysis Type Selection */}
        <div className="flex gap-2">
          <label className="flex items-center gap-1 text-xs">
            <input
              type="radio"
              name="analysisType"
              value="general"
              checked={analysisType === 'general'}
              onChange={(e) => setAnalysisType(e.target.value as 'general' | 'regulatory')}
              className="text-blue-600"
            />
            General Compliance
          </label>
          <label className="flex items-center gap-1 text-xs">
            <input
              type="radio"
              name="analysisType"
              value="regulatory"
              checked={analysisType === 'regulatory'}
              onChange={(e) => setAnalysisType(e.target.value as 'general' | 'regulatory')}
              className="text-blue-600"
            />
            Regulatory (OECD/NIST/EU)
          </label>
        </div>

        {/* Country Status for General Analysis */}
        {analysisType === 'general' && (
          <div className={`text-xs p-2 rounded ${
            hasCountries 
              ? 'bg-green-50 text-green-700' 
              : 'bg-yellow-50 text-yellow-700'
          }`}>
            {hasCountries 
              ? `✅ Countries selected: ${selectedCountries.join(', ')}`
              : '⚠️ Please select countries in the Explore Policy section'
            }
          </div>
        )}

        {/* File Upload Area */}
        <div
          className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors ${
            selectedFile 
              ? 'border-green-500 bg-green-50' 
              : 'border-gray-300 hover:border-blue-400'
          }`}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={triggerFileInput}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.doc,.docx,.txt"
            onChange={handleFileSelect}
            className="hidden"
          />
          
          <div className="flex flex-col items-center gap-2">
            <img
              src="/icons/Upload.svg"
              alt="Upload Icon"
              className="w-8 h-8"
            />
            
            {selectedFile ? (
              <div className="text-sm">
                <p className="font-medium text-green-600">{selectedFile.name}</p>
                <p className="text-xs text-gray-500">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            ) : (
              <div className="text-sm">
                <p className="font-medium">Drop your file here or click to browse</p>
                <p className="text-xs text-gray-500">
                  Supports PDF, DOC, DOCX, TXT (max 10MB)
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="text-red-500 text-xs p-2 bg-red-50 rounded">
            {error}
          </div>
        )}

        {/* Upload Progress */}
        {isUploading && (
          <div className="space-y-2">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
            <p className="text-xs text-gray-600 text-center">
              {uploadProgress < 100 ? 'Analyzing document...' : 'Analysis complete!'}
            </p>
          </div>
        )}

        {/* Upload Button */}
        <Button
          className="w-full text-white text-xs mt-auto"
          onClick={handleUpload}
          disabled={!selectedFile || (analysisType === 'general' && !hasCountries) || isUploading}
        >
          {isUploading ? 'Analyzing...' : 'Upload & Analyze'}
        </Button>
      </div>
    </Card>
  );
};

export default UploadProjectWidget; 
