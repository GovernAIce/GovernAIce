import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5002';

// ML API functions
export const mlAPI = {
  // Get ML model status
  getMLStatus: () => {
    return axios.get(`${API_BASE_URL}/ml/status/`);
  },

  // Advanced compliance analysis using ML models
  analyzeCompliance: (documentText: string, country: string, inputType: string = 'text') => {
    return axios.post(`${API_BASE_URL}/ml/compliance-analysis/`, {
      document_text: documentText,
      country: country,
      input_type: inputType
    });
  },

  // Policy comparison using ML models
  comparePolicies: (userDocument: string, referenceFile: string, country?: string) => {
    return axios.post(`${API_BASE_URL}/ml/policy-comparison/`, {
      user_document: userDocument,
      reference_file: referenceFile,
      country: country
    });
  },

  // Principle assessment using ML models
  assessPrinciples: (documentPath: string, embeddingsFile: string) => {
    return axios.post(`${API_BASE_URL}/ml/principle-assessment/`, {
      document_path: documentPath,
      embeddings_file: embeddingsFile
    });
  },

  // Get ML analysis results by document ID
  getMLAnalysis: (docId: string) => {
    return axios.get(`${API_BASE_URL}/ml/analysis/${docId}/`);
  }
};

// Types for ML API responses
export interface MLStatus {
  ml_models_available: boolean;
  compliance_checker: boolean;
  policy_comparator: boolean;
  principle_assessor: boolean;
  together_api_key: boolean;
}

export interface ComplianceAnalysisResult {
  doc_id: string;
  country: string;
  analysis_type: string;
  results: {
    overall_score: number;
    major_gaps: string[];
    excellencies: string[];
    improvement_strategy: string[];
    detailed_analysis: any;
    referenced_policies: any[];
    analysis_type: string;
  };
}

export interface PolicyComparisonResult {
  doc_id: string;
  reference_file: string;
  country?: string;
  analysis_type: string;
  results: {
    overall_score: number;
    risk_level: string;
    detailed_scores: any;
    alignment_analysis: any;
    recommendations: string[];
    gaps_identified: string[];
    analysis_type: string;
  };
}

export interface PrincipleAssessmentResult {
  doc_id: string;
  document_path: string;
  embeddings_file: string;
  analysis_type: string;
  results: {
    oecd_analysis: any;
    nist_analysis: any;
    eu_analysis: any;
    cross_framework_insights: any;
    overall_risk_assessment: string;
    strategic_recommendations: string[];
    implementation_roadmap: any;
    analysis_type: string;
  };
} 
