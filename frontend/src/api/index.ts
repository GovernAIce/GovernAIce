/// <reference types="vite/client" />

import axios from "axios";

// API Configuration
export const api = axios.create({
  baseURL: "http://localhost:5002", // Flask backend URL
  // you can add headers/interceptors here as needed
});

// ============================================================================
// UPLOAD & ANALYSIS APIs
// ============================================================================

export const uploadAPI = {
  /**
   * Upload file for comprehensive compliance analysis
   * Includes policy fetching, risk assessment, and compliance scoring
   */
  upload_analyze_policies: (file: File, countries: string[], domain?: string, searchQuery?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('countries', JSON.stringify(countries));
    formData.append('domain', domain || '');
    formData.append('search', searchQuery || '');
    return api.post('/api/upload-analyze-policies/', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },

  /**
   * Upload product info text for analysis
   */
  uploadProductInfo: (file: File, countries: string[], policies?: string[]) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('countries', JSON.stringify(countries));
    return api.post('/product-info-upload/', {
      countries: countries,
      policies: policies || []
    });
  },
}
// ============================================================================
// POLICY MANAGEMENT APIs
// ============================================================================

export const policyAPI = {
  /**
 * Get all available policies for a country
 */
  getPoliciesByCountry: (country: string) => {
    return api.get(`/metadata/policies/?country=${encodeURIComponent(country)}`);
  },

  /**
   * View full policy document by title and country
   */
  view_policy_document: (title: string, country: string) => {
    return api.get(`/view-policy/?title=${encodeURIComponent(title)}&country=${encodeURIComponent(country)}`);
  },
};

// ============================================================================
// METADATA APIs
// ============================================================================

export const metadataAPI = {
  /**
   * Get all available countries
   */
  getCountries: () => {
    return api.get('/metadata/country/');
  },

  /**
   * Get available domains
   */
  getDomains: () => {
    return api.get('/metadata/domains/');
  },

  /**
   * Get available policies
   */
  getPolicies: () => {
    return api.get('/metadata/policies/');
  }
};

// ============================================================================
// DOCUMENT MANAGEMENT APIs
// ============================================================================

export const documentAPI = {
  /**
   * Get all documents
   */
  getDocuments: () => {
    return api.get('/documents/');
  },

  /**
   * Get specific document by ID
   */
  getDocument: (docId: string) => {
    return api.get(`/documents/${docId}/`);
  },

  /**
   * Search documents by query
   */
  searchDocuments: (query: string) => {
    return api.get(`/documents/search/?query=${encodeURIComponent(query)}`);
  }
};

// ============================================================================
// DASHBOARD WIDGET APIs
// ============================================================================

export const dashboardAPI = {
  /**
   * Get OECD scores for dashboard widget
   */
  getOECDScores: () => {
    return api.get('/api/oecd-scores');
  },

  /**
   * Get NIST lifecycle scores for dashboard widget
   */
  getNISTScores: () => {
    return api.get('/api/nist-lifecycle-scores');
  },

  /**
   * Get EU risk level data for dashboard widget
   */
  getEURiskLevel: () => {
    return api.get('/api/eu-risk-level');
  },

  /**
   * Get relevant policies chart data
   */
  getRelevantPolicies: () => {
    return api.get('/api/relevant-policies');
  },

  /**
   * Get radar chart data
   */
  getRadarData: () => {
    return api.get('/api/radar-data');
  }
};


