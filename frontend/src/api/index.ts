/// <reference types="vite/client" />


import axios from "axios";

export const api = axios.create({
  baseURL: "http://localhost:5001", // Docker backend URL
  // you can add headers/interceptors here as needed
});

// Upload API functions
export const uploadAPI = {
  // Upload file for general compliance analysis

  uploadAndAnalyze: (file: File, countries: string[], policies?: string[]) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('countries', JSON.stringify(countries));
    if (policies) {
      formData.append('policies', JSON.stringify(policies));
    }
    return api.post('/upload-and-analyze/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  // Upload product info text for analysis
  uploadProductInfo: (file: File, countries: string[], policies?: string[]) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('countries', JSON.stringify(countries));
    return api.post('/product-info-upload/', {
      countries: countries,
      policies: policies || []
    });
  },

  // Upload file for regulatory compliance analysis (OECD, NIST, EU)
  uploadRegulatoryCompliance: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/upload-regulatory-compliance/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  // Upload product info for regulatory compliance analysis
  analyzeRegulatoryProductInfo: (productInfo: string) => {
    return api.post('/analyze-regulatory-product-info/', {
      product_info: productInfo
    });
  }
};

// Policy API functions
export const policyAPI = {
  // Fetch relevant policies based on countries and search criteria
  use_case_one: (countries: string[], domain?: string, searchQuery?: string) => {
    return api.post('/api/policies/relevant', {
      countries,
      domain,
      search: searchQuery
    });
  },

  // Get all available policies for a country
  getPoliciesByCountry: (country: string) => {
    return api.get(`/metadata/policies/?country=${encodeURIComponent(country)}`);
  }

  // getUseCaseOne: (countries: string[], domain?: string, searchQuery?: string) => {
  //   return api.post('/api/policies/use-case-one', {
  //     countries,
  //     domain,
  //     search: searchQuery
  //   });
  // }
};
