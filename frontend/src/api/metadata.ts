import { api } from "./index";
import { AxiosResponse } from "axios";

// ============================================================================
// METADATA API TYPES
// ============================================================================

export interface CountriesResponse {
  countries: string[];
}

export interface DomainsResponse {
  domains: string[];
}

export interface PoliciesResponse {
  policies: string[];
}

// ============================================================================
// METADATA API FUNCTIONS
// ============================================================================

/**
 * Fetch all available countries for policy analysis
 * @returns Promise with countries list
 */
export function fetchCountries(): Promise<AxiosResponse<CountriesResponse>> {
  return api.get('/metadata/country/');
}

/**
 * Fetch all available domains for policy analysis
 * @returns Promise with domains list
 */
export function fetchDomains(): Promise<AxiosResponse<DomainsResponse>> {
  return api.get('/metadata/domains/');
}

/**
 * Fetch all available policies
 * @returns Promise with policies list
 */
export function fetchPolicies(): Promise<AxiosResponse<PoliciesResponse>> {
  return api.get('/metadata/policies/');
}
