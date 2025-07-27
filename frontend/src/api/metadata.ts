import { api } from "./index";
import { AxiosResponse } from "axios";

export interface CountriesResponse {
  countries: string[];
}

export function fetchCountries(): Promise<AxiosResponse<CountriesResponse>> {
  return api.get("/metadata/country/");
}
