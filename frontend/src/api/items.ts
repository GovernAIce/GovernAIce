import { api } from "./index";
import { AxiosResponse } from "axios";

export interface Item {
  // Define your item fields here, e.g.:
  name: string;
  // id?: string; // add more fields as needed
}

export function fetchItems(): Promise<AxiosResponse<Item[]>> {
  return api.get("/items");
}

export function postItem(item: Item): Promise<AxiosResponse<any>> {
  return api.post("/items", item);
}

export function updateItem(id: string, updates: Partial<Item>): Promise<AxiosResponse<any>> {
  return api.patch(`/items/${id}`, updates);
}

export function deleteItem(id: string): Promise<AxiosResponse<any>> {
  return api.delete(`/items/${id}`);
}

// Analyze document for excellencies and major gaps
export async function analyzeDocument(file: File, countries: string[]): Promise<any> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('countries', JSON.stringify(countries));
  const response = await api.post('/upload-and-analyze/', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}
