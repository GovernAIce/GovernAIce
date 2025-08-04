import { api } from "./index";
import { AxiosResponse } from "axios";

// ============================================================================
// ITEM API TYPES
// ============================================================================

export interface Item {
  name: string;
  id?: string;
  // Add more fields as needed
}

// ============================================================================
// ITEM MANAGEMENT API FUNCTIONS
// ============================================================================

/**
 * Fetch all items from the backend
 * @returns Promise with items list
 */
export function fetchItems(): Promise<AxiosResponse<Item[]>> {
  return api.get("/items");
}

/**
 * Create a new item
 * @param item - The item to create
 * @returns Promise with creation result
 */
export function postItem(item: Item): Promise<AxiosResponse<any>> {
  return api.post("/items", item);
}

/**
 * Update an existing item
 * @param id - Item ID
 * @param updates - Partial item updates
 * @returns Promise with update result
 */
export function updateItem(id: string, updates: Partial<Item>): Promise<AxiosResponse<any>> {
  return api.patch(`/items/${id}`, updates);
}

/**
 * Delete an item
 * @param id - Item ID
 * @returns Promise with deletion result
 */
export function deleteItem(id: string): Promise<AxiosResponse<any>> {
  return api.delete(`/items/${id}`);
}
