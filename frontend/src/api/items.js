import { api } from "./index";

export function fetchItems() {
  return api.get("/items");
}

export function postItem(item) {
  return api.post("/items", item);
}

export function updateItem(id, updates) {
  return api.patch(`/items/${id}`, updates);
}

export function deleteItem(id) {
  return api.delete(`/items/${id}`);
}
