// src/components/ItemList.jsx
import { useEffect, useState } from "react";
import { fetchItems } from "../api/items";

export default function ItemList() {
  const [items, setItems] = useState([]);
  useEffect(() => {
    fetchItems().then(res => setItems(res.data));
  }, []);
  return (
    <ul>
      {items.map((it, i) => (
        <li key={i}>{it.name}</li>
      ))}
    </ul>
  );
}
