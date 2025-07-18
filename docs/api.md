# GovernAIce API Documentation

## Base URL
```
http://localhost:5000/
```

---

## Authentication (Future TODO)
- All endpoints will require authentication in the future (e.g., JWT or session-based).
- For now, endpoints are public for prototype/development/testing.

---

## Metadata

### GET `/metadata/countries/`
- **Description:** List all available countries.
- **Response:**
  ```json
  { "countries": ["USA", "UK", ...] }
  ```

### GET `/metadata/domains/`
- **Description:** List all available domains.
- **Response:**
  ```json
  { "domains": ["Healthcare", "Finance", ...] }
  ```

### GET `/metadata/policies/?country=USA`
- **Description:** List all policies for a given country.
- **Response:**
  ```json
  { "policies": ["GDPR", "CCPA", ...] }
  ```

---

## Documents

### POST `/upload-and-analyze/`
- **Description:** Upload a file and analyze for compliance.
- **Form Data:**
  - `file`: File to upload
  - `countries`: JSON array of country names
  - `policies`: (optional) JSON array of policy names
- **Response:**
  ```json
  { "doc_id": "...", "filename": "...", "insights": [...] }
  ```

### POST `/product-info-upload/`
- **Description:** Analyze product info text for compliance.
- **Body:**
  ```json
  { "product_info": "...", "countries": ["USA"], "policies": [] }
  ```
- **Response:**
  ```json
  { "doc_id": "...", "filename": "...", "insights": [...] }
  ```

### GET `/documents/`
- **Description:** List all documents.
- **Response:** Array of document objects.

### GET `/documents/<doc_id>/`
- **Description:** Get details for a specific document.
- **Response:** Document object.

### GET `/documents/search/?query=...`
- **Description:** Search documents by content.
- **Response:** Array of document objects.

---

## Reports

### GET `/reports/`
- **Description:** List all reports.
- **Response:** Array of report objects.

### POST `/reports/`
- **Description:** Create a new report.
- **Body:**
  ```json
  { "title": "...", "content": "..." }
  ```
- **Response:**
  ```json
  { "report_id": "..." }
  ```

### GET `/reports/<report_id>/`
- **Description:** Get a report by ID.
- **Response:** Report object.

### PUT `/reports/<report_id>/`
- **Description:** Update a report.
- **Body:**
  ```json
  { "title": "...", "content": "..." }
  ```
- **Response:** Updated report object.

### DELETE `/reports/<report_id>/`
- **Description:** Delete a report.
- **Response:**
  ```json
  { "message": "Deleted" }
  ```

---

## Project Folders

### GET `/folders/`
- **Description:** List all project folders.
- **Response:** Array of folder objects.

### POST `/folders/`
- **Description:** Create a new folder.
- **Body:**
  ```json
  { "name": "..." }
  ```
- **Response:**
  ```json
  { "folder_id": "..." }
  ```

### GET `/folders/<folder_id>/`
- **Description:** Get a folder by ID.
- **Response:** Folder object.

### PUT `/folders/<folder_id>/`
- **Description:** Update a folder.
- **Body:**
  ```json
  { "name": "..." }
  ```
- **Response:** Updated folder object.

### DELETE `/folders/<folder_id>/`
- **Description:** Delete a folder.
- **Response:**
  ```json
  { "message": "Deleted" }
  ```

---

## Team Members

### GET `/team/`
- **Description:** List all team members.
- **Response:** Array of member objects.

### POST `/team/`
- **Description:** Add a new team member.
- **Body:**
  ```json
  { "name": "...", "email": "..." }
  ```
- **Response:**
  ```json
  { "member_id": "..." }
  ```

### GET `/team/<member_id>/`
- **Description:** Get a team member by ID.
- **Response:** Member object.

### PUT `/team/<member_id>/`
- **Description:** Update a team member.
- **Body:**
  ```json
  { "name": "...", "email": "..." }
  ```
- **Response:** Updated member object.

### DELETE `/team/<member_id>/`
- **Description:** Delete a team member.
- **Response:**
  ```json
  { "message": "Deleted" }
  ```

---

## Chatbot

### POST `/chat/`
- **Description:** Send a message to the chatbot.
- **Body:**
  ```json
  { "message": "How do I comply with GDPR?" }
  ```
- **Response:**
  ```json
  { "reply": "..." }
  ```

---

## Notes
- All endpoints will require authentication in the future (see TODOs in backend code).
- For now, endpoints are public for prototype/development/testing. 

