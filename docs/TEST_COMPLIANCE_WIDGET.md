# Testing ComplianceAnalysisWidget Behavior

## Test Scenarios

### 1. Initial State (No Countries, No File)
**Expected Behavior:**
- Widget shows warning message: "Please select countries and upload a document to view relevant policies."
- No API calls made
- No policies displayed

### 2. Countries Selected, No File
**Expected Behavior:**
- Widget shows info message: "Please upload a document to view relevant policies for your selected countries."
- No API calls made
- No policies displayed

### 3. File Uploaded, No Countries
**Expected Behavior:**
- Widget shows warning message: "Please select countries in the Explore Policy section to view relevant policies."
- No API calls made
- No policies displayed

### 4. Both Countries and File Selected
**Expected Behavior:**
- Widget makes API call to fetch relevant policies
- Shows loading spinner during API call
- Displays policies with country badges
- Shows policy count in header

### 5. File Changed
**Expected Behavior:**
- Previous analysis results cleared
- New API call made with updated file context
- Policies updated based on new file content

### 6. Countries Changed
**Expected Behavior:**
- New API call made with updated countries
- Policies filtered based on new country selection
- Country badges updated

## Manual Testing Steps

1. **Start the application:**
   ```bash
   # Backend
   cd backend && python app.py
   
   # Frontend  
   cd frontend && npm run dev
   ```

2. **Test Initial State:**
   - Open the application
   - Navigate to Policy Analysis
   - Check ComplianceAnalysisWidget shows appropriate message

3. **Test Country Selection:**
   - Select countries in ExplorePolicyWidget
   - Verify ComplianceAnalysisWidget updates message
   - No policies should be fetched yet

4. **Test File Upload:**
   - Upload a document in UploadProjectWidget
   - Verify ComplianceAnalysisWidget starts fetching policies
   - Check loading state and results

5. **Test Combined State:**
   - Have both countries selected and file uploaded
   - Verify policies are displayed with country badges
   - Check that policy count shows in header

6. **Test Dynamic Updates:**
   - Change country selection
   - Upload a different file
   - Verify widget responds appropriately

## API Testing

Test the backend endpoint directly:

```bash
curl -X POST http://localhost:5001/api/policies/relevant \
  -H "Content-Type: application/json" \
  -d '{
    "countries": ["USA", "EU"],
    "domain": "AI", 
    "search": "privacy"
  }'
```

Expected response:
```json
{
  "countries_searched": ["USA", "EU"],
  "domain": "AI",
  "policies": [
    {
      "country": "USA",
      "domain": "AI", 
      "source": "https://example.com/usa",
      "text": "Policy document for USA: California Consumer Privacy Act (CCPA)",
      "title": "California Consumer Privacy Act (CCPA)"
    }
  ],
  "search_query": "privacy",
  "total_count": 1
}
```

## Success Criteria

✅ **Widget waits for both conditions** (countries + file) before fetching policies  
✅ **Clear user guidance** with appropriate status messages  
✅ **No unnecessary API calls** when conditions aren't met  
✅ **Dynamic updates** when conditions change  
✅ **Proper loading states** during API calls  
✅ **Error handling** for failed requests  
✅ **Country badges** display correctly  
✅ **File context** influences policy relevance 
