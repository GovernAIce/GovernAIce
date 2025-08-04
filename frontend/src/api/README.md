# API Organization

This folder contains all API-related functions for the GovernAIce frontend application.

## File Structure

```
api/
├── index.ts          # Main API exports and configuration
├── ml.ts            # Machine Learning API functions
├── metadata.ts      # Metadata API (countries, domains, policies)
├── items.ts         # Item management and document analysis
└── README.md        # This documentation
```

## API Categories

### 1. Main API (`index.ts`)
Contains the core API configuration and organized function groups:

- **uploadAPI**: File upload and analysis functions
- **policyAPI**: Policy management and retrieval
- **metadataAPI**: Metadata operations (countries, domains, policies)
- **documentAPI**: Document management operations
- **dashboardAPI**: Dashboard widget data endpoints

### 2. Machine Learning API (`ml.ts`)
Specialized ML model integration functions:

- Model status checking
- Compliance analysis using ML models
- Policy comparison with ML
- Principle assessment

### 3. Metadata API (`metadata.ts`)
Metadata retrieval functions with TypeScript interfaces:

- Country fetching
- Domain fetching
- Policy fetching

### 4. Items API (`items.ts`)
Item management and document analysis:

- CRUD operations for items
- Document analysis functions
- TypeScript interfaces for responses

## Usage Examples

### Upload and Analyze
```typescript
import { uploadAPI } from '../api';

const result = await uploadAPI.upload_analyze_policies(
  file, 
  ['EU', 'Canada'], 
  'AI', 
  'data protection'
);
```

### Fetch Countries
```typescript
import { fetchCountries } from '../api/metadata';

const response = await fetchCountries();
const countries = response.data.countries;
```

### ML Analysis
```typescript
import { mlAPI } from '../api/ml';

const status = await mlAPI.getMLStatus();
const analysis = await mlAPI.analyzeCompliance(text, 'EU');
```

## Best Practices

1. **Use the main exports**: Import from `../api` for most functions
2. **Use specific imports**: Import from specific files for specialized functions
3. **Check deprecation warnings**: Some functions are marked as deprecated
4. **Handle errors**: All API calls should be wrapped in try-catch blocks
5. **Type safety**: Use the provided TypeScript interfaces for better type safety

## Deprecated Functions

The following functions are deprecated and should be replaced:

- `uploadAPI.uploadAndAnalyze` → Use `uploadAPI.upload_analyze_policies`
- `uploadAPI.uploadRegulatoryCompliance` → Use `uploadAPI.upload_analyze_policies`
- `uploadAPI.analyzeRegulatoryProductInfo` → Use `uploadAPI.upload_analyze_policies`

## Adding New APIs

When adding new API functions:

1. Add them to the appropriate category in `index.ts`
2. Create TypeScript interfaces for request/response types
3. Add JSDoc comments for documentation
4. Update this README if needed 
