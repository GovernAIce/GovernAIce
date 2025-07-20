# GovernAIce Backend API
# ----------------------
# This Flask app provides endpoints for document compliance analysis, metadata, reports, folders, team management, and chatbot features.
# MongoDB is used for data storage. The backend is model-agnostic and can use OpenAI (default), Gemini, or future Llama/RAG models for document analysis.
# Model selection is handled via the 'model' argument in analyze_document and llm_utils.py.
# TODO: Add authentication to protect endpoints in the future.

from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson import json_util
import uuid
import json
from datetime import datetime
import PyPDF2
from io import BytesIO
import os
import re
import google.generativeai as genai
import logging
from sentence_transformers import SentenceTransformer
from pymongo.server_api import ServerApi
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
# Import model-agnostic LLM interface
from llm_utils import analyze_policy


# Initialize Flask app and load environment variables
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
load_dotenv()

# Configure logging for debugging and error tracking
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# --- DATABASE CONNECTION ---
# Helper function to connect to MongoDB using the URI from environment variables
# Raises an exception if connection fails

def get_mongo_client():
    try:
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            raise Exception("MONGO_URI not set in environment variables")
        client = MongoClient(mongo_uri)
        client.server_info()  # Test connection
        logger.info("Successfully connected to MongoDB")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise Exception(f"MongoDB connection failed: {str(e)}")

# --- METADATA HELPERS ---
# Fetches available countries and their policies from the Training database
# Used for populating dropdowns in the frontend

# Simple in-memory cache for country policies
_country_policies_cache = None
_cache_timestamp = None
CACHE_DURATION = 300  # 5 minutes

def get_country_policies():
    global _country_policies_cache, _cache_timestamp
    
    # Check if cache is still valid
    if (_country_policies_cache is not None and 
        _cache_timestamp is not None and 
        (datetime.now() - _cache_timestamp).seconds < CACHE_DURATION):
        return _country_policies_cache
    
    mongo_client = get_mongo_client()
    try:
        mongo_db = mongo_client['Training']
        
        # Check if we need to populate sample data
        collections = mongo_db.list_collection_names()
        if not collections:
            # Populate with sample data for testing
            sample_countries = [
                "Canada", "UAE", "Taiwan", "Saudi Arabia", "Australia", 
                "Singapore", "South Korea", "Europe", "Brazil", "India", 
                "USA", "Japan", "UK", "China", "EU"
            ]
            
            for country in sample_countries:
                # Create a sample document for each country
                sample_doc = {
                    "title": f"Sample Policy for {country}",
                    "source": f"https://example.com/{country.lower().replace(' ', '-')}",
                    "text": f"This is a sample policy document for {country}."
                }
                mongo_db[country].insert_one(sample_doc)
            
            logger.info("Populated sample country data")
        
        policies = {}
        for country in mongo_db.list_collection_names():
            # Only fetch the 'title' field instead of entire documents
            country_policies = mongo_db[country].find({}, {'title': 1, '_id': 0})
            policies[country] = [doc.get('title', '') for doc in country_policies if doc.get('title')]
        
        # Update cache
        _country_policies_cache = policies
        _cache_timestamp = datetime.now()
        return policies
    finally:
        mongo_client.close()

def clear_country_policies_cache():
    """Clear the country policies cache (for debugging/testing)."""
    global _country_policies_cache, _cache_timestamp
    _country_policies_cache = None
    _cache_timestamp = None
    logger.info("Country policies cache cleared")

# --- PDF & LLM HELPERS ---
# Extracts text from uploaded PDF files

def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return f"Error extracting text: {str(e)}"

def sanitize_field_name(field_name):
    """
    Sanitize a field name for MongoDB by replacing invalid characters.
    MongoDB field names cannot contain: ., $, or be empty.
    """
    if not field_name:
        return "unnamed_field"
    
    # Replace problematic characters with underscores
    sanitized = field_name.replace('.', '_').replace('$', '_').replace(':', '_').replace('—', '_').replace('-', '_')
    sanitized = sanitized.replace(' ', '_').replace('/', '_').replace('\\', '_')
    
    # Remove any remaining special characters
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', sanitized)
    
    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = 'field_' + sanitized
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "unnamed_field"
    
    return sanitized

# Attempts to extract a JSON object from a string (e.g., LLM API response)
def extract_json_from_response(response_text):
    if not response_text or not response_text.strip():
        return None
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass
    json_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{.*\}',
    ]
    for pattern in json_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    return None

# Creates a fallback insight if analysis fails
def create_fallback_insight(policy, error_msg):
    return {
        'policy': policy,
        'compliance_score': 0,
        'policy_details': f'Unable to analyze: {error_msg}',
        'excellent_points': [],
        'major_gaps': [f'Analysis failed due to: {error_msg}']
    }

# --- MODEL-AGNOSTIC DOCUMENT ANALYSIS ---
# This function analyzes a document for compliance using the specified LLM model.
# Supported models: 'openai' (default), 'gemini', 'llama' (future).
# The model argument is passed to llm_utils.analyze_policy, which dispatches to the correct backend.

def analyze_document(text, countries, policies, doc_id, model="gemini"):
    """
    Analyze a document for compliance using the specified LLM model (default: openai).
    The model argument can be 'openai', 'gemini', or 'llama' (future).
    """
    if not text or not text.strip():
        logger.error("Empty or invalid text provided for analysis")
        return [create_fallback_insight(policy, "Empty document text") for policy in policies]
    insights = []
    # Generate prompts for each policy
    prompts = {}
    for country in countries:
        country_policies = policies if policies else get_country_policies().get(country, [])
        for policy in country_policies:
            prompts[policy] = f"""
            You are an expert in the {policy} from {country}. Analyze the following product description for compliance with this policy.
            Product Description: {text}
            IMPORTANT: Return ONLY valid JSON in the exact format below, with no additional text, comments, or formatting. The compliance_score must be an integer between 0 and 100:
            {{
                \"policy\": \"{policy}\",
                \"compliance_score\": integer,
                \"policy_details\": \"string describing key aspects of the policy relevant to the document\",
                \"excellent_points\": [\"string\", \"string\"],
                \"major_gaps\": [\"string\", \"string\"]
            }}
            """
    for policy in prompts:
        logger.info(f"Analyzing policy: {policy}")
        try:
            max_retries = 3
            response_text = None
            for attempt in range(max_retries):
                try:
                    logger.debug(f"Attempt {attempt + 1} for policy {policy}")
                    response_text = analyze_policy(prompts[policy], model=model)
                    if response_text:
                        logger.debug(f"Raw response for {policy}: {response_text[:200]}...")
                        break
                except Exception as e:
                    logger.error(f"API call failed on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise
                    continue
            if not response_text:
                raise Exception("No valid response received after retries")
            insight = extract_json_from_response(response_text)
            if insight is None:
                logger.error(f"Failed to extract JSON from response: {response_text}")
                insight = create_fallback_insight(policy, "Invalid JSON response format")
            else:
                required_fields = ['policy', 'compliance_score', 'policy_details', 'excellent_points', 'major_gaps']
                missing_fields = [field for field in required_fields if field not in insight]
                if missing_fields:
                    logger.warning(f"Missing fields in {policy} response: {missing_fields}")
                    insight = create_fallback_insight(policy, f"Missing required fields: {missing_fields}")
                elif not isinstance(insight['compliance_score'], int) or not (0 <= insight['compliance_score'] <= 100):
                    logger.warning(f"Invalid compliance_score in {policy} response: {insight.get('compliance_score')}")
                    insight = create_fallback_insight(policy, "Invalid compliance score")
            insights.append(insight)
        except Exception as e:
            logger.error(f"Analysis failed for {policy}: {str(e)}")
            insight = create_fallback_insight(policy, str(e))
            insights.append(insight)
    return insights

# --- METADATA ENDPOINTS ---
# These endpoints provide lists of countries, domains, and policies for populating dropdowns in the frontend.
# TODO: Add authentication in the future.

@app.route('/metadata/country/', methods=['GET'])
def get_countries():
    """Return a list of available countries for dropdowns."""
    # TODO: Add authentication (future)
    countries = list(get_country_policies().keys())
    return jsonify({'countries': countries}), 200

@app.route('/debug/cache/clear/', methods=['POST'])
def clear_cache():
    """Clear the country policies cache (debug endpoint)."""
    clear_country_policies_cache()
    return jsonify({'message': 'Cache cleared successfully'}), 200

@app.route('/debug/cache/status/', methods=['GET'])
def cache_status():
    """Get cache status (debug endpoint)."""
    global _country_policies_cache, _cache_timestamp
    if _country_policies_cache is None:
        return jsonify({'cached': False, 'timestamp': None}), 200
    else:
        age_seconds = (datetime.now() - _cache_timestamp).seconds if _cache_timestamp else None
        return jsonify({
            'cached': True, 
            'timestamp': _cache_timestamp.isoformat() if _cache_timestamp else None,
            'age_seconds': age_seconds,
            'cache_duration': CACHE_DURATION
        }), 200

@app.route('/metadata/domains/', methods=['GET'])
def get_domains():
    """Return a list of available domains for dropdowns."""
    # TODO: Add authentication (future)
    # TODO: Implement logic to fetch domains
    return jsonify({'domains': []}), 200

@app.route('/metadata/policies/', methods=['GET'])
def get_policies():
    """Return a list of policies for a given country (query param)."""
    # TODO: Add authentication (future)
    # TODO: Implement logic to fetch policies by country
    return jsonify({'policies': []}), 200

# @app.route('/api/policies/relevant', methods=['POST'])
def get_relevant_policies():
    """
    Get relevant policies based on selected countries and uploaded file content.
    Expects JSON body with 'countries' (list), 'domain' (optional), and 'search' (optional).
    Returns: Array of relevant policy objects (limited to 5 most relevant).
    """
    import os
    if os.environ.get('MOCK_POLICIES', '0') == '1':
        # Return mock data for frontend testing
        mock_policies = [
            {
                'title': 'Mock Privacy Policy',
                'source': 'https://example.com/mock-privacy',
                'text': 'This is a mock privacy policy for testing.',
                'country': 'USA',
                'domain': 'privacy'
            },
            {
                'title': 'Mock Data Protection Act',
                'source': 'https://example.com/mock-dpa',
                'text': 'This is a mock data protection act for testing.',
                'country': 'EU',
                'domain': 'data protection'
            },
            {
                'title': 'Mock AI Regulation',
                'source': 'https://example.com/mock-ai',
                'text': 'This is a mock AI regulation for testing.',
                'country': 'UK',
                'domain': 'AI'
            },
            {
                'title': 'Mock Consumer Rights Law',
                'source': 'https://example.com/mock-consumer',
                'text': 'This is a mock consumer rights law for testing.',
                'country': 'Canada',
                'domain': 'consumer rights'
            },
            {
                'title': 'Mock Cybersecurity Framework',
                'source': 'https://example.com/mock-cyber',
                'text': 'This is a mock cybersecurity framework for testing.',
                'country': 'Australia',
                'domain': 'cybersecurity'
            }
        ]
        return jsonify({
            'policies': mock_policies,
            'total_count': len(mock_policies),
            'countries_searched': ['MOCK'],
            'domain': 'mock',
            'search_query': 'mock',
            'max_results': 5
        }), 200
    try:
        data = request.get_json()
        if not data or 'countries' not in data:
            return jsonify({'error': 'Missing countries parameter'}), 400
        
        countries = data['countries']
        
        domain = data.get('domain', '')
        search_query = data.get('search', '')
        
        if not isinstance(countries, list):
            return jsonify({'error': 'Countries must be a list'}), 400
        
        # Get all policies for the selected countries
        country_policies = get_country_policies()
        relevant_policies = []
        
        for country in countries:
            if country in country_policies:
                policies = country_policies[country]
                for policy in policies:
                    # Calculate relevance score based on multiple factors
                    relevance_score = 0
                    
                    # Base score for country match
                    relevance_score += 10
                    
                    # Domain relevance
                    if domain and domain.lower() in policy.lower():
                        relevance_score += 5
                    
                    # Search query relevance
                    if search_query:
                        query_terms = search_query.lower().split()
                        for term in query_terms:
                            if term in policy.lower():
                                relevance_score += 3
                            if term in country.lower():
                                relevance_score += 2
                    
                    # Create a policy object with metadata and relevance score
                    policy_obj = {
                        'title': policy,
                        'source': f"https://example.com/{country.lower().replace(' ', '-')}",
                        'text': f"Policy document for {country}: {policy}",
                        'country': country,
                        'domain': domain or 'general',
                        'relevance_score': relevance_score
                    }
                    
                    # Always include policies for selected countries, but score them
                    relevant_policies.append(policy_obj)
        
        # Sort by relevance score (highest first) and limit to top 5
        relevant_policies.sort(key=lambda x: x['relevance_score'], reverse=True)
        relevant_policies = relevant_policies[:5]
        
        # Remove relevance_score from final response (internal use only)
        for policy in relevant_policies:
            policy.pop('relevance_score', None)
        
        return jsonify({
            'policies': relevant_policies,
            'total_count': len(relevant_policies),
            'countries_searched': countries,
            'domain': domain,
            'search_query': search_query,
            'max_results': 5
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching relevant policies: {str(e)}")
        return jsonify({'error': str(e)}), 500

# --- DOCUMENT ENDPOINTS ---
# These endpoints handle document upload, analysis, listing, and search.
# The analysis endpoints use the model-agnostic analyze_document function, which defaults to OpenAI but can be extended.

@app.route('/upload-and-analyze/', methods=['POST'])
def document_upload():
    """
    Upload a file and analyze it for compliance.
    Expects multipart form-data with 'file', 'countries' (JSON list), and optional 'policies' (JSON list).
    Returns: doc_id, filename, and analysis insights.
    Uses the default LLM model (OpenAI) for analysis.
    """
    if 'file' not in request.files or 'countries' not in request.form:
        return jsonify({'error': 'Missing file or countries'}), 400
    file = request.files['file']
    try:
        countries = json.loads(request.form['countries'])
        policies = json.loads(request.form.get('policies', '[]'))
        if not isinstance(countries, list) or not isinstance(policies, list):
            return jsonify({'error': 'Countries and policies must be lists'}), 400
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid countries or policies format'}), 400
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    # Validate countries and policies
    valid_countries = set(get_country_policies().keys())
    if not all(c in valid_countries for c in countries):
        return jsonify({'error': 'Invalid country specified'}), 400
    if policies:
        for policy in policies:
            if not any(policy in get_country_policies().get(c, []) for c in countries):
                return jsonify({'error': f'Policy {policy} not valid for specified countries'}), 400
    text = extract_text_from_pdf(file)
    doc_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    mongo_client = get_mongo_client()
    try:
        mongo_db = mongo_client['regulatory_mongo']
        document = {
            'doc_id': doc_id,
            'content': text,
            'compliance_scores': {}
        }
        mongo_db.documents.insert_one(document)
        insights = analyze_document(text, countries, policies or [p for c in countries for p in get_country_policies().get(c, [])], doc_id)
        # Update document with compliance details
        for insight in insights:
            policy = insight.get('policy')
            sanitized_policy_name = sanitize_field_name(policy)
            mongo_db.documents.update_one(
                {'doc_id': doc_id},
                {'$set': {
                    f'compliance_scores.{sanitized_policy_name}': {
                        'score': int(insight.get('compliance_score', 0)),
                        'excellent_points': insight.get('excellent_points', []),
                        'major_gaps': insight.get('major_gaps', [])
                    }
                }}
            )
        response_data = {
            'doc_id': doc_id,
            'filename': filename,
            'insights': insights
        }
        return jsonify(response_data), 201
    except Exception as e:
        logger.error(f"Error in document upload: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        mongo_client.close()

@app.route('/product-info-upload/', methods=['POST'])
def product_info_upload():
    """
    Analyze product info text for compliance.
    Expects JSON body with 'product_info', 'countries' (list), and optional 'policies' (list).
    Returns: doc_id, filename, and analysis insights.
    Uses the default LLM model (OpenAI) for analysis.
    """
    try:
        data = request.get_json()
        if not data or 'product_info' not in data or 'countries' not in data:
            return jsonify({'error': 'Missing product_info or countries'}), 400
        product_info = data['product_info']
        countries = data['countries']
        policies = data.get('policies', [])
        if not isinstance(countries, list) or not isinstance(policies, list) or len(product_info) < 10:
            return jsonify({'error': 'Invalid input: countries and policies must be lists, product_info must be at least 10 characters'}), 400
        # Validate countries and policies
        valid_countries = set(get_country_policies().keys())
        if not all(c in valid_countries for c in countries):
            return jsonify({'error': 'Invalid country specified'}), 400
        if policies:
            for policy in policies:
                if not any(policy in get_country_policies().get(c, []) for c in countries):
                    return jsonify({'error': f'Policy {policy} not valid for specified countries'}), 400
        doc_id = str(uuid.uuid4())
        doc_title = f"Product_Info_{doc_id[:8]}"
        mongo_client = get_mongo_client()
        try:
            mongo_db = mongo_client['regulatory_mongo']
            document = {
                'doc_id': doc_id,
                'content': product_info,
                'compliance_scores': {}
            }
            mongo_db.documents.insert_one(document)
            insights = analyze_document(product_info, countries, policies or [p for c in countries for p in get_country_policies().get(c, [])], doc_id)
            # Update document with compliance details
            for insight in insights:
                policy = insight.get('policy')
                sanitized_policy_name = sanitize_field_name(policy)
                mongo_db.documents.update_one(
                    {'doc_id': doc_id},
                    {'$set': {
                        f'compliance_scores.{sanitized_policy_name}': {
                            'score': int(insight.get('compliance_score', 0)),
                            'excellent_points': insight.get('excellent_points', []),
                            'major_gaps': insight.get('major_gaps', [])
                        }
                    }}
                )
            response_data = {
                'doc_id': doc_id,
                'filename': doc_title,
                'insights': insights
            }
            return jsonify(response_data), 201
        except Exception as e:
            logger.error(f"Error in product info upload: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            mongo_client.close()
    except Exception as e:
        logger.error(f"Error parsing request: {str(e)}")
        return jsonify({'error': 'Invalid JSON data'}), 400

@app.route('/documents/', methods=['GET'])
def document_list():
    """
    List all documents in the database.
    Returns: Array of document objects.
    """
    mongo_client = get_mongo_client()
    try:
        mongo_db = mongo_client['regulatory_mongo']
        documents = mongo_db.documents.find()
        return jsonify(json.loads(json_util.dumps(documents))), 200
    except Exception as e:
        logger.error(f"Error fetching documents: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        mongo_client.close()

@app.route('/documents/<doc_id>/', methods=['GET'])
def document_detail(doc_id):
    """
    Get details for a specific document by doc_id.
    Returns: Document object or 404 if not found.
    """
    mongo_client = get_mongo_client()
    try:
        mongo_db = mongo_client['regulatory_mongo']
        document = mongo_db.documents.find_one({'doc_id': doc_id})
        if not document:
            return jsonify({'error': 'Document not found'}), 404
        return jsonify(json.loads(json_util.dumps(document))), 200
    except Exception as e:
        logger.error(f"Error fetching document {doc_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        mongo_client.close()


@app.route('/documents/search/', methods=['GET'])
def document_search():
    """
    Search documents by content using a query string.
    Returns: Array of matching document objects.
    """
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400
    mongo_client = get_mongo_client()
    try:
        mongo_db = mongo_client['regulatory_mongo']
        documents = mongo_db.documents.find({
            'content': {'$regex': query, '$options': 'i'}
        })
        return jsonify(json.loads(json_util.dumps(documents))), 200
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        mongo_client.close()    


# --- REPORTS CRUD ---
# Endpoints for creating, listing, updating, and deleting reports.
# TODO: Add authentication and implement logic.

@app.route('/reports/', methods=['GET', 'POST'])
def reports():
    """
    GET: List all reports.
    POST: Create a new report.
    Returns: Array of reports or new report_id.
    """
    # TODO: Add authentication (future)
    # TODO: Implement logic for GET/POST
    if request.method == 'GET':
        return jsonify([]), 200
    if request.method == 'POST':
        return jsonify({'report_id': 'new_id'}), 201

@app.route('/reports/<report_id>/', methods=['GET', 'PUT', 'DELETE'])
def report_detail(report_id):
    """
    GET: Retrieve a report by ID.
    PUT: Update a report.
    DELETE: Delete a report.
    Returns: Report object or status message.
    """
    # TODO: Add authentication (future)
    # TODO: Implement logic for GET/PUT/DELETE
    if request.method == 'GET':
        return jsonify({}), 200
    if request.method == 'PUT':
        return jsonify({}), 200
    if request.method == 'DELETE':
        return jsonify({'message': 'Deleted'}), 200

# --- FOLDERS CRUD ---
# Endpoints for creating, listing, updating, and deleting project folders.
# TODO: Add authentication and implement logic.

@app.route('/folders/', methods=['GET', 'POST'])
def folders():
    """
    GET: List all project folders.
    POST: Create a new folder.
    Returns: Array of folders or new folder_id.
    """
    # TODO: Add authentication (future)
    # TODO: Implement logic for GET/POST
    if request.method == 'GET':
        return jsonify([]), 200
    if request.method == 'POST':
        return jsonify({'folder_id': 'new_id'}), 201

@app.route('/folders/<folder_id>/', methods=['GET', 'PUT', 'DELETE'])
def folder_detail(folder_id):
    """
    GET: Retrieve a folder by ID.
    PUT: Update a folder.
    DELETE: Delete a folder.
    Returns: Folder object or status message.
    """
    # TODO: Add authentication (future)
    # TODO: Implement logic for GET/PUT/DELETE
    if request.method == 'GET':
        return jsonify({}), 200
    if request.method == 'PUT':
        return jsonify({}), 200
    if request.method == 'DELETE':
        return jsonify({'message': 'Deleted'}), 200

# --- TEAM MEMBERS CRUD ---
# Endpoints for managing team members (list, add, update, delete).
# TODO: Add authentication and implement logic.

@app.route('/team/', methods=['GET', 'POST'])
def team():
    """
    GET: List all team members.
    POST: Add a new team member.
    Returns: Array of members or new member_id.
    """
    # TODO: Add authentication (future)
    # TODO: Implement logic for GET/POST
    if request.method == 'GET':
        return jsonify([]), 200
    if request.method == 'POST':
        return jsonify({'member_id': 'new_id'}), 201

@app.route('/team/<member_id>/', methods=['GET', 'PUT', 'DELETE'])
def team_member_detail(member_id):
    """
    GET: Retrieve a team member by ID.
    PUT: Update a team member.
    DELETE: Delete a team member.
    Returns: Member object or status message.
    """
    # TODO: Add authentication (future)
    # TODO: Implement logic for GET/PUT/DELETE
    if request.method == 'GET':
        return jsonify({}), 200
    if request.method == 'PUT':
        return jsonify({}), 200
    if request.method == 'DELETE':
        return jsonify({'message': 'Deleted'}), 200

# --- CHATBOT ---
# Endpoint for chatbot interaction. Receives a message and returns a reply.
# TODO: Add authentication and implement chatbot logic.

@app.route('/chat/', methods=['POST'])
def chat():
    """
    POST: Send a message to the chatbot and receive a reply.
    Expects JSON body with 'message'.
    Returns: Chatbot reply.
    """
    # TODO: Add authentication (future)
    # TODO: Implement chatbot logic
    return jsonify({'reply': 'This is a placeholder response.'}), 200

# --- NEW REGULATORY COMPLIANCE ENDPOINTS ---
# These endpoints handle document upload and product info analysis specifically for OECD, NIST AI, and EU AI policies,
# returning compliance scores and comparisons.

def analyze_regulatory_document(text, doc_id, model="gemini"):
    """
    Analyze a document for compliance with OECD, NIST AI, and EU AI policies.
    Returns a dictionary with scores, details, and comparisons for each framework.
    """
    if not text or not text.strip():
        logger.error("Empty or invalid text provided for analysis")
        return {
            "OECD": {"score": 0, "details": "Empty document text", "comparison": "No data for comparison"},
            "NIST_AI": {"score": 0, "details": "Empty document text", "comparison": "No data for comparison"},
            "EU_AI": {"score": 0, "details": "Empty document text", "comparison": "No data for comparison"}
        }

    frameworks = {
        "OECD": "OECD AI Principles",
        "NIST_AI": "NIST AI Risk Management Framework",
        "EU_AI": "EU AI Act"
    }
    insights = {}

    # First pass: Get individual scores
    for framework, policy in frameworks.items():
        prompt = f"""
        You are an expert in the {policy}. Analyze the following product description for compliance with this policy.
        Product Description: {text}
        IMPORTANT: Return ONLY valid JSON in the exact format below, with no additional text, comments, or formatting. The compliance_score must be an integer between 0 and 100:
        {{
            "score": integer,
            "details": "string describing key aspects of the policy relevant to the document"
        }}

        LIMIT TO 100 WORDS.
        """
        try:
            max_retries = 3
            response_text = None
            for attempt in range(max_retries):
                try:
                    logger.debug(f"Attempt {attempt + 1} for {framework}")
                    response_text = analyze_policy(prompt, model=model)
                    if response_text:
                        logger.debug(f"Raw response for {framework}: {response_text[:200]}...")
                        break
                except Exception as e:
                    logger.error(f"API call failed on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise
                    continue
            if not response_text:
                raise Exception("No valid response received after retries")
            insight = extract_json_from_response(response_text)
            if insight is None:
                logger.error(f"Failed to extract JSON from response: {response_text}")
                insights[framework] = {"score": 0, "details": "Invalid JSON response format", "comparison": "No data for comparison"}
            else:
                insights[framework] = {"score": insight["score"], "details": insight["details"], "comparison": ""}
        except Exception as e:
            logger.error(f"Analysis failed for {framework}: {str(e)}")
            insights[framework] = {"score": 0, "details": str(e), "comparison": "No data for comparison"}

    # Second pass: Generate comparisons
    if any(insights[fw]["score"] > 0 for fw in frameworks):
        comparison_prompt = f"""
        You are an expert in AI regulatory compliance. Compare the compliance of the following product description across the OECD AI Principles, NIST AI Risk Management Framework, and EU AI Act.
        Product Description: {text}
        Based on the following scores: OECD={insights['OECD']['score']}, NIST_AI={insights['NIST_AI']['score']}, EU_AI={insights['EU_AI']['score']},
        provide a comparative analysis.
        IMPORTANT: Return ONLY valid JSON in the exact format below, with no additional text, comments, or formatting:
        {{
            "OECD": "string comparing OECD score to others",
            "NIST_AI": "string comparing NIST_AI score to others",
            "EU_AI": "string comparing EU_AI score to others"
        }}

        LIMIT TO 100 WORDS.
        """
        try:
            max_retries = 3
            response_text = None
            for attempt in range(max_retries):
                try:
                    logger.debug(f"Attempt {attempt + 1} for comparison")
                    response_text = analyze_policy(comparison_prompt, model=model)
                    if response_text:
                        logger.debug(f"Raw comparison response: {response_text[:200]}...")
                        break
                except Exception as e:
                    logger.error(f"API call failed on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise
                    continue
            if not response_text:
                raise Exception("No valid response received after retries")
            comparison = extract_json_from_response(response_text)
            if comparison is None:
                logger.error(f"Failed to extract JSON from comparison response: {response_text}")
            else:
                for framework in frameworks:
                    insights[framework]["comparison"] = comparison.get(framework, "No valid comparison data")
        except Exception as e:
            logger.error(f"Comparison analysis failed: {str(e)}")
            for framework in frameworks:
                insights[framework]["comparison"] = f"Comparison failed: {str(e)}"

    return insights

# Upload and analyze document for regulatory compliance
@app.route('/upload-regulatory-compliance/', methods=['POST'])
def upload_regulatory_compliance():
    """
    Upload a file and analyze it for compliance with OECD, NIST AI, and EU AI policies.
    Expects multipart form-data with 'file'.
    Returns: doc_id, filename, and insights with comparisons.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file'}), 400

    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    text = extract_text_from_pdf(file)
    doc_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)

    mongo_client = get_mongo_client()
    try:
        mongo_db = mongo_client['regulatory_mongo']
        document = {
            'doc_id': doc_id,
            'content': text,
            'compliance_scores': {}
        }
        mongo_db.documents.insert_one(document)

        insights = analyze_regulatory_document(text, doc_id)

        # Store compliance scores
        for framework, data in insights.items():
            sanitized_framework_name = sanitize_field_name(framework)
            mongo_db.documents.update_one(
                {'doc_id': doc_id},
                {'$set': {f'compliance_scores.{sanitized_framework_name}': {'score': int(data['score'])}}}
            )

        response_data = {
            'doc_id': doc_id,
            'filename': filename,
            'insights': insights
        }
        return jsonify(response_data), 201

    except Exception as e:
        logger.error(f"Error in regulatory document upload: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        mongo_client.close()

# Analyze product description for regulatory compliance
@app.route('/analyze-regulatory-product-info/', methods=['POST'])
def analyze_regulatory_product_info():
    """
    Analyze product info text for compliance with OECD, NIST AI, and EU AI policies.
    Expects JSON body with 'product_info'.
    Returns: doc_id, filename, and insights with comparisons.
    """
    try:
        data = request.get_json()
        if not data or 'product_info' not in data:
            return jsonify({'error': 'Missing product_info'}), 400

        product_info = data['product_info']
        if len(product_info) < 10:
            return jsonify({'error': 'Product info must be at least 10 characters'}), 400

        doc_id = str(uuid.uuid4())
        doc_title = f"Product_Info_{doc_id[:8]}"

        mongo_client = get_mongo_client()
        try:
            mongo_db = mongo_client['regulatory_mongo']
            document = {
                'doc_id': doc_id,
                'content': product_info,
                'compliance_scores': {}
            }
            mongo_db.documents.insert_one(document)

            insights = analyze_regulatory_document(product_info, doc_id)

            # Store compliance scores
            for framework, data in insights.items():
                sanitized_framework_name = sanitize_field_name(framework)
                mongo_db.documents.update_one(
                    {'doc_id': doc_id},
                    {'$set': {f'compliance_scores.{sanitized_framework_name}': {'score': int(data['score'])}}}
                )

            response_data = {
                'doc_id': doc_id,
                'filename': doc_title,
                'insights': insights
            }
            return jsonify(response_data), 201

        except Exception as e:
            logger.error(f"Error in regulatory product info upload: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            mongo_client.close()

    except Exception as e:
        logger.error(f"Error parsing request: {str(e)}")
        return jsonify({'error': 'Invalid JSON data'}), 400

@app.route('/api/oecd-scores', methods=['GET'])
def get_oecd_scores():
    """
    Returns OECD Principle scores and colors.
    """
    oecd_scores = [
        {"name": "Inclusive Growth & Sustainability", "score": 85, "color": "#4CAF50"},
        {"name": "Fairness & Privacy", "score": 78, "color": "#2196F3"},
        {"name": "Transparency & Explainability", "score": 72, "color": "#FF9800"},
        {"name": "Robustness, Security & Safety", "score": 88, "color": "#9C27B0"},
        {"name": "Accountability", "score": 81, "color": "#F44336"}
    ]
    return jsonify(oecd_scores), 200

@app.route('/api/nist-lifecycle-scores', methods=['GET'])
def get_nist_scores():
    """
    Returns NIST Lifecycle stages with risk and mitigability levels.
    """
    data = [
        {"name": "Plan & Design", "riskLevel": 7, "mitigabilityLevel": 5},
        {"name": "Collect & Process Data", "riskLevel": 10, "mitigabilityLevel": 8},
        {"name": "Build & Use Model", "riskLevel": 9, "mitigabilityLevel": 7},
        {"name": "Verify & Validate", "riskLevel": 6, "mitigabilityLevel": 4},
        {"name": "Deploy and Use", "riskLevel": 3, "mitigabilityLevel": 2},
        {"name": "Operate & Monitor", "riskLevel": 5, "mitigabilityLevel": 6},
    ]
    return jsonify(data), 200


# --- MAIN ENTRY POINT ---
# Starts the Flask development server if this file is run directly.
@app.route('/api/policies/relevant', methods=['POST'])
def use_case_one(u_input, selected_country):
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise Exception("MONGO_URI not set in environment variables")
    DATABASE_NAME = "chunked_data"
    COLLECTION_NAME = "chunked_data"
    VECTOR_FIELD = "plot_embedding"
    INDEX_NAME = "vector_index"  # Must match your MongoDB Atlas vector index name


    client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
    collection = client[DATABASE_NAME][COLLECTION_NAME]

    # For Gemini API key, use os.getenv
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise Exception("GEMINI_API_KEY not set in environment variables")
    genai.configure(api_key=GEMINI_API_KEY)

    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    def search_and_answer(user_input, country=None):
        # Generate embedding
        query_vector = embedder.encode(user_input).tolist()

        # Build vector search stage with country filter if provided
        vector_stage = {
            "$vectorSearch": {
                "index": "vector_index",
                "filter": {"country" : {"$eq" :country}},
                "path": "plot_embedding",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": 5
            }
        }

        # Run aggregation
        results = collection.aggregate([vector_stage])
        retrieved_chunks = [doc["text"] for doc in results]
        
        # Prepare RAG prompt
        context = "\n\n".join(retrieved_chunks)
        prompt = f"""
        You are an intelligent assistant. Use the context below to answer the user's question as concisely and informatively as possible. Your job is to provide a clean and
        concise evaluation of the users proposed company / initiative based on the context you will be provided and, as a fallback, your background knowledge. The user will describe
        their company/initiative and you will provide a risk asessment based on the guidelines set out in your context. Provide brief summaries of risk areas and compliance gaps.
        Cite the names of the documents you are referencing with short direct quotes and provide analysis as to how they relate. Summarize relevant information only. Respond exclusively in english

        Context:
        {context}

        Question: {user_input}

        Answer:
        """

        # Use Gemini Pro to generate answer
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        print("help")
        return response.text

    # ==== 4. Run example ====

    answer = search_and_answer(u_input, selected_country)
    return ("\nAnswer:\n", answer)


# app.py  (or wherever you call app.run)
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

