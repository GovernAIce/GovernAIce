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
# Import ML service layer
from ml_service import ml_service

# Initialize Flask app and load environment variables
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
load_dotenv()

# Configure logging for debugging and error tracking
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- DATABASE CONNECTION ---
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
_country_policies_cache = None
_cache_timestamp = None
CACHE_DURATION = 300  # 5 minutes

def get_country_policies():
    global _country_policies_cache, _cache_timestamp
    if (_country_policies_cache is not None and 
        _cache_timestamp is not None and 
        (datetime.now() - _cache_timestamp).seconds < CACHE_DURATION):
        return _country_policies_cache
    mongo_client = get_mongo_client()
    try:
        mongo_db = mongo_client['Training']
        collections = mongo_db.list_collection_names()
        if not collections:
            sample_countries = [
                "Canada", "UAE", "Taiwan", "Saudi Arabia", "Australia", 
                "Singapore", "South Korea", "Europe", "Brazil", "India", 
                "USA", "Japan", "UK", "China", "EU"
            ]
            for country in sample_countries:
                sample_doc = {
                    "title": f"Sample Policy for {country}",
                    "source": f"https://example.com/{country.lower().replace(' ', '-')}",
                    "text": f"This is a sample policy document for {country}."
                }
                mongo_db[country].insert_one(sample_doc)
            logger.info("Populated sample country data")
        policies = {}
        for country in mongo_db.list_collection_names():
            country_policies = mongo_db[country].find({}, {'title': 1, '_id': 0})
            policies[country] = [doc.get('title', '') for doc in country_policies if doc.get('title')]
        _country_policies_cache = policies
        _cache_timestamp = datetime.now()
        return policies
    finally:
        mongo_client.close()

def get_country_policies_list():
    global _country_policies_cache, _cache_timestamp
    if (_country_policies_cache is not None and 
        _cache_timestamp is not None and 
        (datetime.now() - _cache_timestamp).seconds < CACHE_DURATION):
        return _country_policies_cache
    mongo_client = get_mongo_client()
    try:
        mongo_db = mongo_client['Training']
        collections = mongo_db.list_collection_names()
        if not collections:
            sample_countries = [
                "Canada", "UAE", "Taiwan", "Saudi Arabia", "Australia", 
                "Singapore", "South Korea", "Europe", "Brazil", "India", 
                "USA", "Japan", "UK", "China", "EU"
            ]
            for country in sample_countries:
                sample_doc = {
                    "title": f"Sample Policy for {country}",
                    "source": f"https://example.com/{country.lower().replace(' ', '-')}",
                    "text": f"This is a sample policy document for {country}.",
                    "metadata": {"regulator": f"{country} Regulatory Authority"}
                }
                mongo_db[country].insert_one(sample_doc)
            logger.info("Populated sample country data")
        policies = {}
        for country in mongo_db.list_collection_names():
            country_policies = mongo_db[country].find({}, {'title': 1, 'source': 1, 'metadata': 1, '_id': 0})
            policies[country] = list(country_policies)
        _country_policies_cache = policies
        _cache_timestamp = datetime.now()
        return policies
    finally:
        mongo_client.close()

def clear_country_policies_cache():
    global _country_policies_cache, _cache_timestamp
    _country_policies_cache = None
    _cache_timestamp = None
    logger.info("Country policies cache cleared")

# --- PDF & LLM HELPERS ---
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
    if not field_name:
        return "unnamed_field"
    sanitized = field_name.replace('.', '_').replace('$', '_').replace(':', '_').replace('—', '_').replace('-', '_')
    sanitized = sanitized.replace(' ', '_').replace('/', '_').replace('\\', '_')
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', sanitized)
    if sanitized and sanitized[0].isdigit():
        sanitized = 'field_' + sanitized
    if not sanitized:
        sanitized = "unnamed_field"
    return sanitized

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

def create_fallback_insight(policy, error_msg):
    return {
        'policy': policy,
        'compliance_score': 0,
        'policy_details': f'Unable to analyze: {error_msg}',
        'excellent_points': [],
        'major_gaps': [f'Analysis failed due to: {error_msg}']
    }

# --- MODEL-AGNOSTIC DOCUMENT ANALYSIS ---
def analyze_document(text, countries, policies, doc_id, model="gemini"):
    """
    Analyze a document for compliance using the specified LLM model (default: gemini).
    Ensures excellent_points and major_gaps are only for relevant policies.
    """
    if not text or not text.strip():
        logger.error("Empty or invalid text provided for analysis")
        return [create_fallback_insight(policy, "Empty document text") for policy in policies]
    insights = []
    # Validate policies against relevant policies
    relevant_policy_titles = set(policies)
    prompts = {}
    for country in countries:
        for policy in policies:
            if policy not in relevant_policy_titles:
                logger.warning(f"Policy {policy} not in relevant policies, skipping")
                insights.append(create_fallback_insight(policy, "Not a relevant policy"))
                continue
            if not isinstance(policy, str):
                logger.warning(f"Invalid policy type for {policy}, expected string, got {type(policy)}")
                insights.append(create_fallback_insight(str(policy), "Invalid policy format"))
                continue
            prompts[policy] = f"""
            You are an expert in the {policy} from {country}. Analyze the following product description for compliance with this policy.
            Product Description: {text}
            IMPORTANT: Return ONLY valid JSON in the exact format below, with no additional text, comments, or formatting. The compliance_score must be an integer between 0 and 100:
            {{
                "policy": "{policy}",
                "compliance_score": integer,
                "policy_details": "string describing key aspects of the policy relevant to the document",
                "excellent_points": ["string", "string"],
                "major_gaps": ["string", "string"]
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

# --- NEW POLICY FETCH FUNCTION ---
def fetch_relevant_policies(countries, domain='', search='', mongo_client=None):
    """
    Fetch relevant policies from the Training database based on countries, domain, and search query.
    Returns a list of policy objects with title, source, regulator, country, domain, excluding text.
    Limited to the top 5 most relevant policies based on relevance scoring.
    """
    try:
        close_client = False
        if mongo_client is None:
            mongo_client = get_mongo_client()
            close_client = True
        try:
            relevant_policies = []
            training_db = mongo_client['Training']
            
            logger.info(f"Fetching policies for countries: {countries}")
            logger.info(f"Available collections: {training_db.list_collection_names()}")
            
            for country in countries:
                # Check if the country collection exists
                if country not in training_db.list_collection_names():
                    logger.warning(f"Country collection '{country}' not found in database")
                    continue
                
                # Get all policies for this country
                country_policies = training_db[country].find({}, {'_id': 0, 'title': 1, 'source': 1, 'metadata': 1, 'text': 1})
                policy_count = 0
                
                for policy_doc in country_policies:
                    policy_count += 1
                    if not policy_doc or 'title' not in policy_doc:
                        logger.warning(f"Invalid policy document in {country} collection: {policy_doc}")
                        continue
                    
                    # Calculate relevance score
                    relevance_score = 10
                    title_lower = policy_doc['title'].lower()
                    
                    # Domain matching
                    if domain and domain.lower() in title_lower:
                        relevance_score += 5
                    
                    # Search query matching
                    if search:
                        query_terms = search.lower().split()
                        for term in query_terms:
                            if term in title_lower:
                                relevance_score += 3
                            if term in country.lower():
                                relevance_score += 2
                    
                    # Create policy object
                    policy_obj = {
                        'title': policy_doc['title'],
                        'source': policy_doc.get('source', ''),
                        'regulator': policy_doc.get('metadata', {}).get('regulator', 'N/A'),
                        'country': country,
                        'domain': domain or 'general',
                        'relevance_score': relevance_score
                    }
                    relevant_policies.append(policy_obj)
                
                logger.info(f"Found {policy_count} policies for country: {country}")
            
            # Sort by relevance score and take top 5
            relevant_policies.sort(key=lambda x: x['relevance_score'], reverse=True)
            relevant_policies = relevant_policies[:5]
            
            # Remove relevance score from final output
            for policy in relevant_policies:
                policy.pop('relevance_score', None)
            
            logger.info(f"Returning {len(relevant_policies)} relevant policies for countries: {countries}")
            return relevant_policies
            
        finally:
            if close_client:
                mongo_client.close()
    except Exception as e:
        logger.error(f"Error fetching relevant policies: {str(e)}")
        return []

# ============================================================================
# CORE UPLOAD & ANALYSIS ENDPOINTS
# ============================================================================

@app.route('/api/upload-analyze-policies/', methods=['POST'])
def upload_analyze_policies():
    """
    Upload a file, analyze its extracted text for compliance, fetch relevant policies, and perform a risk assessment.
    Expects multipart form-data with 'file', 'countries' (JSON list), 'domain', and 'search'.
    Returns: doc_id, filename, overall_score, insights, policies, risk_assessments.
    """
    if 'file' not in request.files or 'countries' not in request.form:
        return jsonify({'error': 'Missing file or countries'}), 400

    file = request.files['file']
    try:
        countries = json.loads(request.form['countries'])
        domain = request.form.get('domain', '')
        search_query = request.form.get('search', '')
        if not isinstance(countries, list):
            return jsonify({'error': 'Invalid countries format'}), 400
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid countries format'}), 400

    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    text = extract_text_from_pdf(file)
    if text.startswith("Error extracting text"):
        return jsonify({'error': text}), 400
    if len(text) < 10:
        return jsonify({'error': 'Extracted text must be at least 10 characters'}), 400

    doc_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)

    mongo_client = get_mongo_client()
    try:
        valid_countries = set(get_country_policies_list().keys())
        if not all(c in valid_countries for c in countries):
            return jsonify({'error': 'Invalid country specified'}), 400

        relevant_policies = fetch_relevant_policies(countries, domain, search_query, mongo_client)
        policy_titles = [policy['title'] for policy in relevant_policies if isinstance(policy, dict) and 'title' in policy]

        insights = analyze_document(
            text=text,
            countries=countries,
            policies=policy_titles,
            doc_id=doc_id,
            model="gemini"
        )

        valid_scores = [insight['compliance_score'] for insight in insights if isinstance(insight.get('compliance_score'), int) and 0 <= insight['compliance_score'] <= 100]
        overall_score = int(sum(valid_scores) / len(valid_scores)) if valid_scores else 0

        MONGO_URI = os.getenv("MONGO_URI")
        if not MONGO_URI:
            raise Exception("MONGO_URI not set in environment variables")
        DATABASE_NAME = "chunked_data"
        COLLECTION_NAME = "chunked_data"
        VECTOR_FIELD = "plot_embedding"
        INDEX_NAME = "vector_index"

        collection = mongo_client[DATABASE_NAME][COLLECTION_NAME]

        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            risk_assessments = {country: "Risk assessment not available - configure GEMINI_API_KEY" for country in countries}
        else:
            genai.configure(api_key=GEMINI_API_KEY)

        embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        query_vector = embedder.encode(text).tolist()

        def search_and_answer(text, country=None):
            vector_stage = {
                "$vectorSearch": {
                    "index": INDEX_NAME,
                    "filter": {"country": {"$eq": country}} if country else {},
                    "path": VECTOR_FIELD,
                    "queryVector": query_vector,
                    "numCandidates": 100,
                    "limit": 5
                }
            }
            results = collection.aggregate([vector_stage])
            retrieved_chunks = [doc["text"] for doc in results]
            context = "\n\n".join(retrieved_chunks)
            prompt = f"""
            You are an intelligent assistant. Use the context below to answer the user's question as concisely and informatively as possible. Your job is to provide a clean and
            concise evaluation of the user's proposed company/initiative based on the context and, as a fallback, your background knowledge. Provide brief summaries of risk areas
            and compliance gaps. Cite document names with short direct quotes and provide analysis. Summarize relevant information only. Respond exclusively in English.

            Context:
            {context}

            Question: {text}

            Answer:
            """
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            return response.text

        risk_assessments = {}
        for country in countries:
            risk_assessments[country] = search_and_answer(text, country)

        mongo_db = mongo_client['regulatory_mongo']
        document = {
            'doc_id': doc_id,
            'content': text,
            'compliance_scores': {
                sanitize_field_name(insight.get('policy')): {
                    'score': int(insight.get('compliance_score', 0)),
                    'excellent_points': insight.get('excellent_points', []),
                    'major_gaps': insight.get('major_gaps', [])
                } for insight in insights
            },
            'countries': countries,
            'domain': domain,
            'search_query': search_query,
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'overall_score': overall_score
        }
        mongo_db.documents.insert_one(document)

        response_data = {
            'doc_id': doc_id,
            'filename': filename,
            'overall_score': overall_score,
            'insights': insights,
            'policies': relevant_policies,
            'risk_assessments': risk_assessments,
            'total_count': len(relevant_policies),
            'countries_searched': countries,
            'domain': domain,
            'search_query': search_query
        }
        return jsonify(response_data), 201

    except Exception as e:
        logger.error(f"Error in upload-analyze-policies: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        mongo_client.close()

# ============================================================================
# POLICY MANAGEMENT ENDPOINTS
# ============================================================================

@app.route('/view-policy/', methods=['GET'])
def view_policy_document():
    title = request.args.get('title')
    country = request.args.get('country')
    if not title or not country:
        return jsonify({'error': 'Missing title or country parameter'}), 400
    mongo_client = get_mongo_client()
    try:
        mongo_db = mongo_client['Training']
        document = mongo_db[country].find_one({'title': title}, {'title': 1, 'source': 1, 'text': 1, 'metadata': 1, '_id': 0})
        if not document:
            return jsonify({'error': f"Policy '{title}' not found for country '{country}'"}), 404
        return jsonify({
            'title': document['title'],
            'source': document['source'],
            'regulator': document['metadata'].get('regulator', 'N/A'),
            'country': country,
            'text': document['text']
        }), 200
    except Exception as e:
        logger.error(f"Error retrieving policy document {title} for {country}: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        mongo_client.close()

# ============================================================================
# DOCUMENT MANAGEMENT ENDPOINTS
# ============================================================================

@app.route('/product-info-upload/', methods=['POST'])
def product_info_upload():
    try:
        data = request.get_json()
        if not data or 'product_info' not in data or 'countries' not in data:
            return jsonify({'error': 'Missing product_info or countries'}), 400
        product_info = data['product_info']
        countries = data['countries']
        policies = data.get('policies', [])
        if not isinstance(countries, list) or not isinstance(policies, list) or len(product_info) < 10:
            return jsonify({'error': 'Invalid input: countries and policies must be lists, product_info must be at least 10 characters'}), 400
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

# --- PLACEHOLDER ENDPOINTS REMOVED ---
# Removed unused CRUD endpoints for reports, folders, team, and chat
# These were placeholder endpoints that weren't being used by the frontend

# ============================================================================
# REGULATORY COMPLIANCE ENDPOINTS (DEPRECATED)
# ============================================================================
# These endpoints are deprecated but kept for backward compatibility
# Use /api/upload-analyze-policies/ for comprehensive analysis

def analyze_regulatory_document(text, doc_id, model="gemini"):
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

@app.route('/api/oecd-scores', methods=['GET'])
def get_oecd_scores():
    # Get the most recent analysis results from the database
    mongo_client = get_mongo_client()
    try:
        mongo_db = mongo_client['regulatory_mongo']
        # Get the most recent document with OECD analysis
        latest_doc = mongo_db.documents.find_one(
            {'compliance_scores': {'$exists': True}},
            sort=[('timestamp', -1)]
        )
        
        if latest_doc and 'compliance_scores' in latest_doc:
            # Extract OECD-related scores from the analysis
            compliance_scores = latest_doc['compliance_scores']
            
            # Map the analysis results to OECD principles
            oecd_scores = [
                {
                    "name": "Inclusive and Sustainability",
                    "score": compliance_scores.get('inclusive_growth_sustainability', {}).get('score', 55),
                    "color": "#4CAF50"
                },
                {
                    "name": "Fairness and Privacy", 
                    "score": compliance_scores.get('fairness_privacy', {}).get('score', 50),
                    "color": "#2196F3"
                },
                {
                    "name": "Transparency and explainability",
                    "score": compliance_scores.get('transparency_explainability', {}).get('score', 45),
                    "color": "#9C27B0"
                },
                {
                    "name": "Robustness, security, and safety",
                    "score": compliance_scores.get('robustness_security_safety', {}).get('score', 35),
                    "color": "#FF9800"
                },
                {
                    "name": "AI",
                    "score": compliance_scores.get('ai_governance', {}).get('score', 30),
                    "color": "#F44336"
                },
                {
                    "name": "Accountability",
                    "score": compliance_scores.get('accountability', {}).get('score', 45),
                    "color": "#4CAF50"
                }
            ]
        else:
            # Fallback to default scores if no analysis data available
            oecd_scores = [
                {"name": "Inclusive and Sustainability", "score": 55, "color": "#4CAF50"},
                {"name": "Fairness and Privacy", "score": 50, "color": "#2196F3"},
                {"name": "Transparency and explainability", "score": 45, "color": "#9C27B0"},
                {"name": "Robustness, security, and safety", "score": 35, "color": "#FF9800"},
                {"name": "AI", "score": 30, "color": "#F44336"},
                {"name": "Accountability", "score": 45, "color": "#4CAF50"}
            ]
        
        return jsonify(oecd_scores), 200
    except Exception as e:
        logger.error(f"Error fetching OECD scores: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        mongo_client.close()

@app.route('/api/nist-lifecycle-scores', methods=['GET'])
def get_nist_scores():
    # Get the most recent analysis results from the database
    mongo_client = get_mongo_client()
    try:
        mongo_db = mongo_client['regulatory_mongo']
        # Get the most recent document with NIST analysis
        latest_doc = mongo_db.documents.find_one(
            {'compliance_scores': {'$exists': True}},
            sort=[('timestamp', -1)]
        )
        
        if latest_doc and 'compliance_scores' in latest_doc:
            # Extract NIST-related scores from the analysis
            compliance_scores = latest_doc['compliance_scores']
            
            # Map the analysis results to NIST lifecycle stages
            data = [
                {
                    "name": "Plan & Design",
                    "riskLevel": compliance_scores.get('plan_design', {}).get('risk_score', 7),
                    "mitigabilityLevel": compliance_scores.get('plan_design', {}).get('mitigation_score', 5)
                },
                {
                    "name": "Collect & Process Data",
                    "riskLevel": compliance_scores.get('collect_process_data', {}).get('risk_score', 8),
                    "mitigabilityLevel": compliance_scores.get('collect_process_data', {}).get('mitigation_score', 5)
                },
                {
                    "name": "Build & Use Model",
                    "riskLevel": compliance_scores.get('build_use_model', {}).get('risk_score', 9),
                    "mitigabilityLevel": compliance_scores.get('build_use_model', {}).get('mitigation_score', 4)
                },
                {
                    "name": "Verify & Validate",
                    "riskLevel": compliance_scores.get('verify_validate', {}).get('risk_score', 6),
                    "mitigabilityLevel": compliance_scores.get('verify_validate', {}).get('mitigation_score', 2)
                },
                {
                    "name": "Deploy and Use",
                    "riskLevel": compliance_scores.get('deploy_use', {}).get('risk_score', 4),
                    "mitigabilityLevel": compliance_scores.get('deploy_use', {}).get('mitigation_score', 1)
                },
                {
                    "name": "Operate & Monitor",
                    "riskLevel": compliance_scores.get('operate_monitor', {}).get('risk_score', 6),
                    "mitigabilityLevel": compliance_scores.get('operate_monitor', {}).get('mitigation_score', 3)
                }
            ]
        else:
            # Fallback to default scores if no analysis data available
            data = [
                {"name": "Plan & Design", "riskLevel": 7, "mitigabilityLevel": 5},
                {"name": "Collect & Process Data", "riskLevel": 8, "mitigabilityLevel": 5},
                {"name": "Build & Use Model", "riskLevel": 9, "mitigabilityLevel": 4},
                {"name": "Verify & Validate", "riskLevel": 6, "mitigabilityLevel": 2},
                {"name": "Deploy and Use", "riskLevel": 4, "mitigabilityLevel": 1},
                {"name": "Operate & Monitor", "riskLevel": 6, "mitigabilityLevel": 3}
            ]
        
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error fetching NIST scores: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        mongo_client.close()

@app.route('/api/eu-risk-level', methods=['GET'])
def eu_risk_level():
    try:
        data = [
            {
                "label": "Level 1-Minimal / No Risk",
                "riskScore": 5.5,
                "mitigationScore": 6
            },
            {
                "label": "Level 2-Limited Risk",
                "riskScore": 3,
                "mitigationScore": 5
            },
            {
                "label": "Level 3-High Risk",
                "riskScore": 2,
                "mitigationScore": 2
            },
            {
                "label": "Level 4-Unacceptable Risk",
                "riskScore": 0,
                "mitigationScore": 0
            }
        ]
        return jsonify({"data": data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/relevant-policies', methods=['GET'])
def get_relevant_policies_endpoint():
    chart_data = [
        {
            "label": "Figma",
            "values": {
                "2020": 30,
                "2021": 70,
                "2022": 30,
                "2023": 30,
                "2024": 90,
                "2025": 45
            }
        },
        {
            "label": "AI",
            "values": {
                "2020": 60,
                "2021": 65,
                "2022": 70,
                "2023": 10,
                "2024": 75,
                "2025": 45
            }
        }
    ]
    return jsonify(chart_data), 200

@app.route('/api/radar-data', methods=['GET'])
def get_radar_data():
    radar_data = [
        { "category": "Inclusive and Sustainability", "2020": 80, "2021": 100, "2022": 60 },
        { "category": "Fairness and Privacy", "2020": 95, "2021": 80, "2022": 90 },
        { "category": "Transparency and explainability", "2020": 50, "2021": 60, "2022": 65 },
        { "category": "Robustness, security, and safety", "2020": 100, "2021": 75, "2022": 55 },
        { "category": "AI", "2020": 90, "2021": 90, "2022": 40 },
        { "category": "Accountability", "2020": 80, "2021": 70, "2022": 35 }
    ]
    return jsonify(radar_data), 200

# ============================================================================
# MACHINE LEARNING ENDPOINTS
# ============================================================================

@app.route('/ml/status/', methods=['GET'])
def ml_model_status():
    try:
        status = ml_service.get_model_status()
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Error getting ML model status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/ml/compliance-analysis/', methods=['POST'])
def ml_compliance_analysis():
    try:
        data = request.get_json()
        if not data or 'document_text' not in data or 'country' not in data:
            return jsonify({'error': 'Missing document_text or country'}), 400
        document_text = data['document_text']
        country = data['country']
        input_type = data.get('input_type', 'text')
        if len(document_text) < 10:
            return jsonify({'error': 'Document text must be at least 10 characters'}), 400
        result = ml_service.analyze_compliance(document_text, country, input_type)
        doc_id = str(uuid.uuid4())
        mongo_client = get_mongo_client()
        try:
            mongo_db = mongo_client['regulatory_mongo']
            document = {
                'doc_id': doc_id,
                'content': document_text,
                'country': country,
                'analysis_type': 'ml_compliance',
                'ml_results': result,
                'timestamp': datetime.now().isoformat()
            }
            mongo_db.documents.insert_one(document)
            response_data = {
                'doc_id': doc_id,
                'country': country,
                'analysis_type': 'ml_compliance',
                'results': result
            }
            return jsonify(response_data), 201
        except Exception as e:
            logger.error(f"Error storing ML compliance results: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            mongo_client.close()
    except Exception as e:
        logger.error(f"Error in ML compliance analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/ml/policy-comparison/', methods=['POST'])
def ml_policy_comparison():
    try:
        data = request.get_json()
        if not data or 'user_document' not in data or 'reference_file' not in data:
            return jsonify({'error': 'Missing user_document or reference_file'}), 400
        user_document = data['user_document']
        reference_file = data['reference_file']
        country = data.get('country')
        if len(user_document) < 10:
            return jsonify({'error': 'User document must be at least 10 characters'}), 400
        result = ml_service.compare_policies(user_document, reference_file, country)
        doc_id = str(uuid.uuid4())
        mongo_client = get_mongo_client()
        try:
            mongo_db = mongo_client['regulatory_mongo']
            document = {
                'doc_id': doc_id,
                'user_document': user_document,
                'reference_file': reference_file,
                'country': country,
                'analysis_type': 'ml_policy_comparison',
                'ml_results': result,
                'timestamp': datetime.now().isoformat()
            }
            mongo_db.documents.insert_one(document)
            response_data = {
                'doc_id': doc_id,
                'reference_file': reference_file,
                'country': country,
                'analysis_type': 'ml_policy_comparison',
                'results': result
            }
            return jsonify(response_data), 201
        except Exception as e:
            logger.error(f"Error storing ML policy comparison results: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            mongo_client.close()
    except Exception as e:
        logger.error(f"Error in ML policy comparison: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/ml/principle-assessment/', methods=['POST'])
def ml_principle_assessment():
    try:
        data = request.get_json()
        if not data or 'document_path' not in data or 'embeddings_file' not in data:
            return jsonify({'error': 'Missing document_path or embeddings_file'}), 400
        document_path = data['document_path']
        embeddings_file = data['embeddings_file']
        if not os.path.exists(document_path):
            return jsonify({'error': f'Document file not found: {document_path}'}), 400
        if not os.path.exists(embeddings_file):
            return jsonify({'error': f'Embeddings file not found: {embeddings_file}'}), 400
        result = ml_service.assess_principles(document_path, embeddings_file)
        doc_id = str(uuid.uuid4())
        mongo_client = get_mongo_client()
        try:
            mongo_db = mongo_client['regulatory_mongo']
            document = {
                'doc_id': doc_id,
                'document_path': document_path,
                'embeddings_file': embeddings_file,
                'analysis_type': 'ml_principle_assessment',
                'ml_results': result,
                'timestamp': datetime.now().isoformat()
            }
            mongo_db.documents.insert_one(document)
            response_data = {
                'doc_id': doc_id,
                'document_path': document_path,
                'embeddings_file': embeddings_file,
                'analysis_type': 'ml_principle_assessment',
                'results': result
            }
            return jsonify(response_data), 201
        except Exception as e:
            logger.error(f"Error storing ML principle assessment results: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            mongo_client.close()
    except Exception as e:
        logger.error(f"Error in ML principle assessment: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/ml/analysis/<doc_id>/', methods=['GET'])
def get_ml_analysis(doc_id):
    mongo_client = get_mongo_client()
    try:
        mongo_db = mongo_client['regulatory_mongo']
        document = mongo_db.documents.find_one({'doc_id': doc_id})
        if not document:
            return jsonify({'error': 'Document not found'}), 404
        response_data = {
            'doc_id': doc_id,
            'analysis_type': document.get('analysis_type'),
            'timestamp': document.get('timestamp'),
            'results': document.get('ml_results', {})
        }
        return jsonify(response_data), 200
    except Exception as e:
        logger.error(f"Error retrieving ML analysis {doc_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        mongo_client.close()

# ============================================================================
# METADATA ENDPOINTS
# ============================================================================

@app.route('/metadata/country/', methods=['GET'])
def get_countries():
    countries = list(get_country_policies().keys())
    return jsonify({'countries': countries}), 200

# --- DEBUG ENDPOINTS REMOVED ---
# Removed debug cache endpoints as they're not needed in production

@app.route('/metadata/domains/', methods=['GET'])
def get_domains():
    return jsonify({'domains': []}), 200

@app.route('/metadata/policies/', methods=['GET'])
def get_policies():
    return jsonify({'policies': []}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)
