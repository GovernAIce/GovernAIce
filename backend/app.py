from flask import Flask, request, jsonify
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
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Initialize Flask app
app = Flask(__name__)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# MongoDB connection
def get_mongo_client():
    try:
        client = MongoClient(os.getenv("MONGO_URI"))
        client.server_info()  # Test connection
        logger.info("Successfully connected to MongoDB")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise Exception(f"MongoDB connection failed: {str(e)}")

# Fetch country policies from MongoDB
def get_country_policies():
    mongo_client = get_mongo_client()
    try:
        mongo_db = mongo_client['Training']
        policies = {}
        for country in mongo_db.list_collection_names():
            country_policies = mongo_db[country].find()
            policies[country] = [doc.get('title', '') for doc in country_policies if doc.get('title')]
        return policies
    finally:
        mongo_client.close()

# Extract text from PDF
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

# Extract JSON from Gemini response
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

# Create fallback insight
def create_fallback_insight(policy, error_msg):
    return {
        'policy': policy,
        'compliance_score': 0,
        'policy_details': f'Unable to analyze: {error_msg}',
        'excellent_points': [],
        'major_gaps': [f'Analysis failed due to: {error_msg}']
    }

# Analyze document using Gemini API
def analyze_document(text, countries, policies, doc_id):
    if not text or not text.strip():
        logger.error("Empty or invalid text provided for analysis")
        return [create_fallback_insight(policy, "Empty document text") for policy in policies]
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        return [create_fallback_insight(policy, "Missing GEMINI_API_KEY") for policy in policies]
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
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
            response = None
            for attempt in range(max_retries):
                try:
                    logger.debug(f"Attempt {attempt + 1} for policy {policy}")
                    response = model.generate_content(
                        prompts[policy],
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.1,
                            max_output_tokens=1000,
                        )
                    )
                    if response and response.text:
                        logger.debug(f"Raw response for {policy}: {response.text[:200]}...")
                        break
                except Exception as e:
                    logger.error(f"API call failed on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise
                    continue
            
            if not response or not response.text:
                raise Exception("No valid response received after retries")
            
            insight = extract_json_from_response(response.text)
            if insight is None:
                logger.error(f"Failed to extract JSON from response: {response.text}")
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

# Routes
@app.route('/upload-and-analyze/', methods=['POST'])
def document_upload():
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
    
    # Validate countries
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
            mongo_db.documents.update_one(
                {'doc_id': doc_id},
                {'$set': {
                    f'compliance_scores.{policy}': {
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
    try:
        data = request.get_json()
        if not data or 'product_info' not in data or 'countries' not in data:
            return jsonify({'error': 'Missing product_info or countries'}), 400
        
        product_info = data['product_info']
        countries = data['countries']
        policies = data.get('policies', [])
        if not isinstance(countries, list) or not isinstance(policies, list) or len(product_info) < 10:
            return jsonify({'error': 'Invalid input: countries and policies must be lists, product_info must be at least 10 characters'}), 400
        
        # Validate countries
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
                mongo_db.documents.update_one(
                    {'doc_id': doc_id},
                    {'$set': {
                        f'compliance_scores.{policy}': {
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

if __name__ == '__main__':
    app.run(debug=True)