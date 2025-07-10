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

# Country to policies mapping
COUNTRY_POLICIES = {
    'EU': [
        'EU Artificial Intelligence Act',
        'General Data Protection Regulation (GDPR)',
        'European Health Data Space Regulation (EHDS)',
        'EU Data Governance Act (DGA)',
        'EU Data Act',
        'EU Network and Information Systems Directive (NIS2 Directive)',
        'Council of Europe Framework Convention on Artificial Intelligence and Human Rights, Democracy, and the Rule of Law',
        'The European Convention on Human Rights'
    ],
    'USA': [
        'Executive Order 14179: Removing Barriers to American Leadership in Artificial Intelligence',
        'Memorandum M-25-21: Advancing Governance, Innovation, and Risk Management for Agency Use of Artificial Intelligence',
        'Memorandum M-25-22: Driving Efficient and Ethical Acquisition of Artificial Intelligence',
        'Executive Order on Advancing Artificial Intelligence Education for American Youth',
        'Executive Order on Cybersecurity and AI',
        'Planned Executive Order on AI Infrastructure and Energy Supply',
        'Maintaining American Leadership in Artificial Intelligence',
        'National AI Initiative Act',
        'California Consumer Privacy Act (CCPA)',
        'AV Policy Guidance 4.0',
        'St. Louis Ordinance on the City\'s Use of Surveillance Technology',
        'Illinois Biometric Information Privacy Act',
        'NIST AI Risk Management Framework (RMF 1.0)'
    ],
    'Canada': [
        'Bill C-27: Digital Charter Implementation Act, 2022',
        'Consumer Privacy Protection Act (CPPA)',
        'Personal Information and Data Protection Tribunal Act (CPPA)',
        'Artificial Intelligence and Data Act (AIDA)',
        'Voluntary Code of Conduct on the Responsible Development and Management of Advanced Generative AI Systems',
        'Canada Personal Information Protection and Electronic Documents Act (PIPEDA)',
        'Canada Privacy Act',
        'The Canada Consumer Product Safety Act',
        'The Food and Drugs Act',
        'The Motor Vehicle Safety Act',
        'Canada Bank Act',
        'Canadian Human Rights Act',
        'The Criminal Code'
    ],
    'UK': [
        'UK Data Protection Act 2018',
        'UK Online Safety Act',
        'UK Code of Conduct for Data-Driven Health and Care Technology',
        'UK National Data Strategy',
        'UK\'s National AI Strategy',
        'UK Innovation Strategy: leading the future by creating it',
        'UK Digital Strategy',
        'The UK Government Resilience Framework',
        'Digital Regulation: driving growth and unlocking innovation',
        'UK\'s AI Regulation White Paper',
        'UK Digital Development Strategy (DDS) 2024-2030',
        'UK Signed the Council of Europe Framework Convention on Artificial Intelligence and Human Rights, Democracy, and the Rule of Law',
        'UK AI Opportunity Action Plan',
        'Data Protection and Digital Information Bill',
        'GOV.UK Chat',
        'AI and data protection risk toolkit',
        'Consultation to Copyright and Artificial Intelligence',
        'Financial Services and Markets Act 2000',
        'UK Digital Markets, Competition and Consumers Act 2024',
        'UK\'s Equality Act 2010',
        'UK Human Rights Act 1998',
        'Enterprise Act 2002',
        'Competition Act 1998'
    ],
    'China': [
        'The Interim Administrative Measures for Generative AI Services',
        'The Criminal Law of China',
        'the Civil Code of China',
        'Copyright Law of the People’s Republic of China',
        'China\'s Cybersecurity Law',
        'China’s Data Security Law',
        'China\'s Personal Information Protection Law (PIPL)',
        'The Law on the Progress of Science and Technology',
        'Code of Ethics for the New Generation Artificial Intelligence',
        'The Measures for Review of Scientific and Technological Ethics (for Trial Implementation)',
        'Guidance for the Classification and Definition of AI-Based Medical Software Products',
        'Administrative Provisions on Autonomous Vehicle Road Testing (Trial)',
        'Provisions on the Administration of Automotive Data Security (for Trial Implementation)',
        'Basic Security Requirements for Generative AI Services',
        'Cybersecurity Technology—Generative Artificial Intelligence Data Annotation Security Specification',
        'Cybersecurity Technology—Basic Security Requirements for Generative Artificial Intelligence Service'
    ],
    'Japan': [
        'Act on Promotion of Research and Development and Utilization of Artificial Intelligence-Related Technologies',
        'Act on the Protection of Personal Information (APPI)',
        'AI Strategy 2019',
        'Social Principles of Human-centric AI',
        'AI Guidelines for Business (2024 version)',
        'Guide to Evaluation Perspectives on AI Safety'
    ],
    'South Korea': [
        'National Strategy for Artificial Intelligence',
        'Personal Information Protection Act (PIPA)',
        'Basic Act on the Development of Artificial Intelligence and Creation of a Trust Base',
        'Bill on the Promotion of Artificial Intelligence Industry and Securing Trust'
    ],
    'Singapore': [
        'Singapore National AI Strategy 2.0',
        'Singapore Model AI Governance Framework',
        'Model AI Governance Framework for Generative AI'
    ],
    'India': [
        'India Responsible AI Guidelines',
        'NATIONAL STRATEGY FOR ARTIFICIAL INTELLIGENCE',
        'THE DIGITAL PERSONAL DATA PROTECTION ACT',
        'THE INFORMATION TECHNOLOGY ACT'
    ],
    'Australia': [
        'Australia AI Ethics Principles',
        'Australia\'s Privacy Act 1988',
        'Online Safety Act 2021',
        'Voluntary AI Safety Standard',
        'Mandatory Guardrails for Safe & Responsible AI'
    ],
    'Taiwan': [
        'Basic Law for Developments of Artificial Intelligence',
        'AI Technology R&D Guidelines',
        'Taiwan Artificial Intelligence Action Plan 2.0',
        'AI Basic Act'
    ],
    'UAE': [
        'UAE AI Ethics Guidelines',
        'UAE National Strategy for Artificial Intelligence',
        'UAE Charter for the Development and Use of Artificial Intelligence',
        'Law No. 3 of 2024 Establishing the Artificial Intelligence and Advanced Technology Council (AIATC)'
    ],
    'Saudi Arabia': [
        'National Strategy for Data and Artificial Intelligence',
        'AI Ethics Principles',
        'Personal Data Protection Law'
    ],
    'Brazil': [
        'Brazil’s AI Act PL 21/20',
        'Brazilian General Data Protection Law',
        'Brazil AI Act PL 2338/2023',
        'Brazilian Data Protection Law LGPD LAW No. 13,709'
    ],
    'International': [
        'AS ISO/IEC 42001:2023 - Artificial intelligence Management system',
        'Hiroshima AI Process Friends Group Declaration',
        'Hiroshima Process International Guiding Principles for All AI Actors',
        'OECD AI principle',
        'UNESCO Ethics of Artificial Intelligence',
        'Global Digital Compact'
    ]
}

# MongoDB connection
def get_mongo_client():
    try:
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
        client.server_info()  # Test connection
        logger.info("Successfully connected to MongoDB")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise Exception(f"MongoDB connection failed: {str(e)}")

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
        country_policies = policies if policies else COUNTRY_POLICIES.get(country, [])
        for policy in country_policies:
            if policy not in COUNTRY_POLICIES.get(country, []):
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
    
    # Validate countries and policies
    valid_countries = set(COUNTRY_POLICIES.keys())
    if not all(c in valid_countries for c in countries):
        return jsonify({'error': 'Invalid country specified'}), 400
    if policies:
        for policy in policies:
            if not any(policy in COUNTRY_POLICIES[c] for c in countries):
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
        
        insights = analyze_document(text, countries, policies or [p for c in countries for p in COUNTRY_POLICIES[c]], doc_id)
        
        # Update compliance_scores for specific policies
        for insight in insights:
            policy = insight.get('policy')
            if policy == 'EU Artificial Intelligence Act':
                mongo_db.documents.update_one(
                    {'doc_id': doc_id},
                    {'$set': {'compliance_scores.EU_AI_ACT': insight.get('compliance_score', 0)}}
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
        
        # Validate countries and policies
        valid_countries = set(COUNTRY_POLICIES.keys())
        if not all(c in valid_countries for c in countries):
            return jsonify({'error': 'Invalid country specified'}), 400
        if policies:
            for policy in policies:
                if not any(policy in COUNTRY_POLICIES[c] for c in countries):
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
            
            insights = analyze_document(product_info, countries, policies or [p for c in countries for p in COUNTRY_POLICIES[c]], doc_id)
            
            # Update compliance_scores for specific policies
            for insight in insights:
                policy = insight.get('policy')
                if policy == 'EU Artificial Intelligence Act':
                    mongo_db.documents.update_one(
                        {'doc_id': doc_id},
                        {'$set': {'compliance_scores.EU_AI_ACT': insight.get('compliance_score', 0)}}
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