import os
import json
import re
import google.generativeai as genai
from datetime import datetime
from pymongo import MongoClient

import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def extract_json_from_response(response_text):
    """
    Extract JSON from response text, handling cases where the response
    might contain extra text or markdown formatting.
    """
    if not response_text or not response_text.strip():
        return None
    
    # Try to parse the entire response as JSON first
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass
    
    # Look for JSON within code blocks or other formatting
    json_patterns = [
        r'```json\s*(.*?)\s*```',  # JSON in markdown code blocks
        r'```\s*(.*?)\s*```',      # JSON in generic code blocks
        r'\{.*\}',                 # Any JSON object in the text
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    return None


def create_fallback_insight(framework, error_msg):
    """Create a structured fallback insight when analysis fails."""
    return {
        'framework': framework,
        'risk_classification': 'Analysis Failed',
        'key_requirements': [f'Unable to analyze: {error_msg}'],
        'implementation_actions': ['Manual review required', 'Retry with different prompt']
    }


def analyze_document(text, frameworks, doc_id):
    """
    Analyze document text against specified regulatory frameworks.
    
    Args:
        text (str): Document text to analyze
        frameworks (list): List of framework names to analyze against
        doc_id (str): Unique document identifier
    
    Returns:
        list: List of insights for each framework
    """
    # Validate input
    if not text or not text.strip():
        logger.error("Empty or invalid text provided for analysis")
        return [create_fallback_insight(fw, "Empty document text") for fw in frameworks]
    
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    insights = []

    # Enhanced prompts with clearer JSON formatting instructions
    prompts = {
        'EU_AI_ACT': f"""
        You are an expert in the EU AI Act. Analyze the following product description for compliance with the EU AI Act, which classifies AI systems into Unacceptable Risk (banned, e.g., social scoring), High Risk (e.g., healthcare, transport; requires oversight, accuracy, logging), Limited Risk (e.g., chatbots; requires notification), and Minimal Risk (e.g., spam filters; no obligations).
        
        Product Description: {text}
        
        IMPORTANT: Return ONLY valid JSON in the exact format below, with no additional text or formatting:
        {{
            "framework": "EU_AI_ACT",
            "risk_classification": "Unacceptable, High, Limited, or Minimal",
            "key_requirements": ["string", "string"],
            "implementation_actions": ["string", "string"]
        }}
        """,
        'NIST_RMF': f"""
        You are an expert in the NIST AI Risk Management Framework (AI RMF). Analyze the product description for compliance with NIST AI RMF, focusing on GOVERN (policy, ethics), MAP (risk identification), MEASURE (fairness, robustness), and MANAGE (risk mitigation).
        
        Product Description: {text}
        
        IMPORTANT: Return ONLY valid JSON in the exact format below, with no additional text or formatting:
        {{
            "framework": "NIST_RMF",
            "risk_classification": "High, Medium, Low, or None",
            "key_requirements": ["string", "string"],
            "implementation_actions": ["string", "string"]
        }}
        """,
        'CPRA': f"""
        You are an expert in the California Privacy Rights Act (CPRA). Analyze the product description for CPRA compliance, focusing on data subject rights: know, access, delete, correct, opt-out of sharing, restrict sensitive data.
        
        Product Description: {text}
        
        IMPORTANT: Return ONLY valid JSON in the exact format below, with no additional text or formatting:
        {{
            "framework": "CPRA",
            "risk_level": "High, Medium, Low, or None",
            "key_functions": ["string", "string"],
            "implementation_steps": ["string", "string"]
        }}
        """
    }

    # MongoDB connection with error handling
    mongo_client = None
    mongo_db = None
    try:
        mongo_client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
        mongo_db = mongo_client['regulatory_mongo']
        # Test the connection
        mongo_client.server_info()
        logger.info("Successfully connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        mongo_client = None
        mongo_db = None

    for framework in frameworks:
        logger.info(f"Analyzing framework: {framework}")
        
        if framework not in prompts:
            insight = create_fallback_insight(framework, 'Unknown framework')
            insights.append(insight)
            
            if mongo_db is not None:
                try:
                    mongo_db.insight_logs.insert_one({
                        'doc_id': doc_id,
                        'framework': framework,
                        'insight': insight,
                        'timestamp': datetime.utcnow()
                    })
                except Exception as e:
                    logger.error(f"MongoDB insert error: {e}")
            continue

        try:
            # Generate content with retry logic
            max_retries = 3
            response = None
            
            for attempt in range(max_retries):
                try:
                    logger.debug(f"Attempt {attempt + 1} for framework {framework}")
                    response = model.generate_content(
                        prompts[framework],
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.1,  # Lower temperature for more consistent output
                            max_output_tokens=1000,
                        )
                    )
                    
                    if response and response.text:
                        logger.debug(f"Raw response for {framework}: {response.text[:200]}...")
                        break
                    else:
                        logger.warning(f"Empty response on attempt {attempt + 1}")
                        
                except Exception as e:
                    logger.error(f"API call failed on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise
                    continue
            
            if not response or not response.text:
                raise Exception("No valid response received after retries")
            
            # Extract and parse JSON from response
            insight = extract_json_from_response(response.text)
            
            if insight is None:
                logger.error(f"Failed to extract JSON from response: {response.text}")
                insight = create_fallback_insight(framework, "Invalid JSON response format")
            else:
                # Validate required fields based on framework
                required_fields = {
                    'EU_AI_ACT': ['framework', 'risk_classification', 'key_requirements', 'implementation_actions'],
                    'NIST_RMF': ['framework', 'risk_classification', 'key_requirements', 'implementation_actions'],
                    'CPRA': ['framework', 'risk_level', 'key_functions', 'implementation_steps']
                }
                
                if framework in required_fields:
                    missing_fields = [field for field in required_fields[framework] if field not in insight]
                    if missing_fields:
                        logger.warning(f"Missing fields in {framework} response: {missing_fields}")
                        insight = create_fallback_insight(framework, f"Missing required fields: {missing_fields}")
                
                logger.info(f"Successfully analyzed {framework}")
            
            insights.append(insight)
            
            # Log to MongoDB
            if mongo_db is not None:
                try:
                    mongo_db.insight_logs.insert_one({
                        'doc_id': doc_id,
                        'framework': framework,
                        'insight': insight,
                        'timestamp': datetime.utcnow(),
                        'raw_response': response.text if response else None
                    })
                except Exception as e:
                    logger.error(f"MongoDB insert error for {framework}: {e}")
                    
        except Exception as e:
            logger.error(f"Analysis failed for {framework}: {str(e)}")
            insight = create_fallback_insight(framework, str(e))
            insights.append(insight)
            
            if mongo_db is not None:
                try:
                    mongo_db.insight_logs.insert_one({
                        'doc_id': doc_id,
                        'framework': framework,
                        'insight': insight,
                        'timestamp': datetime.utcnow(),
                        'error': str(e)
                    })
                except Exception as e:
                    logger.error(f"MongoDB insert error for failed analysis: {e}")

    # Close MongoDB connection
    if mongo_client is not None:
        try:
            mongo_client.close()
            logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")
    
    logger.info(f"Analysis complete. Generated {len(insights)} insights.")
    return insights