import json
import os
from together import Together
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dataclasses import dataclass
from datetime import datetime

# MongoDB Integration (unchanged as requested)
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel
import torch

# Import configuration and utilities
from ml.config.settings import MLConfig
from ml.utils.db_connection import search_global_chunks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB Constants (unchanged)
MODEL_NAME = "joelniklaus/legal-xlm-roberta-large"
DEVICE = 'cpu'
VECTOR_DIM = 1024

# Load tokenizer and model (unchanged)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

# MongoDB Atlas connection (unchanged)
uri = MLConfig.MONGODB_URI
client = MongoClient(uri)
db = client[MLConfig.MONGODB_DB_NAME]
collection = db[MLConfig.MONGODB_COLLECTION]

# MongoDB Embedding utility (unchanged)
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

def embed_query(text: str):
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        output = model(**encoded)
    embedding = mean_pooling(output, encoded["attention_mask"])
    return embedding[0].cpu().numpy()


def search_global_chunks(query: str, k: int = 10000, country: str = None, index_name: str = "global_vector_index"):
    print(f"🔎 Searching for: {query}")
    if country:
        print(f"🌍 Country filter: {country}")
    
    query_vector = embed_query(query)

    pipeline = []
    pipeline.append({
        "$vectorSearch": {
            "index": index_name,
            "queryVector": query_vector.tolist(),
            "path": "embedding",
            "numCandidates": 10000,
            "limit": k
        }
    })

    if country:
        pipeline.append({
            "$match": {
                "country": country
            }
        })

    pipeline.append({
        "$project": {
            "title": 1,
            "text": 1,
            "metadata": 1,
            "country": 1,
            "chunk_index": 1,
            "score": {"$meta": "vectorSearchScore"}
        }
    })

    results = list(collection.aggregate(pipeline))
    return results

@dataclass
class ComplianceResult:
    """Data class for compliance analysis results"""
    overall_score: float
    major_gaps: List[Dict[str, Any]]
    excellencies: List[Dict[str, Any]]
    improvement_strategy: List[Dict[str, Any]]
    detailed_analysis: Dict[str, Any]
    referenced_policies: List[Dict[str, Any]]  # New field to store policies used

class AIComplianceChecker:
    def __init__(self, together_api_key: str = None, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the AI Compliance Checker
        
        Args:
            together_api_key: Together AI API key for strategic insights
            embedding_model: Sentence transformer model for embeddings
        """
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        
        # Initialize embedding model
        self.initialize_embedding_model()
        
        # Initialize Together AI client
        if together_api_key:
            self.client = Together(api_key=together_api_key)
        else:
            self.client = Together()
        
        # AI governance risk categories for compliance checking
        self.risk_categories = {
            "data_privacy": {
                "keywords": ["data protection", "privacy", "personal data", "consent", "data subject rights", "gdpr", "ccpa"],
                "weight": 0.15
            },
            "algorithmic_bias": {
                "keywords": ["bias", "fairness", "discrimination", "equitable", "inclusive", "diverse", "prejudice"],
                "weight": 0.14
            },
            "transparency": {
                "keywords": ["transparent", "explainable", "interpretable", "disclosure", "communication", "accountability"],
                "weight": 0.13
            },
            "safety_security": {
                "keywords": ["safety", "security", "protection", "harm", "risk", "vulnerability", "cybersecurity"],
                "weight": 0.12
            },
            "human_oversight": {
                "keywords": ["human oversight", "human control", "human in the loop", "supervision", "intervention", "monitoring"],
                "weight": 0.11
            },
            "compliance_governance": {
                "keywords": ["compliance", "governance", "regulation", "legal", "standards", "certification", "audit"],
                "weight": 0.10
            },
            "risk_management": {
                "keywords": ["risk assessment", "risk management", "mitigation", "evaluation", "impact assessment"],
                "weight": 0.10
            },
            "ethical_considerations": {
                "keywords": ["ethics", "ethical", "moral", "responsible", "integrity", "values", "principles"],
                "weight": 0.09
            },
            "liability_accountability": {
                "keywords": ["liability", "accountability", "responsibility", "ownership", "legal responsibility"],
                "weight": 0.08
            },
            "innovation_development": {
                "keywords": ["innovation", "development", "research", "advancement", "technology", "progress"],
                "weight": 0.08
            }
        }
        
        # Available countries (can be expanded)
        self.available_countries = ["USA", "UK", "EU", "SINGAPORE", "CANADA", "AUSTRALIA", "JAPAN", "CHINA"]

    def initialize_embedding_model(self):
        """Initialize the embedding model"""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Initialized embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Function 1: Document Upload and Processing
    def upload_and_process_document(self, file_path: str = None, text_content: str = None) -> Dict[str, Any]:
        """
        Function 1: Upload and process user document for compliance checking
        
        Args:
            file_path: Path to document file
            text_content: Direct text input
            
        Returns:
            Processed document data
        """
        try:
            if file_path:
                # Read file based on extension
                if file_path.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                elif file_path.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        content = str(data)  # Convert JSON to string for analysis
                else:
                    raise ValueError("Supported formats: .txt, .json")
            elif text_content:
                content = text_content
            else:
                raise ValueError("Either file_path or text_content must be provided")
            
            # Process document into chunks for analysis
            chunks = self._chunk_document(content)
            
            # Create embeddings for document chunks
            embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
            
            document_data = {
                "content": content,
                "chunks": chunks,
                "embeddings": embeddings.tolist(),
                "word_count": len(content.split()),
                "chunk_count": len(chunks),
                "processed_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Document processed: {len(chunks)} chunks, {len(content.split())} words")
            return document_data
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise

    # Function 2: Country Policy Retrieval
    def retrieve_country_policies(self, country: str, relevant_terms: List[str] = None) -> List[Dict[str, Any]]:
        """
        Function 2: Retrieve relevant policy documents for selected country
        
        Args:
            country: Country code for policy retrieval
            relevant_terms: Specific terms to search for
            
        Returns:
            List of relevant policy chunks
        """
        try:
            if country not in self.available_countries:
                raise ValueError(f"Country {country} not available. Available: {', '.join(self.available_countries)}")
            
            # Default search terms if none provided
            if not relevant_terms:
                relevant_terms = [
                    "AI governance", "artificial intelligence regulation", "data protection",
                    "algorithmic accountability", "AI safety", "machine learning compliance"
                ]
            
            all_policy_chunks = []
            
            # Search for each relevant term
            for term in relevant_terms:
                results = search_global_chunks(query=term, country=country, k=50)
                for result in results:
                    result['search_term'] = term
                    all_policy_chunks.append(result)
            
            # Remove duplicates based on text content
            unique_chunks = []
            seen_texts = set()
            for chunk in all_policy_chunks:
                text_hash = hash(chunk['text'][:200])  # Use first 200 chars as identifier
                if text_hash not in seen_texts:
                    seen_texts.add(text_hash)
                    unique_chunks.append(chunk)
            
            logger.info(f"Retrieved {len(unique_chunks)} unique policy chunks for {country}")
            return unique_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving policies for {country}: {e}")
            raise

    # Function 3: Overall Compliance Score Calculation
    def calculate_overall_score(self, document_data: Dict[str, Any], policy_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Function 3: Calculate overall compliance score comparing document to policies
        
        Args:
            document_data: Processed document data
            policy_chunks: Retrieved policy chunks
            
        Returns:
            Overall compliance analysis with score
        """
        try:
            doc_embeddings = np.array(document_data["embeddings"])
            policy_texts = [chunk['text'] for chunk in policy_chunks]
            
            # Create embeddings for policy chunks
            policy_embeddings = self.embedding_model.encode(policy_texts, show_progress_bar=True)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(doc_embeddings, policy_embeddings)
            
            # Calculate coverage scores for each risk category
            category_scores = {}
            for category, details in self.risk_categories.items():
                category_score = self._calculate_category_compliance(
                    document_data["content"], policy_chunks, details["keywords"]
                )
                category_scores[category] = category_score
            
            # Calculate weighted overall score
            weighted_score = sum(
                score * self.risk_categories[category]["weight"] 
                for category, score in category_scores.items()
            ) * 10  # Scale to 0-10
            
            # Calculate semantic alignment score
            max_similarities = np.max(similarity_matrix, axis=1)
            semantic_score = np.mean(max_similarities) * 10
            
            # Combine scores (70% category-based, 30% semantic)
            overall_score = (weighted_score * 0.7) + (semantic_score * 0.3)
            
            score_analysis = {
                "overall_compliance_score": float(overall_score),
                "weighted_category_score": float(weighted_score),
                "semantic_alignment_score": float(semantic_score),
                "category_breakdown": {k: float(v) for k, v in category_scores.items()},
                "similarity_matrix_stats": {
                    "max_similarity": float(np.max(similarity_matrix)),
                    "avg_similarity": float(np.mean(similarity_matrix)),
                    "coverage_percentage": float(np.mean(max_similarities > 0.3) * 100)
                },
                "compliance_level": self._interpret_compliance_level(overall_score)
            }
            
            logger.info(f"Overall compliance score calculated: {overall_score:.2f}/10")
            return score_analysis
            
        except Exception as e:
            logger.error(f"Error calculating compliance score: {e}")
            raise

    # Function 4: Major Gaps Analysis
    def identify_major_gaps(self, document_data: Dict[str, Any], policy_chunks: List[Dict[str, Any]], 
                          score_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Function 4: Identify major gaps - risks mentioned in policy but missing in document
        
        Args:
            document_data: Processed document data
            policy_chunks: Retrieved policy chunks
            score_analysis: Overall score analysis
            
        Returns:
            List of major compliance gaps
        """
        try:
            major_gaps = []
            document_text = document_data["content"].lower()
            
            # Analyze each risk category for gaps
            for category, details in self.risk_categories.items():
                category_score = score_analysis["category_breakdown"][category]
                
                # Identify specific missing requirements
                missing_keywords = []
                policy_requirements = []
                
                for keyword in details["keywords"]:
                    if keyword.lower() not in document_text:
                        missing_keywords.append(keyword)
                        
                        # Find policy chunks that mention this keyword
                        relevant_policies = [
                            chunk for chunk in policy_chunks 
                            if keyword.lower() in chunk['text'].lower()
                        ]
                        
                        for policy in relevant_policies[:2]:  # Top 2 most relevant
                            requirement = self._extract_requirement_context(policy['text'], keyword)
                            if requirement:
                                policy_requirements.append({
                                    "keyword": keyword,
                                    "requirement": requirement,
                                    "policy_source": policy.get('title', 'Unknown'),
                                    "country": policy.get('country', 'Unknown')
                                })
                
                # Only include as major gap if score is below threshold and missing keywords exist
                if category_score < 0.4 and missing_keywords:  # Below 40% compliance
                    gap = {
                        "category": category.replace('_', ' ').title(),
                        "severity": "HIGH" if category_score < 0.2 else "MEDIUM",
                        "compliance_score": f"{category_score * 10:.1f}/10",
                        "missing_keywords": missing_keywords,
                        "policy_requirements": policy_requirements,
                        "risk_description": self._get_risk_description(category),
                        "business_impact": self._get_business_impact(category)
                    }
                    major_gaps.append(gap)
            
            # Sort by severity and compliance score
            major_gaps.sort(key=lambda x: (x["severity"] == "HIGH", -float(x["compliance_score"].split("/")[0])))
            
            logger.info(f"Identified {len(major_gaps)} major compliance gaps")
            return major_gaps
            
        except Exception as e:
            logger.error(f"Error identifying gaps: {e}")
            raise

    # Function 5: Excellencies Identification
    def identify_excellencies(self, document_data: Dict[str, Any], policy_chunks: List[Dict[str, Any]], 
                            score_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Function 5: Identify excellencies - parts of document that satisfy policy requirements
        
        Args:
            document_data: Processed document data
            policy_chunks: Retrieved policy chunks
            score_analysis: Overall score analysis
            
        Returns:
            List of compliance excellencies
        """
        try:
            excellencies = []
            document_text = document_data["content"].lower()
            doc_chunks = document_data["chunks"]
            
            # Analyze each risk category for strengths
            for category, details in self.risk_categories.items():
                category_score = score_analysis["category_breakdown"][category]
                
                # Only include as excellence if score is above threshold
                if category_score > 0.6:  # Above 60% compliance
                    
                    # Find specific document sections that address this category
                    relevant_doc_sections = []
                    covered_keywords = []
                    
                    for keyword in details["keywords"]:
                        if keyword.lower() in document_text:
                            covered_keywords.append(keyword)
                            
                            # Find document chunks containing this keyword
                            for i, chunk in enumerate(doc_chunks):
                                if keyword.lower() in chunk.lower():
                                    context = self._extract_excellence_context(chunk, keyword)
                                    if context:
                                        relevant_doc_sections.append({
                                            "chunk_index": i,
                                            "keyword": keyword,
                                            "context": context,
                                            "text_snippet": chunk[:200] + "..."
                                        })
                    
                    # Find matching policy requirements
                    policy_matches = []
                    for policy in policy_chunks:
                        for keyword in covered_keywords:
                            if keyword.lower() in policy['text'].lower():
                                policy_matches.append({
                                    "keyword": keyword,
                                    "policy_text": policy['text'][:300] + "...",
                                    "policy_source": policy.get('title', 'Unknown'),
                                    "similarity_score": float(self._calculate_chunk_similarity(
                                        document_text, policy['text']
                                    ))
                                })
                    
                    excellence = {
                        "category": category.replace('_', ' ').title(),
                        "strength_level": "EXCELLENT" if category_score > 0.8 else "GOOD",
                        "compliance_score": f"{category_score * 10:.1f}/10",
                        "covered_keywords": covered_keywords,
                        "document_sections": relevant_doc_sections[:3],  # Top 3 sections
                        "policy_matches": sorted(policy_matches, 
                                               key=lambda x: x["similarity_score"], reverse=True)[:3],
                        "competitive_advantage": self._get_competitive_advantage(category)
                    }
                    excellencies.append(excellence)
            
            # Sort by compliance score (highest first)
            excellencies.sort(key=lambda x: -float(x["compliance_score"].split("/")[0]))
            
            logger.info(f"Identified {len(excellencies)} compliance excellencies")
            return excellencies
            
        except Exception as e:
            logger.error(f"Error identifying excellencies: {e}")
            raise

    # Function 6: Strategy for Improvement
    def generate_improvement_strategy(self, document_data: Dict[str, Any], major_gaps: List[Dict[str, Any]], 
                                    excellencies: List[Dict[str, Any]], country: str) -> List[Dict[str, Any]]:
        """
        Function 6: Generate strategic recommendations for improving compliance
        
        Args:
            document_data: Processed document data
            major_gaps: Identified compliance gaps
            excellencies: Identified compliance strengths
            country: Target country for compliance
            
        Returns:
            Strategic improvement recommendations
        """
        try:
            improvement_strategy = []
            
            # Immediate priorities (0-3 months)
            immediate_actions = []
            for gap in major_gaps[:3]:  # Top 3 critical gaps
                if gap["severity"] == "HIGH":
                    action = {
                        "priority": "CRITICAL",
                        "timeline": "0-3 months",
                        "category": gap["category"],
                        "action": f"Address {gap['category']} compliance gap",
                        "specific_steps": self._generate_specific_steps(gap),
                        "estimated_effort": "High",
                        "business_risk": gap["business_impact"]
                    }
                    immediate_actions.append(action)
            
            # Medium-term improvements (3-12 months)
            medium_term_actions = []
            for gap in major_gaps[3:]:  # Remaining gaps
                if gap["severity"] == "MEDIUM":
                    action = {
                        "priority": "MEDIUM",
                        "timeline": "3-12 months",
                        "category": gap["category"],
                        "action": f"Enhance {gap['category']} framework",
                        "specific_steps": self._generate_specific_steps(gap),
                        "estimated_effort": "Medium",
                        "business_risk": gap["business_impact"]
                    }
                    medium_term_actions.append(action)
            
            # Leverage existing strengths
            leverage_actions = []
            for excellence in excellencies[:2]:  # Top 2 strengths
                action = {
                    "priority": "LEVERAGE",
                    "timeline": "Ongoing",
                    "category": excellence["category"],
                    "action": f"Expand and showcase {excellence['category']} excellence",
                    "specific_steps": [
                        f"Document and formalize existing {excellence['category']} practices",
                        f"Create best practice templates based on current {excellence['category']} approach",
                        f"Use {excellence['category']} strength as competitive advantage"
                    ],
                    "estimated_effort": "Low",
                    "business_value": excellence["competitive_advantage"]
                }
                leverage_actions.append(action)
            
            # Generate AI-powered strategic insights
            strategic_insights = self._generate_ai_insights(major_gaps, excellencies, country)
            
            # Compile comprehensive strategy
            improvement_strategy = {
                "immediate_priorities": immediate_actions,
                "medium_term_improvements": medium_term_actions,
                "leverage_opportunities": leverage_actions,
                "strategic_insights": strategic_insights,
                "implementation_roadmap": self._create_implementation_roadmap(
                    immediate_actions, medium_term_actions, leverage_actions
                ),
                "success_metrics": self._define_success_metrics(major_gaps, excellencies)
            }
            
            logger.info("Improvement strategy generated successfully")
            return improvement_strategy
            
        except Exception as e:
            logger.error(f"Error generating improvement strategy: {e}")
            raise

    # Helper methods
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types and other non-serializable types to JSON-serializable formats"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    def _chunk_document(self, content: str, chunk_size: int = 500) -> List[str]:
        """Split document into manageable chunks"""
        words = content.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def _calculate_category_compliance(self, document_text: str, policy_chunks: List[Dict], keywords: List[str]) -> float:
        """Calculate compliance score for a specific risk category"""
        doc_lower = document_text.lower()
        keyword_coverage = sum(1 for keyword in keywords if keyword.lower() in doc_lower) / len(keywords)
        
        # Also check for semantic similarity with relevant policies
        category_policies = [
            chunk for chunk in policy_chunks 
            if any(keyword.lower() in chunk['text'].lower() for keyword in keywords)
        ]
        
        if category_policies:
            # Calculate semantic alignment with category-specific policies
            policy_texts = [policy['text'] for policy in category_policies[:5]]  # Top 5 relevant policies
            policy_embeddings = self.embedding_model.encode(policy_texts)
            doc_embedding = self.embedding_model.encode([document_text])
            
            similarities = cosine_similarity(doc_embedding, policy_embeddings)[0]
            semantic_alignment = np.mean(similarities)
        else:
            semantic_alignment = 0
        
        # Combine keyword coverage and semantic alignment
        combined_score = (keyword_coverage * 0.6) + (semantic_alignment * 0.4)
        return float(combined_score)

    def _interpret_compliance_level(self, score: float) -> str:
        """Interpret compliance score into risk level"""
        if score >= 8.0:
            return "EXCELLENT"
        elif score >= 6.5:
            return "GOOD"
        elif score >= 5.0:
            return "MODERATE"
        elif score >= 3.0:
            return "POOR"
        else:
            return "CRITICAL"

    def _extract_requirement_context(self, policy_text: str, keyword: str) -> str:
        """Extract relevant context around a keyword from policy text"""
        sentences = policy_text.split('.')
        for sentence in sentences:
            if keyword.lower() in sentence.lower():
                return sentence.strip()[:200] + "..."
        return ""

    def _extract_excellence_context(self, doc_chunk: str, keyword: str) -> str:
        """Extract context showing how document addresses a requirement"""
        sentences = doc_chunk.split('.')
        for sentence in sentences:
            if keyword.lower() in sentence.lower():
                return sentence.strip()[:200] + "..."
        return ""

    def _calculate_chunk_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text chunks"""
        embeddings = self.embedding_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    def _get_risk_description(self, category: str) -> str:
        """Get risk description for a category"""
        descriptions = {
            "data_privacy": "Risk of data breaches, privacy violations, and regulatory penalties",
            "algorithmic_bias": "Risk of discriminatory outcomes and fairness issues",
            "transparency": "Risk of lack of explainability and stakeholder trust issues",
            "safety_security": "Risk of system failures and security vulnerabilities",
            "human_oversight": "Risk of autonomous decision-making without human control",
            "compliance_governance": "Risk of regulatory non-compliance and legal issues",
            "risk_management": "Risk of inadequate risk assessment and mitigation",
            "ethical_considerations": "Risk of ethical violations and reputational damage",
            "liability_accountability": "Risk of unclear responsibility and liability issues",
            "innovation_development": "Risk of stifled innovation and competitive disadvantage"
        }
        return descriptions.get(category, "Risk of regulatory non-compliance")

    def _get_business_impact(self, category: str) -> str:
        """Get business impact description for a category"""
        impacts = {
            "data_privacy": "Potential fines, lawsuits, and loss of customer trust",
            "algorithmic_bias": "Legal liability, reputational damage, and market access restrictions",
            "transparency": "Regulatory scrutiny, customer churn, and partnership difficulties",
            "safety_security": "System downtime, data breaches, and liability claims",
            "human_oversight": "Regulatory intervention and operational restrictions",
            "compliance_governance": "Legal penalties, operational shutdown risk",
            "risk_management": "Unexpected failures and crisis situations",
            "ethical_considerations": "Brand damage and stakeholder backlash",
            "liability_accountability": "Legal uncertainty and insurance issues",
            "innovation_development": "Competitive disadvantage and market share loss"
        }
        return impacts.get(category, "Potential regulatory and business risks")

    def _get_competitive_advantage(self, category: str) -> str:
        """Get competitive advantage description for a category"""
        advantages = {
            "data_privacy": "Strong customer trust and premium positioning",
            "algorithmic_bias": "Inclusive market reach and ethical leadership",
            "transparency": "Stakeholder confidence and regulatory favorability",
            "safety_security": "Market reliability and enterprise trust",
            "human_oversight": "Responsible AI reputation and partnership opportunities",
            "compliance_governance": "Regulatory readiness and market access",
            "risk_management": "Operational resilience and investor confidence",
            "ethical_considerations": "Brand differentiation and stakeholder loyalty",
            "liability_accountability": "Clear governance and insurance benefits",
            "innovation_development": "Technology leadership and innovation culture"
        }
        return advantages.get(category, "Competitive differentiation opportunity")

    def _generate_specific_steps(self, gap: Dict[str, Any]) -> List[str]:
        """Generate specific action steps for addressing a gap"""
        category = gap["category"].lower().replace(" ", "_")
        steps = {
            "data_privacy": [
                "Implement comprehensive data mapping and classification",
                "Establish data subject rights procedures",
                "Conduct privacy impact assessments"
            ],
            "algorithmic_bias": [
                "Implement bias testing throughout development lifecycle",
                "Establish diverse testing datasets",
                "Create fairness metrics and monitoring"
            ],
            "transparency": [
                "Develop explainable AI documentation",
                "Create stakeholder communication protocols",
                "Implement algorithmic transparency measures"
            ]
        }
        return steps.get(category, [
            "Conduct gap analysis and requirements mapping",
            "Develop implementation plan and timeline",
            "Establish monitoring and compliance procedures"
        ])

    def _generate_ai_insights(self, major_gaps: List[Dict], excellencies: List[Dict], country: str) -> str:
        """Generate AI-powered strategic insights"""
        try:
            prompt = f"""
            Analyze this AI compliance assessment for {country} and provide strategic insights:
            
            Major Gaps: {[gap['category'] for gap in major_gaps]}
            Excellencies: {[exc['category'] for exc in excellencies]}
            
            Provide:
            1. Strategic prioritization of gaps
            2. Market positioning opportunities
            3. Regulatory relationship strategy
            4. Competitive advantages to leverage
            
            Be specific and actionable. Maximum 300 words.
            """
            
            response = self.client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            return response.choices[0].message.content
        except:
            return "Strategic AI insights require API configuration. Contact administrator for advanced analysis."

    def _create_implementation_roadmap(self, immediate: List, medium_term: List, leverage: List) -> Dict:
        """Create implementation roadmap"""
        return {
            "phase_1": {"timeline": "0-3 months", "actions": len(immediate), "focus": "Critical gaps"},
            "phase_2": {"timeline": "3-12 months", "actions": len(medium_term), "focus": "Framework enhancement"},
            "phase_3": {"timeline": "Ongoing", "actions": len(leverage), "focus": "Competitive advantage"}
        }

    def _define_success_metrics(self, major_gaps: List, excellencies: List) -> Dict:
        """Define success metrics for improvement"""
        return {
            "compliance_score_target": "8.0/10",
            "gaps_to_address": len(major_gaps),
            "strengths_to_leverage": len(excellencies),
            "key_metrics": [
                "Overall compliance score improvement",
                "Number of critical gaps resolved",
                "Regulatory readiness assessment",
                "Stakeholder confidence index"
            ]
        }

    def generate_text_report(self, result: ComplianceResult, country: str, document_info: str) -> str:
        """
        Generate a comprehensive text report of compliance analysis
        
        Args:
            result: ComplianceResult object with analysis data
            country: Country analyzed against
            document_info: Information about the document analyzed
            
        Returns:
            Formatted text report
        """
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("AI POLICY COMPLIANCE ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Document: {document_info}")
        report_lines.append(f"Compared Against: {country} AI Policy Framework")
        report_lines.append(f"Analysis Method: Hybrid Semantic + Risk Category Assessment")
        report_lines.append("")
        
        # Overall Score
        report_lines.append("📊 OVERALL COMPLIANCE SCORE")
        report_lines.append("-" * 40)
        score = result.overall_score
        level = result.detailed_analysis['compliance_level']
        report_lines.append(f"Score: {score:.1f}/10 ({level})")
        
        # Risk level interpretation
        if score >= 8.0:
            risk_level = "LOW RISK - Excellent compliance"
        elif score >= 6.5:
            risk_level = "MODERATE RISK - Good compliance with minor gaps"
        elif score >= 5.0:
            risk_level = "MEDIUM RISK - Acceptable compliance, improvement needed"
        elif score >= 3.0:
            risk_level = "HIGH RISK - Poor compliance, significant action required"
        else:
            risk_level = "CRITICAL RISK - Major compliance failures"
        
        report_lines.append(f"Risk Assessment: {risk_level}")
        report_lines.append("")
        
        # Detailed Score Breakdown
        report_lines.append("📈 SCORE BREAKDOWN")
        report_lines.append("-" * 40)
        report_lines.append(f"Semantic Alignment: {result.detailed_analysis['semantic_alignment_score']:.1f}/10")
        report_lines.append(f"Risk Category Coverage: {result.detailed_analysis['weighted_category_score']:.1f}/10")
        report_lines.append(f"Policy Text Overlap: {result.detailed_analysis['semantic_alignment_score']/10:.1%}")
        report_lines.append("")
        
        # Major Gaps
        report_lines.append("🚨 MAJOR COMPLIANCE GAPS")
        report_lines.append("-" * 40)
        if result.major_gaps:
            for i, gap in enumerate(result.major_gaps, 1):
                report_lines.append(f"{i}. {gap['category']} ({gap['severity']} RISK)")
                report_lines.append(f"   Score: {gap['compliance_score']}")
                report_lines.append(f"   Risk: {gap['risk_description']}")
                report_lines.append(f"   Business Impact: {gap['business_impact']}")
                
                if gap.get('missing_keywords'):
                    report_lines.append(f"   Missing Elements: {', '.join(gap['missing_keywords'][:5])}")
                
                if gap.get('policy_requirements'):
                    report_lines.append("   Policy Requirements:")
                    for req in gap['policy_requirements'][:2]:
                        report_lines.append(f"     • {req['requirement']}")
                        report_lines.append(f"       Source: {req['policy_source']} ({req['country']})")
                
                report_lines.append("")
        else:
            report_lines.append("✅ No major compliance gaps identified!")
            report_lines.append("")
        
        # Excellencies
        report_lines.append("✅ COMPLIANCE EXCELLENCIES")
        report_lines.append("-" * 40)
        if result.excellencies:
            for i, exc in enumerate(result.excellencies, 1):
                report_lines.append(f"{i}. {exc['category']} ({exc['strength_level']})")
                report_lines.append(f"   Score: {exc['compliance_score']}")
                report_lines.append(f"   Competitive Advantage: {exc['competitive_advantage']}")
                
                if exc.get('covered_keywords'):
                    report_lines.append(f"   Strong Areas: {', '.join(exc['covered_keywords'][:5])}")
                
                if exc.get('document_sections'):
                    report_lines.append("   Document Evidence:")
                    for section in exc['document_sections'][:2]:
                        report_lines.append(f"     • {section['text_snippet'][:100]}...")
                
                report_lines.append("")
        else:
            report_lines.append("⚠️ No significant excellencies identified - focus on building strengths")
            report_lines.append("")
        
        # Referenced Policies
        report_lines.append("📋 POLICY FRAMEWORK ANALYZED")
        report_lines.append("-" * 40)
        report_lines.append(f"Country: {country}")
        report_lines.append(f"Total Policy Segments: {len(result.referenced_policies)}")
        report_lines.append("")
        
        # Group policies by source/title for better readability
        policy_sources = {}
        for policy in result.referenced_policies:
            source = policy.get('title', 'Unknown Document')
            if source not in policy_sources:
                policy_sources[source] = []
            policy_sources[source].append(policy)
        
        report_lines.append("Policy Documents Referenced:")
        for i, (source, policies) in enumerate(policy_sources.items(), 1):
            report_lines.append(f"{i}. {source}")
            report_lines.append(f"   Segments Analyzed: {len(policies)}")
            report_lines.append(f"   Country: {policies[0].get('country', 'Unknown')}")
            
            # Show top similarity score for this document
            scores = [p.get('score', 0) for p in policies]
            if scores:
                report_lines.append(f"   Max Relevance Score: {max(scores):.3f}")
            
            # Show a sample text snippet
            if policies and policies[0].get('text'):
                sample_text = policies[0]['text'][:150].replace('\n', ' ').strip()
                report_lines.append(f"   Sample: {sample_text}...")
            
            report_lines.append("")
        
        # Improvement Strategy Summary
        report_lines.append("🎯 IMPROVEMENT STRATEGY SUMMARY")
        report_lines.append("-" * 40)
        strategy = result.improvement_strategy
        
        if strategy.get('immediate_priorities'):
            report_lines.append("IMMEDIATE PRIORITIES (0-3 months):")
            for action in strategy['immediate_priorities']:
                report_lines.append(f"  • {action['action']}")
                report_lines.append(f"    Priority: {action['priority']} | Effort: {action['estimated_effort']}")
        
        if strategy.get('medium_term_improvements'):
            report_lines.append("\nMEDIUM-TERM IMPROVEMENTS (3-12 months):")
            for action in strategy['medium_term_improvements']:
                report_lines.append(f"  • {action['action']}")
                report_lines.append(f"    Priority: {action['priority']} | Effort: {action['estimated_effort']}")
        
        if strategy.get('leverage_opportunities'):
            report_lines.append("\nLEVERAGE OPPORTUNITIES (Ongoing):")
            for action in strategy['leverage_opportunities']:
                report_lines.append(f"  • {action['action']}")
                report_lines.append(f"    Value: {action.get('business_value', 'High')}")
        
        report_lines.append("")
        
        # Strategic Insights
        if strategy.get('strategic_insights'):
            report_lines.append("🧠 STRATEGIC INSIGHTS")
            report_lines.append("-" * 40)
            report_lines.append(strategy['strategic_insights'])
            report_lines.append("")
        
        # Success Metrics
        if strategy.get('success_metrics'):
            report_lines.append("📈 SUCCESS METRICS")
            report_lines.append("-" * 40)
            metrics = strategy['success_metrics']
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, list):
                    report_lines.append(f"{metric_name.replace('_', ' ').title()}:")
                    for item in metric_value:
                        report_lines.append(f"  • {item}")
                else:
                    report_lines.append(f"{metric_name.replace('_', ' ').title()}: {metric_value}")
            report_lines.append("")
        
        # Footer
        report_lines.append("=" * 80)
        report_lines.append("End of Compliance Analysis Report")
        report_lines.append("For detailed technical analysis, refer to the JSON output file")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)

    # Main orchestration method
    def run_compliance_check(self, document_input: str, country: str, input_type: str = "file") -> ComplianceResult:
        """
        Main method to run complete compliance check
        
        Args:
            document_input: File path or text content
            country: Country for policy comparison
            input_type: "file" or "text"
            
        Returns:
            Complete compliance analysis results
        """
        try:
            print(f"\n{'='*80}")
            print(f"AI POLICY COMPLIANCE CHECKER")
            print(f"Document Analysis vs {country} Policy Framework")
            print(f"{'='*80}")
            
            # Function 1: Process document
            print("\n🔍 Phase 1: Processing document...")
            if input_type == "file":
                document_data = self.upload_and_process_document(file_path=document_input)
            else:
                document_data = self.upload_and_process_document(text_content=document_input)
            
            # Function 2: Retrieve policies
            print(f"\n🌍 Phase 2: Retrieving {country} policy framework...")
            policy_chunks = self.retrieve_country_policies(country)
            
            # Function 3: Calculate overall score
            print(f"\n📊 Phase 3: Calculating compliance score...")
            score_analysis = self.calculate_overall_score(document_data, policy_chunks)
            
            # Function 4: Identify gaps
            print(f"\n🚨 Phase 4: Identifying compliance gaps...")
            major_gaps = self.identify_major_gaps(document_data, policy_chunks, score_analysis)
            
            # Function 5: Identify excellencies
            print(f"\n✅ Phase 5: Identifying compliance strengths...")
            excellencies = self.identify_excellencies(document_data, policy_chunks, score_analysis)
            
            # Function 6: Generate strategy
            print(f"\n🎯 Phase 6: Generating improvement strategy...")
            improvement_strategy = self.generate_improvement_strategy(
                document_data, major_gaps, excellencies, country
            )
            
            # Compile results
            result = ComplianceResult(
                overall_score=score_analysis["overall_compliance_score"],
                major_gaps=major_gaps,
                excellencies=excellencies,
                improvement_strategy=improvement_strategy,
                detailed_analysis=score_analysis,
                referenced_policies=policy_chunks  # Include the policies used
            )
            
            print(f"\n✅ Compliance analysis completed!")
            return result
            
        except Exception as e:
            logger.error(f"Error in compliance check: {e}")
            raise

def main():
    """Main function to run the compliance checker"""
    print("AI POLICY COMPLIANCE CHECKER")
    print("=" * 50)
    print("Upload your document and select a country for compliance analysis")
    print("=" * 50)
    
    try:
        # Initialize checker
        checker = AIComplianceChecker()
        
        print(f"\nAvailable countries: {', '.join(checker.available_countries)}")
        
        # Get user inputs
        country = input("\nSelect country for policy comparison: ").strip().upper()
        if country not in checker.available_countries:
            print(f"Invalid country. Available: {', '.join(checker.available_countries)}")
            return
        
        input_method = input("\nChoose input method (1: Upload file, 2: Paste text): ").strip()
        
        if input_method == "1":
            file_path = input("Enter file path: ").strip()
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return
            document_input = file_path
            input_type = "file"
        else:
            print("Paste your document text (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            document_input = "\n".join(lines)
            input_type = "text"
        
        # Run compliance check
        result = checker.run_compliance_check(document_input, country, input_type)
        
        # Display results
        print(f"\n{'='*80}")
        print(f"COMPLIANCE ANALYSIS RESULTS")
        print(f"{'='*80}")
        
        print(f"\n📊 OVERALL SCORE: {result.overall_score:.1f}/10 ({result.detailed_analysis['compliance_level']})")
        
        print(f"\n🚨 MAJOR GAPS ({len(result.major_gaps)}):")
        for i, gap in enumerate(result.major_gaps, 1):
            print(f"{i}. {gap['category']} - {gap['severity']} RISK ({gap['compliance_score']})")
        
        print(f"\n✅ EXCELLENCIES ({len(result.excellencies)}):")
        for i, exc in enumerate(result.excellencies, 1):
            print(f"{i}. {exc['category']} - {exc['strength_level']} ({exc['compliance_score']})")
        
        print(f"\n🎯 IMPROVEMENT STRATEGY:")
        strategy = result.improvement_strategy
        print(f"   Immediate priorities: {len(strategy['immediate_priorities'])}")
        print(f"   Medium-term improvements: {len(strategy['medium_term_improvements'])}")
        print(f"   Leverage opportunities: {len(strategy['leverage_opportunities'])}")
        
        # Save detailed results
        save_results = input("\nSave detailed results to files? (y/n): ").strip().lower()
        if save_results == 'y':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"compliance_analysis_{country}_{timestamp}"
            
            # Determine document info for report
            if input_type == "file":
                document_info = document_input
            else:
                document_info = f"Text input ({len(document_input.split())} words)"
            
            # Generate and save text report
            txt_filename = f"{base_filename}.txt"
            text_report = checker.generate_text_report(result, country, document_info)
            
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(text_report)
            
            print(f"✅ Text report saved to: {txt_filename}")
            
            # Save detailed JSON data
            json_filename = f"{base_filename}.json"
            results_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "country": country,
                    "document_info": document_info,
                    "analysis_type": "AI Policy Compliance Check"
                },
                "overall_score": result.overall_score,
                "compliance_level": result.detailed_analysis['compliance_level'],
                "major_gaps": result.major_gaps,
                "excellencies": result.excellencies,
                "improvement_strategy": result.improvement_strategy,
                "detailed_analysis": result.detailed_analysis,
                "referenced_policies": [
                    {
                        "title": policy.get('title', 'Unknown'),
                        "country": policy.get('country', 'Unknown'),
                        "text_preview": policy.get('text', '')[:300] + "..." if policy.get('text') and len(policy.get('text', '')) > 300 else policy.get('text', ''),
                        "chunk_index": policy.get('chunk_index', 0),
                        "relevance_score": policy.get('score', 0),
                        "metadata": policy.get('metadata', {})
                    }
                    for policy in result.referenced_policies
                ]
            }
            
            # Convert numpy types to JSON-serializable types
            serializable_data = checker._convert_to_json_serializable(results_data)
            
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Detailed data saved to: {json_filename}")
            print(f"\n📊 Summary:")
            print(f"   • Text Report: {txt_filename} (Human-readable)")
            print(f"   • JSON Data: {json_filename} (Machine-readable)")
            print(f"   • Policies Referenced: {len(result.referenced_policies)} segments")
            print(f"   • Analysis Depth: {len(result.major_gaps)} gaps, {len(result.excellencies)} strengths")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Main execution error: {e}", exc_info=True)

if __name__ == "__main__":
    # Set up Together AI API key
    if 'TOGETHER_API_KEY' not in os.environ:
        print("🔑 API Configuration Required")
        api_key = input("Enter your Together AI API key (or set TOGETHER_API_KEY env var): ").strip()
        if api_key:
            os.environ['TOGETHER_API_KEY'] = api_key
    
    main()
