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
uri = "mongodb+srv://smaranrbbtech22:aimd0MyDfx25MmSdL@govai-xlm-r-v1.xx5wl1d.mongodb.net/?retryWrites=true&w=majority&appName=govai-xlm-r-v1"
client = MongoClient(uri)
db = client["govai-xlm-r-v2"]
collection = db["global_chunks"]

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

@dataclass
class MultiAnalysisResult:
    """Data class for multi-country/multi-document analysis results"""
    country_scores: Dict[str, float]  # country -> overall score
    document_scores: Dict[str, float]  # document -> overall score
    detailed_results: Dict[str, Dict[str, ComplianceResult]]  # country -> document -> result
    summary_stats: Dict[str, Any]
    combined_recommendations: Dict[str, Any]

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
        
        # Initialize Together AI client (keeping for backward compatibility)
        if together_api_key:
            self.client = Together(api_key=together_api_key)
        else:
            try:
                self.client = Together()
            except:
                self.client = None
        
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
        
        # Available countries (expanded with ALL option)
        self.base_countries = ["USA", "UK", "EU", "SINGAPORE", "CANADA", "AUSTRALIA", "JAPAN", "CHINA"]
        self.available_countries = ["ALL"] + self.base_countries
        
        # Country name mapping for MongoDB (input format -> database format)
        self.country_mapping = {
            "USA": "USA",
            "UK": "UK", 
            "EU": "EU",
            "SINGAPORE": "Singapore",
            "CANADA": "Canada",
            "AUSTRALIA": "Australia", 
            "JAPAN": "Japan",
            "CHINA": "China"
        }

    def initialize_embedding_model(self):
        """Initialize the embedding model"""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Initialized embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def process_multiple_documents(self, document_inputs: List[str], input_type: str = "file", combine_documents: bool = False) -> Dict[str, Any]:
        """
        Process multiple documents for compliance checking
        
        Args:
            document_inputs: List of file paths or text contents
            input_type: "file" or "text"
            combine_documents: Whether to combine documents into one or analyze separately
            
        Returns:
            Dictionary with processed document data
        """
        try:
            if combine_documents:
                # Combine all documents into one
                all_content = []
                document_info = []
                
                for i, doc_input in enumerate(document_inputs):
                    if input_type == "file":
                        if doc_input.endswith('.txt'):
                            with open(doc_input, 'r', encoding='utf-8') as f:
                                content = f.read()
                        elif doc_input.endswith('.json'):
                            with open(doc_input, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                content = str(data)
                        else:
                            raise ValueError("Supported formats: .txt, .json")
                        document_info.append(os.path.basename(doc_input))
                    else:
                        content = doc_input
                        document_info.append(f"Text Document {i+1}")
                    
                    all_content.append(content)
                
                # Combine all content
                combined_content = "\n\n--- DOCUMENT SEPARATOR ---\n\n".join(all_content)
                combined_document_data = self.upload_and_process_document(text_content=combined_content)
                combined_document_data["document_info"] = f"Combined: {', '.join(document_info)}"
                combined_document_data["source_documents"] = document_info
                
                return {"combined": combined_document_data}
            
            else:
                # Process documents separately
                processed_docs = {}
                
                for i, doc_input in enumerate(document_inputs):
                    if input_type == "file":
                        doc_data = self.upload_and_process_document(file_path=doc_input)
                        doc_name = os.path.basename(doc_input)
                    else:
                        doc_data = self.upload_and_process_document(text_content=doc_input)
                        doc_name = f"Text Document {i+1}"
                    
                    doc_data["document_info"] = doc_name
                    processed_docs[doc_name] = doc_data
                
                return processed_docs
                
        except Exception as e:
            logger.error(f"Error processing multiple documents: {e}")
            raise

    def run_multi_country_analysis(self, document_data: Dict[str, Any], countries: List[str] = None) -> MultiAnalysisResult:
        """
        Run compliance analysis across multiple countries
        
        Args:
            document_data: Processed document data (from process_multiple_documents)
            countries: List of countries to analyze against (if None, uses all base countries)
            
        Returns:
            MultiAnalysisResult with comprehensive analysis
        """
        try:
            if countries is None:
                countries = self.base_countries
            
            # If "ALL" is in countries, replace with all base countries
            if "ALL" in countries:
                countries = self.base_countries
            
            print(f"\n{'='*80}")
            print(f"MULTI-COUNTRY COMPLIANCE ANALYSIS")
            print(f"Analyzing against {len(countries)} jurisdictions: {', '.join(countries)}")
            print(f"Documents: {len(document_data)} document(s)")
            print(f"{'='*80}")
            
            detailed_results = {}
            country_scores = {}
            document_scores = defaultdict(list)
            
            # Run analysis for each country
            for country in countries:
                print(f"\n🌍 Analyzing against {country} policies...")
                detailed_results[country] = {}
                country_document_scores = []
                
                # Analyze each document against this country
                for doc_name, doc_data in document_data.items():
                    print(f"  📄 Processing: {doc_name}")
                    
                    # Run single compliance check
                    result = self._run_single_compliance_check(doc_data, country)
                    detailed_results[country][doc_name] = result
                    
                    score = result.overall_score
                    country_document_scores.append(score)
                    document_scores[doc_name].append(score)
                
                # Calculate average score for this country across all documents
                country_scores[country] = np.mean(country_document_scores)
                print(f"  ✅ {country} Average Score: {country_scores[country]:.1f}/10")
            
            # Calculate average scores for each document across all countries
            avg_document_scores = {doc: np.mean(scores) for doc, scores in document_scores.items()}
            
            # Generate summary statistics
            summary_stats = self._generate_summary_stats(detailed_results, country_scores, avg_document_scores)
            
            # Generate combined recommendations
            combined_recommendations = self._generate_combined_recommendations(detailed_results, countries)
            
            result = MultiAnalysisResult(
                country_scores=country_scores,
                document_scores=avg_document_scores,
                detailed_results=detailed_results,
                summary_stats=summary_stats,
                combined_recommendations=combined_recommendations
            )
            
            print(f"\n✅ Multi-country analysis completed!")
            return result
            
        except Exception as e:
            logger.error(f"Error in multi-country analysis: {e}")
            raise

    def _run_single_compliance_check(self, document_data: Dict[str, Any], country: str) -> ComplianceResult:
        """
        Run compliance check for a single document against a single country (internal method)
        """
        try:
            # Retrieve policies
            policy_chunks = self.retrieve_country_policies(country)
            
            # Calculate overall score
            score_analysis = self.calculate_overall_score(document_data, policy_chunks)
            
            # Identify gaps
            major_gaps = self.identify_major_gaps(document_data, policy_chunks, score_analysis)
            
            # Identify excellencies
            excellencies = self.identify_excellencies(document_data, policy_chunks, score_analysis)
            
            # Generate strategy
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
                referenced_policies=policy_chunks
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in single compliance check for {country}: {e}")
            raise

    def _generate_summary_stats(self, detailed_results: Dict, country_scores: Dict, document_scores: Dict) -> Dict[str, Any]:
        """Generate summary statistics for multi-analysis"""
        try:
            all_scores = list(country_scores.values())
            
            # Overall statistics
            stats = {
                "overall_stats": {
                    "highest_scoring_country": max(country_scores.items(), key=lambda x: x[1]),
                    "lowest_scoring_country": min(country_scores.items(), key=lambda x: x[1]),
                    "average_score_across_countries": np.mean(all_scores),
                    "score_variance": np.var(all_scores),
                    "countries_above_7": sum(1 for score in all_scores if score >= 7.0),
                    "countries_below_5": sum(1 for score in all_scores if score < 5.0)
                },
                "document_stats": {
                    "highest_scoring_document": max(document_scores.items(), key=lambda x: x[1]) if document_scores else None,
                    "lowest_scoring_document": min(document_scores.items(), key=lambda x: x[1]) if document_scores else None,
                    "average_score_across_documents": np.mean(list(document_scores.values())) if document_scores else 0
                },
                "risk_analysis": self._analyze_common_risks(detailed_results),
                "excellence_analysis": self._analyze_common_excellencies(detailed_results)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating summary stats: {e}")
            return {}

    def _analyze_common_risks(self, detailed_results: Dict) -> Dict[str, Any]:
        """Analyze common risks across countries and documents"""
        try:
            risk_frequency = defaultdict(int)
            total_analyses = sum(len(docs) for docs in detailed_results.values())
            
            for country_results in detailed_results.values():
                for doc_result in country_results.values():
                    for gap in doc_result.major_gaps:
                        risk_frequency[gap['category']] += 1
            
            # Calculate risk percentages
            common_risks = {
                risk: {
                    "frequency": count,
                    "percentage": (count / total_analyses) * 100,
                    "severity": "HIGH" if (count / total_analyses) > 0.6 else "MEDIUM" if (count / total_analyses) > 0.3 else "LOW"
                }
                for risk, count in risk_frequency.items()
            }
            
            # Sort by frequency
            sorted_risks = dict(sorted(common_risks.items(), key=lambda x: x[1]['frequency'], reverse=True))
            
            return {
                "most_common_risks": list(sorted_risks.keys())[:5],
                "risk_details": sorted_risks,
                "total_unique_risks": len(risk_frequency)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing common risks: {e}")
            return {}

    def _analyze_common_excellencies(self, detailed_results: Dict) -> Dict[str, Any]:
        """Analyze common excellencies across countries and documents"""
        try:
            excellence_frequency = defaultdict(int)
            total_analyses = sum(len(docs) for docs in detailed_results.values())
            
            for country_results in detailed_results.values():
                for doc_result in country_results.values():
                    for excellence in doc_result.excellencies:
                        excellence_frequency[excellence['category']] += 1
            
            # Calculate excellence percentages
            common_excellencies = {
                excellence: {
                    "frequency": count,
                    "percentage": (count / total_analyses) * 100,
                    "strength": "EXCELLENT" if (count / total_analyses) > 0.7 else "GOOD" if (count / total_analyses) > 0.4 else "MODERATE"
                }
                for excellence, count in excellence_frequency.items()
            }
            
            # Sort by frequency
            sorted_excellencies = dict(sorted(common_excellencies.items(), key=lambda x: x[1]['frequency'], reverse=True))
            
            return {
                "most_common_excellencies": list(sorted_excellencies.keys())[:5],
                "excellence_details": sorted_excellencies,
                "total_unique_excellencies": len(excellence_frequency)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing common excellencies: {e}")
            return {}

    def _generate_combined_recommendations(self, detailed_results: Dict, countries: List[str]) -> Dict[str, Any]:
        """Generate combined strategic recommendations across all analyses"""
        try:
            # Collect all gaps and excellencies
            all_gaps = []
            all_excellencies = []
            
            for country_results in detailed_results.values():
                for doc_result in country_results.values():
                    all_gaps.extend(doc_result.major_gaps)
                    all_excellencies.extend(doc_result.excellencies)
            
            # Prioritize gaps by frequency and severity
            gap_priority = defaultdict(lambda: {"count": 0, "high_severity": 0})
            for gap in all_gaps:
                gap_priority[gap['category']]["count"] += 1
                if gap['severity'] == "HIGH":
                    gap_priority[gap['category']]["high_severity"] += 1
            
            # Sort gaps by priority (high severity count, then total count)
            sorted_gaps = sorted(
                gap_priority.items(),
                key=lambda x: (x[1]["high_severity"], x[1]["count"]),
                reverse=True
            )
            
            # Generate strategic recommendations
            strategic_recommendations = {
                "global_priorities": [
                    {
                        "category": gap[0],
                        "total_occurrences": gap[1]["count"],
                        "high_severity_occurrences": gap[1]["high_severity"],
                        "priority_level": "CRITICAL" if gap[1]["high_severity"] > len(countries) // 2 else "HIGH",
                        "recommended_action": f"Implement comprehensive {gap[0].replace('_', ' ')} framework across all jurisdictions"
                    }
                    for gap in sorted_gaps[:5]  # Top 5 priorities
                ],
                "leverage_opportunities": [
                    excellence['category'] for excellence in all_excellencies
                    if excellence['strength_level'] == "EXCELLENT"
                ][:5],
                "jurisdiction_specific_insights": self._generate_jurisdiction_insights(detailed_results),
                "implementation_sequence": self._recommend_implementation_sequence(sorted_gaps, countries)
            }
            
            return strategic_recommendations
            
        except Exception as e:
            logger.error(f"Error generating combined recommendations: {e}")
            return {}

    def _generate_jurisdiction_insights(self, detailed_results: Dict) -> Dict[str, str]:
        """Generate insights specific to each jurisdiction"""
        insights = {}
        
        for country, country_results in detailed_results.items():
            # Calculate average score for this country
            scores = [result.overall_score for result in country_results.values()]
            avg_score = np.mean(scores)
            
            # Analyze this country's pattern
            common_gaps = defaultdict(int)
            common_excellencies = defaultdict(int)
            
            for result in country_results.values():
                for gap in result.major_gaps:
                    common_gaps[gap['category']] += 1
                for excellence in result.excellencies:
                    common_excellencies[excellence['category']] += 1
            
            # Generate insight
            if avg_score >= 7.0:
                insight = f"Strong compliance foundation. Focus on leveraging {list(common_excellencies.keys())[0] if common_excellencies else 'existing strengths'} for competitive advantage."
            elif avg_score >= 5.0:
                main_gap = max(common_gaps.items(), key=lambda x: x[1])[0] if common_gaps else "governance"
                insight = f"Moderate compliance. Priority focus needed on {main_gap.replace('_', ' ')} to meet regulatory expectations."
            else:
                top_gaps = list(common_gaps.keys())[:2]
                insight = f"Significant compliance gaps. Immediate action required on {' and '.join([gap.replace('_', ' ') for gap in top_gaps])}."
            
            insights[country] = insight
        
        return insights

    def _recommend_implementation_sequence(self, sorted_gaps: List, countries: List[str]) -> List[Dict[str, Any]]:
        """Recommend implementation sequence for addressing gaps"""
        sequence = []
        
        # Phase 1: Critical gaps (0-3 months)
        critical_gaps = [gap for gap in sorted_gaps if gap[1]["high_severity"] > len(countries) // 2][:3]
        if critical_gaps:
            sequence.append({
                "phase": "Phase 1 (0-3 months)",
                "priority": "CRITICAL",
                "focus_areas": [gap[0].replace('_', ' ').title() for gap in critical_gaps],
                "description": "Address critical compliance gaps that appear across multiple jurisdictions"
            })
        
        # Phase 2: High-frequency gaps (3-6 months)
        high_freq_gaps = [gap for gap in sorted_gaps[len(critical_gaps):] if gap[1]["count"] > len(countries) // 2][:3]
        if high_freq_gaps:
            sequence.append({
                "phase": "Phase 2 (3-6 months)",
                "priority": "HIGH",
                "focus_areas": [gap[0].replace('_', ' ').title() for gap in high_freq_gaps],
                "description": "Enhance frameworks for commonly identified gaps"
            })
        
        # Phase 3: Optimization (6-12 months)
        sequence.append({
            "phase": "Phase 3 (6-12 months)",
            "priority": "OPTIMIZATION",
            "focus_areas": ["Continuous monitoring", "Excellence expansion", "Competitive positioning"],
            "description": "Optimize compliance programs and leverage strengths for market advantage"
        })
        
        return sequence

    # Keep all existing methods (upload_and_process_document, retrieve_country_policies, etc.)
    # [Previous methods remain unchanged - including all helper methods]

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
            if country not in self.base_countries:
                raise ValueError(f"Country {country} not available. Available: {', '.join(self.base_countries)}")
            
            # Map country name to database format
            db_country = self.country_mapping.get(country, country)
            logger.info(f"Searching for country: {country} -> mapped to: {db_country}")
            
            # Default search terms if none provided
            if not relevant_terms:
                relevant_terms = [
                    "AI governance", "artificial intelligence regulation", "data protection",
                    "algorithmic accountability", "AI safety", "machine learning compliance"
                ]
            
            all_policy_chunks = []
            
            # Search for each relevant term using mapped country name
            for term in relevant_terms:
                results = search_global_chunks(query=term, country=db_country, k=50)
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
            
            logger.info(f"Retrieved {len(unique_chunks)} unique policy chunks for {db_country}")
            return unique_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving policies for {country}: {e}")
            raise

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
            
            # FIXED: Handle empty policy chunks
            if not policy_chunks:
                logger.warning("No policy chunks found for comparison. Using keyword-based analysis only.")
                
                # Calculate coverage scores for each risk category using keyword analysis only
                category_scores = {}
                for category, details in self.risk_categories.items():
                    category_score = self._calculate_category_compliance_keywords_only(
                        document_data["content"], details["keywords"]
                    )
                    category_scores[category] = category_score
                
                # Calculate weighted overall score
                weighted_score = sum(
                    score * self.risk_categories[category]["weight"] 
                    for category, score in category_scores.items()
                ) * 10  # Scale to 0-10
                
                # No semantic score available
                semantic_score = 0.0
                
                # Use only weighted score when no policies available
                overall_score = weighted_score
                
                score_analysis = {
                    "overall_compliance_score": float(overall_score),
                    "weighted_category_score": float(weighted_score),
                    "semantic_alignment_score": float(semantic_score),
                    "category_breakdown": {k: float(v) for k, v in category_scores.items()},
                    "similarity_matrix_stats": {
                        "max_similarity": 0.0,
                        "avg_similarity": 0.0,
                        "coverage_percentage": 0.0
                    },
                    "compliance_level": self._interpret_compliance_level(overall_score),
                    "analysis_limitation": "Limited to keyword analysis - no policy documents found for comparison"
                }
                
                logger.info(f"Overall compliance score calculated (keyword-only): {overall_score:.2f}/10")
                return score_analysis
            
            # Original logic when policy chunks are available
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
                        
                        # FIXED: Only search policy chunks if they exist
                        if policy_chunks:
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
                        else:
                            # Add generic requirement when no policies available
                            policy_requirements.append({
                                "keyword": keyword,
                                "requirement": f"Consider implementing {keyword} measures as part of AI governance best practices",
                                "policy_source": "General AI Governance Guidelines",
                                "country": "Best Practices"
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
                    
                    # FIXED: Find matching policy requirements only if policies exist
                    policy_matches = []
                    if policy_chunks:
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
                    else:
                        # Add placeholder when no policies available
                        for keyword in covered_keywords:
                            policy_matches.append({
                                "keyword": keyword,
                                "policy_text": f"Document demonstrates {keyword} awareness through implementation",
                                "policy_source": "Inferred Best Practice",
                                "similarity_score": 0.8  # Assume good practice when keyword is present
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

    # All helper methods remain the same...
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

    def _calculate_category_compliance_keywords_only(self, document_text: str, keywords: List[str]) -> float:
        """Calculate compliance score based only on keyword presence (when no policies available)"""
        doc_lower = document_text.lower()
        keyword_coverage = sum(1 for keyword in keywords if keyword.lower() in doc_lower) / len(keywords)
        return float(keyword_coverage)

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
            from google import genai
            
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
            
            client = genai.Client()
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            return response.text
        except Exception as e:
            print(f"Error generating AI insights: {e}")
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

    def display_multi_analysis_summary(self, result: MultiAnalysisResult) -> str:
        """
        Generate comprehensive summary display for multi-analysis results
        """
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("MULTI-COUNTRY AI COMPLIANCE ANALYSIS SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Countries Analyzed: {len(result.country_scores)}")
        lines.append(f"Documents Analyzed: {len(result.document_scores)}")
        lines.append("")
        
        # Country Scores Summary (at the top as requested)
        lines.append("📊 COUNTRY COMPLIANCE SCORES")
        lines.append("-" * 50)
        
        # Sort countries by score (highest first)
        sorted_countries = sorted(result.country_scores.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (country, score) in enumerate(sorted_countries, 1):
            compliance_level = self._interpret_compliance_level(score)
            if score >= 7.0:
                emoji = "🟢"
            elif score >= 5.0:
                emoji = "🟡"
            else:
                emoji = "🔴"
            
            lines.append(f"{rank}. {emoji} {country:12} {score:5.1f}/10 ({compliance_level})")
        
        lines.append("")
        
        # Document Scores Summary (if multiple documents)
        if len(result.document_scores) > 1:
            lines.append("📄 DOCUMENT PERFORMANCE ACROSS COUNTRIES")
            lines.append("-" * 50)
            
            sorted_docs = sorted(result.document_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc, score) in enumerate(sorted_docs, 1):
                compliance_level = self._interpret_compliance_level(score)
                lines.append(f"{rank}. {doc:30} {score:5.1f}/10 ({compliance_level})")
            lines.append("")
        
        # Overall Statistics
        lines.append("📈 OVERALL STATISTICS")
        lines.append("-" * 50)
        stats = result.summary_stats.get("overall_stats", {})
        
        best_country = stats.get("highest_scoring_country", ("N/A", 0))
        worst_country = stats.get("lowest_scoring_country", ("N/A", 0))
        avg_score = stats.get("average_score_across_countries", 0)
        
        lines.append(f"Best Performing Country:     {best_country[0]} ({best_country[1]:.1f}/10)")
        lines.append(f"Lowest Performing Country:   {worst_country[0]} ({worst_country[1]:.1f}/10)")
        lines.append(f"Average Score:               {avg_score:.1f}/10")
        lines.append(f"Countries Above 7.0:         {stats.get('countries_above_7', 0)}")
        lines.append(f"Countries Below 5.0:         {stats.get('countries_below_5', 0)}")
        lines.append("")
        
        # Risk Analysis
        lines.append("🚨 COMMON RISK PATTERNS")
        lines.append("-" * 50)
        risk_analysis = result.summary_stats.get("risk_analysis", {})
        common_risks = risk_analysis.get("most_common_risks", [])
        
        if common_risks:
            lines.append("Most Frequent Compliance Gaps:")
            for i, risk in enumerate(common_risks[:5], 1):
                risk_details = risk_analysis.get("risk_details", {}).get(risk, {})
                frequency = risk_details.get("frequency", 0)
                percentage = risk_details.get("percentage", 0)
                lines.append(f"{i}. {risk.replace('_', ' ').title()} - {frequency} occurrences ({percentage:.0f}%)")
        else:
            lines.append("No common risk patterns identified across analyses.")
        lines.append("")
        
        # Excellence Analysis
        lines.append("✅ COMMON EXCELLENCE PATTERNS")
        lines.append("-" * 50)
        excellence_analysis = result.summary_stats.get("excellence_analysis", {})
        common_excellencies = excellence_analysis.get("most_common_excellencies", [])
        
        if common_excellencies:
            lines.append("Most Frequent Compliance Strengths:")
            for i, excellence in enumerate(common_excellencies[:5], 1):
                exc_details = excellence_analysis.get("excellence_details", {}).get(excellence, {})
                frequency = exc_details.get("frequency", 0)
                percentage = exc_details.get("percentage", 0)
                lines.append(f"{i}. {excellence.replace('_', ' ').title()} - {frequency} occurrences ({percentage:.0f}%)")
        else:
            lines.append("No common excellence patterns identified across analyses.")
        lines.append("")
        
        # Global Recommendations
        lines.append("🎯 GLOBAL STRATEGIC RECOMMENDATIONS")
        lines.append("-" * 50)
        global_priorities = result.combined_recommendations.get("global_priorities", [])
        
        if global_priorities:
            lines.append("Top Global Priorities:")
            for i, priority in enumerate(global_priorities[:3], 1):
                category = priority.get("category", "Unknown")
                total_occ = priority.get("total_occurrences", 0)
                high_sev = priority.get("high_severity_occurrences", 0)
                priority_level = priority.get("priority_level", "MEDIUM")
                
                lines.append(f"{i}. {category} ({priority_level} PRIORITY)")
                lines.append(f"   • Affects {total_occ} analyses ({high_sev} critical)")
                lines.append(f"   • {priority.get('recommended_action', 'Action needed')}")
        lines.append("")
        
        # Implementation Sequence
        lines.append("📅 RECOMMENDED IMPLEMENTATION SEQUENCE")
        lines.append("-" * 50)
        impl_sequence = result.combined_recommendations.get("implementation_sequence", [])
        
        for phase in impl_sequence:
            phase_name = phase.get("phase", "Unknown Phase")
            priority = phase.get("priority", "MEDIUM")
            focus_areas = phase.get("focus_areas", [])
            description = phase.get("description", "")
            
            lines.append(f"{phase_name} - {priority} PRIORITY")
            lines.append(f"Focus: {', '.join(focus_areas)}")
            lines.append(f"Description: {description}")
            lines.append("")
        
        # Jurisdiction-Specific Insights
        lines.append("🌍 JURISDICTION-SPECIFIC INSIGHTS")
        lines.append("-" * 50)
        jurisdiction_insights = result.combined_recommendations.get("jurisdiction_specific_insights", {})
        
        for country, insight in jurisdiction_insights.items():
            lines.append(f"{country}: {insight}")
        lines.append("")
        
        lines.append("=" * 80)
        lines.append("For detailed country-specific analysis, refer to individual reports")
        lines.append("=" * 80)
        
        return "\n".join(lines)

    def generate_multi_country_report(self, result: MultiAnalysisResult, document_info: List[str]) -> str:
        """
        Generate comprehensive multi-country compliance report
        """
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE MULTI-COUNTRY AI COMPLIANCE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Documents Analyzed: {', '.join(document_info)}")
        report_lines.append(f"Countries: {', '.join(result.country_scores.keys())}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("📋 EXECUTIVE SUMMARY")
        report_lines.append("-" * 40)
        
        best_country = max(result.country_scores.items(), key=lambda x: x[1])
        worst_country = min(result.country_scores.items(), key=lambda x: x[1])
        avg_score = np.mean(list(result.country_scores.values()))
        
        report_lines.append(f"Overall Average Compliance Score: {avg_score:.1f}/10")
        report_lines.append(f"Best Performing Jurisdiction: {best_country[0]} ({best_country[1]:.1f}/10)")
        report_lines.append(f"Improvement Opportunity: {worst_country[0]} ({worst_country[1]:.1f}/10)")
        
        if avg_score >= 7.0:
            risk_assessment = "LOW RISK - Strong compliance foundation across jurisdictions"
        elif avg_score >= 5.0:
            risk_assessment = "MODERATE RISK - Some compliance gaps need attention"
        else:
            risk_assessment = "HIGH RISK - Significant compliance improvements needed"
        
        report_lines.append(f"Risk Assessment: {risk_assessment}")
        report_lines.append("")
        
        # Add the detailed summary
        summary_content = self.display_multi_analysis_summary(result)
        report_lines.append(summary_content)
        report_lines.append("")
        
        # Detailed Country Analysis
        report_lines.append("🌍 DETAILED COUNTRY-BY-COUNTRY ANALYSIS")
        report_lines.append("=" * 80)
        
        # Sort countries by score for reporting
        sorted_countries = sorted(result.country_scores.items(), key=lambda x: x[1], reverse=True)
        
        for country, overall_score in sorted_countries:
            report_lines.append(f"\n📍 {country} COMPLIANCE ANALYSIS")
            report_lines.append("-" * 60)
            report_lines.append(f"Overall Score: {overall_score:.1f}/10 ({self._interpret_compliance_level(overall_score)})")
            
            # If multiple documents, show document breakdown for this country
            if len(result.document_scores) > 1:
                report_lines.append("\nDocument Breakdown:")
                country_results = result.detailed_results.get(country, {})
                for doc_name, doc_result in country_results.items():
                    report_lines.append(f"  • {doc_name}: {doc_result.overall_score:.1f}/10")
            
            # Get one representative result for this country (use first document)
            country_results = result.detailed_results.get(country, {})
            if country_results:
                representative_result = list(country_results.values())[0]
                
                # Major Gaps
                report_lines.append(f"\n🚨 Major Gaps ({len(representative_result.major_gaps)}):")
                if representative_result.major_gaps:
                    for i, gap in enumerate(representative_result.major_gaps[:3], 1):  # Top 3
                        report_lines.append(f"{i}. {gap['category']} - {gap['severity']} RISK ({gap['compliance_score']})")
                        report_lines.append(f"   Risk: {gap['risk_description']}")
                else:
                    report_lines.append("   ✅ No major compliance gaps identified")
                
                # Excellencies
                report_lines.append(f"\n✅ Excellencies ({len(representative_result.excellencies)}):")
                if representative_result.excellencies:
                    for i, exc in enumerate(representative_result.excellencies[:3], 1):  # Top 3
                        report_lines.append(f"{i}. {exc['category']} - {exc['strength_level']} ({exc['compliance_score']})")
                        report_lines.append(f"   Advantage: {exc['competitive_advantage']}")
                else:
                    report_lines.append("   ⚠️ Limited excellencies identified - focus on building strengths")
                
                # Strategic Insights for this country
                strategy = representative_result.improvement_strategy
                if strategy.get('strategic_insights'):
                    report_lines.append(f"\n🧠 Strategic Insights for {country}:")
                    report_lines.append(f"{strategy['strategic_insights']}")
            
            report_lines.append("")
        
        # Footer
        report_lines.append("=" * 80)
        report_lines.append("End of Multi-Country Compliance Analysis Report")
        report_lines.append("This report provides strategic guidance for global AI compliance")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)

    # Main orchestration method for single country (backward compatibility)
    def run_compliance_check(self, document_input: str, country: str, input_type: str = "file") -> ComplianceResult:
        """
        Main method to run complete compliance check (single country, single document)
        """
        try:
            print(f"\n{'='*80}")
            print(f"AI POLICY COMPLIANCE CHECKER")
            print(f"Document Analysis vs {country} Policy Framework")
            print(f"{'='*80}")
            
            # Process document
            print("\n🔍 Phase 1: Processing document...")
            if input_type == "file":
                document_data = self.upload_and_process_document(file_path=document_input)
            else:
                document_data = self.upload_and_process_document(text_content=document_input)
            
            # Run single compliance check
            result = self._run_single_compliance_check(document_data, country)
            
            print(f"\n✅ Compliance analysis completed!")
            return result
            
        except Exception as e:
            logger.error(f"Error in compliance check: {e}")
            raise

def main():
    """Enhanced main function with multi-country and multi-document support"""
    print("AI POLICY COMPLIANCE CHECKER - ENHANCED")
    print("=" * 60)
    print("Multi-Country & Multi-Document Analysis Support")
    print("=" * 60)
    
    try:
        # Initialize checker
        checker = AIComplianceChecker()
        
        print(f"\nAvailable countries: {', '.join(checker.available_countries)}")
        print("Note: Select 'ALL' to analyze against all countries")
        
        # Get country selection
        country_input = input("\nSelect country(ies) for policy comparison (or 'ALL'): ").strip().upper()
        
        if country_input == "ALL":
            countries = checker.base_countries
        else:
            # Handle multiple countries (comma-separated)
            countries = [c.strip() for c in country_input.split(",")]
            for country in countries:
                if country not in checker.available_countries and country != "ALL":
                    print(f"Invalid country: {country}. Available: {', '.join(checker.available_countries)}")
                    return
        
        # Get document input method
        input_method = input("\nChoose input method:\n1: Single file\n2: Multiple files\n3: Single text\n4: Multiple texts\nChoice: ").strip()
        
        document_inputs = []
        input_type = ""
        
        if input_method == "1":
            # Single file
            file_path = input("Enter file path: ").strip()
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return
            document_inputs = [file_path]
            input_type = "file"
            
        elif input_method == "2":
            # Multiple files
            print("Enter file paths (one per line, empty line to finish):")
            while True:
                file_path = input().strip()
                if not file_path:
                    break
                if not os.path.exists(file_path):
                    print(f"Warning: File not found: {file_path}")
                    continue
                document_inputs.append(file_path)
            input_type = "file"
            
        elif input_method == "3":
            # Single text
            print("Paste your document text (type 'END' on a new line when done):")
            lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            document_inputs = ["\n".join(lines)]
            input_type = "text"
            
        elif input_method == "4":
            # Multiple texts
            print("Enter multiple documents (type 'NEXT' to start next document, 'END' to finish):")
            current_doc = []
            document_inputs = []
            
            while True:
                line = input()
                if line.strip() == "END":
                    if current_doc:
                        document_inputs.append("\n".join(current_doc))
                    break
                elif line.strip() == "NEXT":
                    if current_doc:
                        document_inputs.append("\n".join(current_doc))
                        current_doc = []
                    print("Starting next document...")
                else:
                    current_doc.append(line)
            input_type = "text"
        
        else:
            print("Invalid choice")
            return
        
        if not document_inputs:
            print("No documents provided")
            return
        
        # Ask about document combination (only for multiple documents)
        combine_documents = False
        if len(document_inputs) > 1:
            combine_choice = input("\nCombine documents into one analysis? (y/n): ").strip().lower()
            combine_documents = (combine_choice == 'y')
        
        # Process documents
        print(f"\n🔍 Processing {len(document_inputs)} document(s)...")
        processed_docs = checker.process_multiple_documents(
            document_inputs, input_type, combine_documents
        )
        
        # Run analysis
        if len(countries) == 1 and len(processed_docs) == 1:
            # Single country, single document - use original method
            country = countries[0]
            doc_data = list(processed_docs.values())[0]
            result = checker._run_single_compliance_check(doc_data, country)
            
            # Display results (original format)
            print(f"\n{'='*80}")
            print(f"COMPLIANCE ANALYSIS RESULTS")
            print(f"{'='*80}")
            
            print(f"\n📊 OVERALL SCORE: {result.overall_score:.1f}/10 ({result.detailed_analysis['compliance_level']})")
            
            if result.detailed_analysis.get("analysis_limitation"):
                print(f"⚠️ {result.detailed_analysis['analysis_limitation']}")
            
            print(f"\n🚨 MAJOR GAPS ({len(result.major_gaps)}):")
            for i, gap in enumerate(result.major_gaps, 1):
                print(f"{i}. {gap['category']} - {gap['severity']} RISK ({gap['compliance_score']})")
            
            print(f"\n✅ EXCELLENCIES ({len(result.excellencies)}):")
            for i, exc in enumerate(result.excellencies, 1):
                print(f"{i}. {exc['category']} - {exc['strength_level']} ({exc['compliance_score']})")
            
            # Save results option
            save_results = input("\nSave detailed results to files? (y/n): ").strip().lower()
            if save_results == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"compliance_analysis_{country}_{timestamp}"
                
                # Determine document info for report
                if input_type == "file":
                    document_info = document_inputs[0]
                else:
                    document_info = f"Text input ({len(document_inputs[0].split())} words)"
                
                # Generate and save text report
                txt_filename = f"{base_filename}.txt"
                text_report = checker.generate_text_report(result, country, document_info)
                
                with open(txt_filename, 'w', encoding='utf-8') as f:
                    f.write(text_report)
                
                print(f"✅ Report saved to: {txt_filename}")
        
        else:
            # Multi-country or multi-document analysis
            print(f"\n🌍 Running multi-country analysis...")
            multi_result = checker.run_multi_country_analysis(processed_docs, countries)
            
            # Display summary
            summary = checker.display_multi_analysis_summary(multi_result)
            print(summary)
            
            # Save results option
            save_results = input("\nSave comprehensive analysis to files? (y/n): ").strip().lower()
            if save_results == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                countries_str = "_".join(countries) if len(countries) <= 3 else f"{len(countries)}countries"
                base_filename = f"multi_compliance_analysis_{countries_str}_{timestamp}"
                
                # Document info
                if input_type == "file":
                    document_info = [os.path.basename(doc) for doc in document_inputs]
                else:
                    document_info = [f"Text Document {i+1}" for i in range(len(document_inputs))]
                
                # Generate comprehensive report
                txt_filename = f"{base_filename}.txt"
                comprehensive_report = checker.generate_multi_country_report(multi_result, document_info)
                
                with open(txt_filename, 'w', encoding='utf-8') as f:
                    f.write(comprehensive_report)
                
                # Save detailed JSON data
                json_filename = f"{base_filename}.json"
                results_data = {
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "countries": countries,
                        "documents": document_info,
                        "analysis_type": "Multi-Country AI Compliance Analysis"
                    },
                    "country_scores": multi_result.country_scores,
                    "document_scores": multi_result.document_scores,
                    "summary_statistics": multi_result.summary_stats,
                    "combined_recommendations": multi_result.combined_recommendations,
                    "detailed_results": {
                        country: {
                            doc: {
                                "overall_score": result.overall_score,
                                "compliance_level": result.detailed_analysis['compliance_level'],
                                "major_gaps": result.major_gaps,
                                "excellencies": result.excellencies,
                                "referenced_policies_count": len(result.referenced_policies)
                            }
                            for doc, result in docs.items()
                        }
                        for country, docs in multi_result.detailed_results.items()
                    }
                }
                
                # Convert to JSON-serializable format
                serializable_data = checker._convert_to_json_serializable(results_data)
                
                with open(json_filename, 'w', encoding='utf-8') as f:
                    json.dump(serializable_data, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Comprehensive report saved to: {txt_filename}")
                print(f"✅ Detailed data saved to: {json_filename}")
                print(f"\n📊 Analysis Summary:")
                print(f"   • Countries Analyzed: {len(countries)}")
                print(f"   • Documents Processed: {len(document_inputs)}")
                print(f"   • Total Analyses: {len(countries) * len(processed_docs)}")
                print(f"   • Average Score: {np.mean(list(multi_result.country_scores.values())):.1f}/10")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Main execution error: {e}", exc_info=True)

if __name__ == "__main__":
    main()