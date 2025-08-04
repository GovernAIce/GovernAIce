import os
import sys
import logging
import json
import re
import requests
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from ml.config.settings import MLConfig
except ImportError:
    # Fallback if ml module is not available
    class MLConfig:
        TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
        MONGODB_URI = os.getenv('MONGODB_URI')
        DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        LEGAL_EMBEDDING_MODEL = "joelniklaus/legal-xlm-roberta-large"
        DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PolicyChunk:
    """Data class for policy document chunks"""
    chunk_id: str
    text: str
    source_file: str
    chunk_index: int
    metadata: Dict[str, Any]
    embedding: List[float] = None

@dataclass
class ComparisonResult:
    """Data class for comparison results"""
    overall_score: float
    risk_level: str
    detailed_scores: Dict[str, float]
    alignment_analysis: Dict[str, Any]
    recommendations: List[str]
    gaps_identified: List[str]

class AIPolicyComparator:
    """
    AI Policy Document Comparison and Risk Assessment Tool
    Uses DeepSeek R1 Distilled LLaMA 70B via Together AI
    """
    
    def __init__(self, together_api_key: str, embedding_model: str = None):
        """
        Initialize the policy comparator
        
        Args:
            together_api_key: Together AI API key for DeepSeek model
            embedding_model: Sentence transformer model for embeddings
        """
        self.together_api_key = together_api_key or MLConfig.TOGETHER_API_KEY
        self.base_url = "https://api.together.xyz/v1"
        self.model_name = MLConfig.DEEPSEEK_MODEL
        
        # Initialize embedding model (will be set later based on reference embeddings)
        self.embedding_model_name = embedding_model or MLConfig.DEFAULT_EMBEDDING_MODEL
        self.embedding_model = None
        
        # Country-specific framework mapping
        self.country_frameworks = {
            "usa": ["usa", "nist", "united states", "america", "us-"],
            "singapore": ["singapore", "sg-", "singapo"],
            "india": ["india", "indian", "in-", "bharat"],
            "eu": ["eu", "europe", "european", "gdpr"],
            "uk": ["uk", "united kingdom", "britain", "british"],
            "canada": ["canada", "canadian", "ca-"],
            "china": ["china", "chinese", "cn-"],
            "japan": ["japan", "japanese", "jp-"],
            "australia": ["australia", "australian", "au-"],
            "korea": ["korea", "korean", "kr-", "south korea"],
            "germany": ["germany", "german", "de-"],
            "france": ["france", "french", "fr-"]
        }
        
        # Core AI governance principles for evaluation
        self.evaluation_criteria = {
            "transparency": {
                "keywords": ["transparent", "explainable", "interpretable", "disclosure", "communication"],
                "weight": 0.15
            },
            "accountability": {
                "keywords": ["accountable", "responsible", "oversight", "governance", "monitoring"],
                "weight": 0.15
            },
            "fairness": {
                "keywords": ["fair", "bias", "discrimination", "equitable", "inclusive"],
                "weight": 0.15
            },
            "safety": {
                "keywords": ["safe", "secure", "risk", "mitigation", "protection", "harm"],
                "weight": 0.15
            },
            "privacy": {
                "keywords": ["privacy", "data protection", "confidential", "consent", "personal data"],
                "weight": 0.10
            },
            "human_oversight": {
                "keywords": ["human oversight", "human control", "human in the loop", "supervision"],
                "weight": 0.10
            },
            "compliance": {
                "keywords": ["compliance", "regulation", "legal", "standards", "certification"],
                "weight": 0.10
            },
            "impact_assessment": {
                "keywords": ["impact assessment", "evaluation", "testing", "validation", "audit"],
                "weight": 0.10
            }
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            "low": 7.5,
            "medium": 5.0,
            "high": 2.5
        }


    def detect_embedding_model(self, reference_embeddings_file: str) -> str:
        """
        Detect the appropriate embedding model based on reference embeddings dimensions
        
        Args:
            reference_embeddings_file: Path to reference embeddings JSON file
            
        Returns:
            Appropriate embedding model name
        """
        try:
            with open(reference_embeddings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                return self.embedding_model_name
            
            # Get embedding dimension from first item
            first_embedding = data[0]["embedding"]
            embedding_dim = len(first_embedding)
            
            # Map dimensions to appropriate models
            dimension_to_model = {
                384: "all-MiniLM-L6-v2",
                768: "all-mpnet-base-v2", 
                512: "all-MiniLM-L12-v2",
                1024: "all-mpnet-base-v2"
            }
            
            model_name = dimension_to_model.get(embedding_dim, "all-MiniLM-L6-v2")
            logger.info(f"Detected embedding dimension: {embedding_dim}, using model: {model_name}")
            return model_name
            
        except Exception as e:
            logger.warning(f"Could not detect embedding model: {e}, using default")
            return self.embedding_model_name

    def initialize_embedding_model(self, model_name: str):
        """Initialize the embedding model"""
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_model_name = model_name
            logger.info(f"Initialized embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing embedding model {model_name}: {e}")
            # Fallback to default
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedding_model_name = "all-MiniLM-L6-v2"

    def get_available_countries(self, reference_chunks: List[PolicyChunk]) -> List[str]:
        """
        Get list of available countries from reference embeddings
        
        Args:
            reference_chunks: List of reference policy chunks
            
        Returns:
            List of detected countries
        """
        detected_countries = set()
        
        for chunk in reference_chunks:
            source_file = chunk.source_file.lower()
            
            for country, keywords in self.country_frameworks.items():
                if any(keyword in source_file for keyword in keywords):
                    detected_countries.add(country.upper())
                    break
        
        return sorted(list(detected_countries))

    def filter_by_country(self, reference_chunks: List[PolicyChunk], target_country: str) -> List[PolicyChunk]:
        """
        Filter reference chunks by specific country
        
        Args:
            reference_chunks: All reference policy chunks
            target_country: Target country to filter by
            
        Returns:
            Filtered list of policy chunks from the specified country
        """
        if not target_country:
            return reference_chunks
        
        target_country = target_country.lower()
        filtered_chunks = []
        
        # Get keywords for the target country
        country_keywords = self.country_frameworks.get(target_country, [target_country])
        
        for chunk in reference_chunks:
            source_file = chunk.source_file.lower()
            
            # Check if source file matches any of the country keywords
            if any(keyword in source_file for keyword in country_keywords):
                filtered_chunks.append(chunk)
        
        logger.info(f"Filtered {len(filtered_chunks)} chunks for country: {target_country.upper()}")
        return filtered_chunks

    def call_deepseek_model(self, prompt: str, max_tokens: int = 2048) -> str:
        """
        Call DeepSeek R1 Distilled LLaMA 70B model via Together AI
        
        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum tokens in response
            
        Returns:
            Model response text
        """
        headers = {
            "Authorization": f"Bearer {self.together_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert AI governance and policy analyst. Analyze AI policy documents with precision and provide detailed, actionable insights."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "top_p": 0.9
        }
        
        try:
            response = requests.post(f"{self.base_url}/chat/completions", 
                                   json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error calling DeepSeek model: {e}")
            return f"Error: {str(e)}"

    def chunk_document(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split document into overlapping chunks
        
        Args:
            text: Document text
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for text chunks
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return np.array([])

    def process_user_document(self, document_text: str, document_name: str = "user_policy") -> List[PolicyChunk]:
        """
        Process user policy document into chunks with embeddings
        
        Args:
            document_text: Raw text of the policy document
            document_name: Name identifier for the document
            
        Returns:
            List of PolicyChunk objects
        """
        logger.info(f"Processing user document: {document_name}")
        
        # Ensure embedding model is initialized
        if self.embedding_model is None:
            logger.info("Initializing default embedding model")
            self.initialize_embedding_model(self.embedding_model_name)
        
        # Clean and preprocess text
        cleaned_text = re.sub(r'\s+', ' ', document_text.strip())
        
        # Create chunks
        text_chunks = self.chunk_document(cleaned_text)
        
        # Create embeddings
        embeddings = self.create_embeddings(text_chunks)
        
        if len(embeddings) == 0:
            raise ValueError("Failed to create embeddings for user document")
        
        # Create PolicyChunk objects
        policy_chunks = []
        for i, (chunk_text, embedding) in enumerate(zip(text_chunks, embeddings)):
            chunk = PolicyChunk(
                chunk_id=f"{document_name}_{i}",
                text=chunk_text,
                source_file=document_name,
                chunk_index=i,
                metadata={
                    "word_count": len(chunk_text.split()),
                    "char_count": len(chunk_text),
                    "created_at": datetime.now().isoformat()
                },
                embedding=embedding.tolist()
            )
            policy_chunks.append(chunk)
        
        logger.info(f"Created {len(policy_chunks)} chunks for user document")
        return policy_chunks

    def load_reference_embeddings(self, embeddings_file: str) -> List[PolicyChunk]:
        """
        Load reference policy embeddings from JSON file
        
        Args:
            embeddings_file: Path to embeddings JSON file
            
        Returns:
            List of PolicyChunk objects
        """
        try:
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                raise ValueError("Embeddings file is empty")
            
            # Initialize embedding model if not already done
            if self.embedding_model is None:
                detected_model = self.detect_embedding_model(embeddings_file)
                self.initialize_embedding_model(detected_model)
            
            policy_chunks = []
            for item in data:
                chunk = PolicyChunk(
                    chunk_id=item["chunk_id"],
                    text=item["text"],
                    source_file=item["source_file"],
                    chunk_index=item["chunk_index"],
                    metadata=item["metadata"],
                    embedding=item["embedding"]
                )
                policy_chunks.append(chunk)
            
            logger.info(f"Loaded {len(policy_chunks)} reference chunks")
            return policy_chunks
            
        except Exception as e:
            logger.error(f"Error loading reference embeddings: {e}")
            return []

    def calculate_semantic_similarity(self, user_chunks: List[PolicyChunk], 
                                    reference_chunks: List[PolicyChunk]) -> Dict[str, Any]:
        """
        Calculate semantic similarity between user and reference documents
        
        Args:
            user_chunks: User document chunks
            reference_chunks: Reference document chunks
            
        Returns:
            Dictionary with similarity analysis
        """
        user_embeddings = np.array([chunk.embedding for chunk in user_chunks])
        ref_embeddings = np.array([chunk.embedding for chunk in reference_chunks])
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(user_embeddings, ref_embeddings)
        
        # Find best matches for each user chunk
        best_matches = []
        for i, user_chunk in enumerate(user_chunks):
            max_sim_idx = np.argmax(similarity_matrix[i])
            max_similarity = similarity_matrix[i][max_sim_idx]
            best_match = reference_chunks[max_sim_idx]
            
            best_matches.append({
                "user_chunk_id": user_chunk.chunk_id,
                "user_text": user_chunk.text[:200] + "...",
                "best_match_chunk_id": best_match.chunk_id,
                "best_match_text": best_match.text[:200] + "...",
                "similarity_score": float(max_similarity),
                "source_framework": best_match.source_file
            })
        
        # Calculate overall similarity metrics
        avg_similarity = np.mean([match["similarity_score"] for match in best_matches])
        max_similarity = max([match["similarity_score"] for match in best_matches])
        min_similarity = min([match["similarity_score"] for match in best_matches])
        
        return {
            "average_similarity": avg_similarity,
            "max_similarity": max_similarity,
            "min_similarity": min_similarity,
            "best_matches": best_matches,
            "total_comparisons": len(best_matches)
        }

    def analyze_criteria_coverage(self, user_chunks: List[PolicyChunk]) -> Dict[str, float]:
        """
        Analyze how well the user document covers key AI governance criteria
        
        Args:
            user_chunks: User document chunks
            
        Returns:
            Dictionary with coverage scores for each criterion
        """
        full_text = " ".join([chunk.text.lower() for chunk in user_chunks])
        
        coverage_scores = {}
        for criterion, details in self.evaluation_criteria.items():
            keyword_matches = 0
            total_keywords = len(details["keywords"])
            
            for keyword in details["keywords"]:
                if keyword.lower() in full_text:
                    keyword_matches += 1
            
            # Calculate coverage score (0-10 scale)
            coverage_score = (keyword_matches / total_keywords) * 10
            coverage_scores[criterion] = coverage_score
        
        return coverage_scores

    def generate_detailed_analysis(self, user_chunks: List[PolicyChunk], 
                                 similarity_analysis: Dict[str, Any],
                                 criteria_coverage: Dict[str, float],
                                 target_country: str = None) -> str:
        """
        Generate detailed analysis using DeepSeek model
        
        Args:
            user_chunks: User document chunks
            similarity_analysis: Similarity analysis results
            criteria_coverage: Criteria coverage scores
            target_country: Target country for comparison context
            
        Returns:
            Detailed analysis text
        """
        # Prepare context for the model
        user_text_sample = "\n".join([chunk.text[:300] for chunk in user_chunks[:3]])
        
        country_context = ""
        if target_country:
            country_context = f"""
        COMPARISON CONTEXT:
        This analysis compares the user's policy specifically against {target_country.upper()} AI governance frameworks and standards.
        Consider {target_country.upper()}-specific regulatory requirements, cultural context, and implementation practices.
        """
        
        prompt = f"""
        Analyze this AI policy document against established AI governance frameworks and provide a comprehensive assessment.

        USER POLICY DOCUMENT SAMPLE:
        {user_text_sample}
        {country_context}
        SIMILARITY ANALYSIS:
        - Average similarity to reference frameworks: {similarity_analysis['average_similarity']:.3f}
        - Best alignment found: {similarity_analysis['max_similarity']:.3f}
        - Weakest alignment: {similarity_analysis['min_similarity']:.3f}

        CRITERIA COVERAGE SCORES (0-10 scale):
        {json.dumps(criteria_coverage, indent=2)}

        Please provide:
        1. STRENGTHS: What aspects of AI governance are well covered?
        2. GAPS: What critical areas are missing or underdeveloped?
        3. ALIGNMENT: How well does this align with {"the " + target_country.upper() + " framework" if target_country else "international frameworks"}?
        4. RECOMMENDATIONS: Specific improvements needed{" for " + target_country.upper() + " compliance" if target_country else ""}
        5. RISK ASSESSMENT: Potential risks from policy gaps{"in the " + target_country.upper() + " context" if target_country else ""}

        Be specific and actionable in your analysis.
        """
        
        return self.call_deepseek_model(prompt, max_tokens=1500)

    def calculate_overall_score(self, similarity_analysis: Dict[str, Any], 
                              criteria_coverage: Dict[str, float]) -> float:
        """
        Calculate overall policy alignment score (0-10)
        
        Args:
            similarity_analysis: Similarity analysis results
            criteria_coverage: Criteria coverage scores
            
        Returns:
            Overall score out of 10
        """
        # Weight semantic similarity (40%) and criteria coverage (60%)
        semantic_score = similarity_analysis['average_similarity'] * 10
        
        # Calculate weighted criteria score
        weighted_criteria_score = 0
        for criterion, score in criteria_coverage.items():
            weight = self.evaluation_criteria[criterion]["weight"]
            weighted_criteria_score += score * weight
        
        # Combine scores
        overall_score = (semantic_score * 0.4) + (weighted_criteria_score * 0.6)
        
        return min(10.0, max(0.0, overall_score))

    def assess_risk_level(self, overall_score: float) -> str:
        """
        Assess risk level based on overall score
        
        Args:
            overall_score: Overall alignment score
            
        Returns:
            Risk level string
        """
        if overall_score >= self.risk_thresholds["low"]:
            return "LOW"
        elif overall_score >= self.risk_thresholds["medium"]:
            return "MEDIUM"
        elif overall_score >= self.risk_thresholds["high"]:
            return "HIGH"
        else:
            return "CRITICAL"

    def compare_policy_document(self, user_document_text: str, 
                              reference_embeddings_file: str,
                              target_country: str = None,
                              document_name: str = "user_policy") -> ComparisonResult:
        """
        Main method to compare user policy document against reference frameworks
        
        Args:
            user_document_text: Text of the user's policy document
            reference_embeddings_file: Path to reference embeddings JSON file
            target_country: Specific country to compare against (e.g., 'usa', 'singapore')
            document_name: Name for the user document
            
        Returns:
            ComparisonResult object with detailed analysis
        """
        logger.info("Starting policy document comparison...")
        
        # Load reference embeddings first to detect correct embedding model
        all_reference_chunks = self.load_reference_embeddings(reference_embeddings_file)
        
        if not all_reference_chunks:
            raise ValueError("Could not load reference embeddings")
        
        # Filter by country if specified
        if target_country:
            reference_chunks = self.filter_by_country(all_reference_chunks, target_country)
            if not reference_chunks:
                available_countries = self.get_available_countries(all_reference_chunks)
                raise ValueError(f"No frameworks found for country '{target_country.upper()}'. "
                               f"Available countries: {', '.join(available_countries)}")
            country_info = f" (comparing against {target_country.upper()} frameworks)"
        else:
            reference_chunks = all_reference_chunks
            available_countries = self.get_available_countries(all_reference_chunks)
            country_info = f" (comparing against all available frameworks: {', '.join(available_countries)})"
        
        logger.info(f"Using {len(reference_chunks)} reference chunks" + country_info)
        
        # Now process user document with the correct embedding model
        user_chunks = self.process_user_document(user_document_text, document_name)
        
        # Verify embedding dimensions match
        if user_chunks and reference_chunks:
            user_dim = len(user_chunks[0].embedding)
            ref_dim = len(reference_chunks[0].embedding)
            if user_dim != ref_dim:
                raise ValueError(f"Embedding dimension mismatch: user={user_dim}, reference={ref_dim}. "
                               f"This usually means the reference embeddings were created with a different model.")
        
        # Calculate semantic similarity
        similarity_analysis = self.calculate_semantic_similarity(user_chunks, reference_chunks)
        
        # Analyze criteria coverage
        criteria_coverage = self.analyze_criteria_coverage(user_chunks)
        
        # Calculate overall score
        overall_score = self.calculate_overall_score(similarity_analysis, criteria_coverage)
        
        # Assess risk level
        risk_level = self.assess_risk_level(overall_score)
        
        # Generate detailed analysis using DeepSeek model
        detailed_analysis = self.generate_detailed_analysis(
            user_chunks, similarity_analysis, criteria_coverage, target_country
        )
        
        # Extract recommendations and gaps from analysis
        recommendations = self.extract_recommendations(detailed_analysis)
        gaps = self.extract_gaps(detailed_analysis)
        
        # Prepare detailed scores
        detailed_scores = {
            "semantic_similarity": similarity_analysis['average_similarity'] * 10,
            "criteria_coverage": criteria_coverage,
            "individual_criteria": {k: v for k, v in criteria_coverage.items()}
        }
        
        result = ComparisonResult(
            overall_score=overall_score,
            risk_level=risk_level,
            detailed_scores=detailed_scores,
            alignment_analysis={
                "similarity_metrics": similarity_analysis,
                "detailed_analysis": detailed_analysis,
                "top_matches": similarity_analysis["best_matches"][:5],
                "target_country": target_country.upper() if target_country else "ALL",
                "frameworks_used": len(reference_chunks)
            },
            recommendations=recommendations,
            gaps_identified=gaps
        )
        
        logger.info(f"Comparison complete. Overall score: {overall_score:.2f}, Risk level: {risk_level}")
        return result

    def extract_recommendations(self, analysis_text: str) -> List[str]:
        """Extract recommendations from analysis text"""
        recommendations = []
        lines = analysis_text.split('\n')
        in_recommendations = False
        
        for line in lines:
            if 'RECOMMENDATIONS' in line.upper():
                in_recommendations = True
                continue
            elif 'RISK ASSESSMENT' in line.upper():
                in_recommendations = False
            elif in_recommendations and line.strip():
                if line.strip().startswith(('-', '•', '*', '1.', '2.', '3.')):
                    recommendations.append(line.strip())
        
        return recommendations[:10]  # Limit to top 10

    def extract_gaps(self, analysis_text: str) -> List[str]:
        """Extract gaps from analysis text"""
        gaps = []
        lines = analysis_text.split('\n')
        in_gaps = False
        
        for line in lines:
            if 'GAPS' in line.upper():
                in_gaps = True
                continue
            elif 'ALIGNMENT' in line.upper():
                in_gaps = False
            elif in_gaps and line.strip():
                if line.strip().startswith(('-', '•', '*', '1.', '2.', '3.')):
                    gaps.append(line.strip())
        
        return gaps[:10]  # Limit to top 10

    def generate_report(self, result: ComparisonResult, output_file: str = None) -> str:
        """
        Generate a comprehensive report of the comparison results
        
        Args:
            result: ComparisonResult object
            output_file: Optional file path to save the report
            
        Returns:
            Report text
        """
        target_country = result.alignment_analysis.get('target_country', 'ALL')
        frameworks_used = result.alignment_analysis.get('frameworks_used', 'N/A')
        
        report = f"""
# AI POLICY DOCUMENT ASSESSMENT REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## EXECUTIVE SUMMARY
- **Overall Score**: {result.overall_score:.2f}/10
- **Risk Level**: {result.risk_level}
- **Comparison Framework**: {target_country} AI Governance Standards
- **Reference Documents Used**: {frameworks_used}
- **Primary Recommendation**: {"Immediate action required" if result.risk_level in ["HIGH", "CRITICAL"] else "Continue monitoring and improvement"}

## DETAILED SCORING

### Overall Metrics
- **Semantic Similarity**: {result.detailed_scores['semantic_similarity']:.2f}/10
- **Policy Alignment Score**: {result.overall_score:.2f}/10
- **Framework Alignment**: {target_country} Standards

### Criteria Coverage Analysis
"""
        
        for criterion, score in result.detailed_scores['criteria_coverage'].items():
            report += f"- **{criterion.replace('_', ' ').title()}**: {score:.2f}/10\n"
        
        country_context = f" against {target_country} standards" if target_country != 'ALL' else " against international standards"
        
        report += f"""

## KEY FINDINGS

### Strengths Identified{country_context}
{chr(10).join([f"- {rec}" for rec in result.recommendations[:5]]) if result.recommendations else "- No specific strengths identified"}

### Critical Gaps{country_context}
{chr(10).join([f"- {gap}" for gap in result.gaps_identified[:5]]) if result.gaps_identified else "- No critical gaps identified"}

### Top Framework Alignments
"""
        
        for match in result.alignment_analysis['top_matches'][:3]:
            report += f"- **{match['source_framework']}**: {match['similarity_score']:.3f} similarity\n"
        
        report += f"""

## RECOMMENDATIONS FOR IMPROVEMENT
{chr(10).join([f"{i+1}. {rec}" for i, rec in enumerate(result.recommendations)]) if result.recommendations else "No specific recommendations at this time."}

## RISK MITIGATION PRIORITIES
Based on the {result.risk_level} risk assessment{country_context}:
"""
        
        if result.risk_level == "CRITICAL":
            report += "- Immediate comprehensive policy revision required\n- Engage AI governance experts\n- Implement interim risk controls"
        elif result.risk_level == "HIGH":
            report += "- Priority review of critical gaps\n- Develop action plan within 30 days\n- Consider external consultation"
        elif result.risk_level == "MEDIUM":
            report += "- Address identified gaps systematically\n- Regular monitoring and updates\n- Stakeholder engagement recommended"
        else:
            report += "- Continue current practices\n- Monitor for emerging requirements\n- Periodic review sufficient"
        
        if target_country != 'ALL':
            report += f"\n- Ensure compliance with {target_country} specific regulations\n- Monitor {target_country} policy updates"
        
        report += f"""

## DETAILED ANALYSIS
{result.alignment_analysis['detailed_analysis']}

---
*Report generated using DeepSeek R1 Distilled LLaMA 70B via Together AI*
*Comparison against {target_country} AI governance frameworks*
"""
        
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"Report saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving report: {e}")
        
        return report


def main():
    """Interactive AI Policy Assessment Tool"""
    
    print("="*60)
    print("AI POLICY DOCUMENT ASSESSMENT TOOL")
    print("="*60)
    print("This tool compares your AI policy against established frameworks")
    print("and provides a risk assessment with actionable recommendations.\n")
    
    # Get API key
    api_key = input("Enter your Together AI API key: ").strip()
    if not api_key:
        print("Error: API key is required")
        return
    
    # Get user document
    print("\nDocument Input Options:")
    print("1. Enter file path to policy document")
    print("2. Paste policy text directly")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        doc_path = input("Enter path to your policy document: ").strip()
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                user_document = f.read()
            document_name = os.path.basename(doc_path).replace('.txt', '').replace('.md', '')
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    elif choice == "2":
        print("Paste your policy document text (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "" and len(lines) > 0 and lines[-1] == "":
                break
            lines.append(line)
        user_document = '\n'.join(lines[:-1])  # Remove last empty line
        document_name = input("Enter a name for this policy document: ").strip() or "user_policy"
    else:
        print("Invalid choice")
        return
    
    if not user_document.strip():
        print("Error: No document text provided")
        return
    
    # Get embeddings file path
    embeddings_file = input("\nEnter path to reference embeddings JSON file: ").strip()
    if not os.path.exists(embeddings_file):
        print(f"Error: Embeddings file not found at {embeddings_file}")
        return
    
    # Get target country for comparison
    print("\nCountry-Specific Comparison:")
    print("Enter a country to compare against its specific AI frameworks")
    print("(e.g., USA, Singapore, India, EU, UK, Canada, etc.)")
    print("Leave blank to compare against all available frameworks")
    
    # Show available countries first
    try:
        temp_comparator = AIPolicyComparator("temp")
        temp_comparator.embedding_model = None  # Don't initialize embedding model yet
        temp_chunks = []
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data[:5]:  # Just check first few items
                temp_chunks.append(PolicyChunk(
                    chunk_id=item["chunk_id"],
                    text=item["text"],
                    source_file=item["source_file"],
                    chunk_index=item["chunk_index"],
                    metadata=item["metadata"],
                    embedding=item["embedding"]
                ))
        if temp_chunks:
            available_countries = temp_comparator.get_available_countries(temp_chunks)
            print(f"Available countries in your embeddings: {', '.join(available_countries)}")
    except:
        pass  # If we can't check, continue anyway
    
    target_country = input("Country (optional): ").strip()
    
    # Initialize comparator and run assessment
    try:
        print("\nInitializing AI Policy Comparator...")
        comparator = AIPolicyComparator(api_key)
        
        # Quick check of available countries
        if target_country:
            print("Checking available frameworks...")
            all_chunks = comparator.load_reference_embeddings(embeddings_file)
            available_countries = comparator.get_available_countries(all_chunks)
            
            if target_country.upper() not in available_countries:
                print(f"⚠️  Warning: '{target_country.upper()}' not found in available frameworks.")
                print(f"Available countries: {', '.join(available_countries)}")
                use_anyway = input("Continue with all frameworks? (y/n): ").strip().lower()
                if use_anyway != 'y':
                    return
                target_country = None
        
        print("Starting policy assessment... This may take a few minutes.")
        print("Processing document and comparing against reference frameworks...")
        
        result = comparator.compare_policy_document(
            user_document_text=user_document,
            reference_embeddings_file=embeddings_file,
            target_country=target_country,
            document_name=document_name
        )
        
        # Display results
        print("\n" + "="*60)
        print("ASSESSMENT RESULTS")
        print("="*60)
        print(f"Document: {document_name}")
        if target_country:
            print(f"Compared against: {target_country.upper()} AI frameworks")
        else:
            print(f"Compared against: All available frameworks")
        print(f"Frameworks used: {result.alignment_analysis.get('frameworks_used', 'N/A')} reference documents")
        print(f"Overall Score: {result.overall_score:.2f}/10")
        print(f"Risk Level: {result.risk_level}")
        print(f"Semantic Similarity: {result.detailed_scores['semantic_similarity']:.2f}/10")
        
        print(f"\n📊 CRITERIA COVERAGE SCORES:")
        for criterion, score in result.detailed_scores['criteria_coverage'].items():
            status = "✅" if score >= 7 else "⚠️" if score >= 4 else "❌"
            print(f"   {status} {criterion.replace('_', ' ').title()}: {score:.1f}/10")
        
        if result.recommendations:
            print(f"\n💡 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(result.recommendations[:5], 1):
                print(f"   {i}. {rec}")
        
        if result.gaps_identified:
            print(f"\n⚠️  CRITICAL GAPS IDENTIFIED:")
            for i, gap in enumerate(result.gaps_identified[:5], 1):
                print(f"   {i}. {gap}")
        
        print(f"\n🎯 RISK ASSESSMENT:")
        if result.risk_level == "CRITICAL":
            print("   🔴 CRITICAL RISK - Immediate comprehensive policy revision required")
        elif result.risk_level == "HIGH":
            print("   🟠 HIGH RISK - Priority review and action plan needed within 30 days")
        elif result.risk_level == "MEDIUM":
            print("   🟡 MEDIUM RISK - Address identified gaps systematically")
        else:
            print("   🟢 LOW RISK - Continue current practices with periodic review")
        
        # Generate and save report
        save_report = input(f"\nSave detailed report? (y/n): ").strip().lower()
        if save_report == 'y':
            report_filename = f"{document_name}_assessment_report.md"
            report = comparator.generate_report(result, report_filename)
            print(f"✅ Detailed report saved to: {report_filename}")
            
            # Save JSON results
            json_filename = f"{document_name}_results.json"
            results_data = {
                "timestamp": datetime.now().isoformat(),
                "document_name": document_name,
                "target_country": target_country.upper() if target_country else "ALL",
                "frameworks_used": result.alignment_analysis.get('frameworks_used', 'N/A'),
                "overall_score": result.overall_score,
                "risk_level": result.risk_level,
                "detailed_scores": result.detailed_scores,
                "recommendations": result.recommendations,
                "gaps_identified": result.gaps_identified
            }
            
            with open(json_filename, "w", encoding='utf-8') as f:
                json.dump(results_data, f, indent=2)
            print(f"✅ JSON results saved to: {json_filename}")
        
        print(f"\n🎉 Assessment completed successfully!")
        
    except ValueError as e:
        if "dimension mismatch" in str(e).lower():
            print(f"\n❌ Embedding Dimension Error: {e}")
            print("\n💡 This usually happens when your reference embeddings were created with a different model.")
            print("   Solutions:")
            print("   1. Recreate your reference embeddings with the current model")
            print("   2. Or check which embedding model was used for your reference file")
        else:
            print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Error during assessment: {e}")
        print("Please check your API key, internet connection, and file paths.")

if __name__ == "__main__":
    main()
