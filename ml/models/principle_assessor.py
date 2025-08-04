import os
import sys
import logging
import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import requests
import datetime

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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedPrinciple:
    """Data class for extracted framework principles"""
    principle_id: str
    title: str
    description: str
    key_requirements: List[str]
    representative_text: str
    importance_weight: float

@dataclass
class PrincipleCompliance:
    """Data class for principle compliance assessment"""
    principle: ExtractedPrinciple
    compliance_status: str  # "FULLY_COMPLIANT", "PARTIALLY_COMPLIANT", "NON_COMPLIANT", "NOT_ADDRESSED"
    compliance_score: float  # 0-10
    evidence_found: List[str]
    gaps_identified: List[str]
    recommendations: List[str]
    logical_analysis: str

@dataclass
class FrameworkAnalysis:
    """Data class for framework analysis"""
    framework_name: str
    extracted_principles: List[ExtractedPrinciple]
    principle_compliance: List[PrincipleCompliance]
    overall_compliance_score: float
    overall_status: str
    critical_gaps: List[str]
    strengths: List[str]

@dataclass
class ComprehensiveAssessment:
    """Data class for comprehensive assessment results"""
    oecd_analysis: FrameworkAnalysis
    nist_analysis: FrameworkAnalysis
    eu_analysis: FrameworkAnalysis
    cross_framework_insights: Dict[str, Any]
    overall_risk_assessment: str
    strategic_recommendations: List[str]
    implementation_roadmap: Dict[str, List[str]]

class PrincipleBasedAIAssessment:
    """
    Principle-Based AI Policy Assessment Tool
    Extracts fundamental principles from frameworks and assesses logical compliance
    """
    
    def __init__(self, together_api_key: str, embedding_model: str = None):
        """Initialize the assessment tool"""
        self.together_api_key = together_api_key or MLConfig.TOGETHER_API_KEY
        self.base_url = "https://api.together.xyz/v1"
        self.model_name = MLConfig.DEEPSEEK_MODEL
        
        self.embedding_model_name = embedding_model or MLConfig.DEFAULT_EMBEDDING_MODEL
        self.embedding_model = None
        
        # Framework identification patterns
        self.framework_patterns = {
            "oecd": ["oecd", "organisation for economic", "council on artificial intelligence"],
            "nist": ["nist", "national institute", "ai risk management", "rmf"],
            "eu": ["eu", "european", "ai act", "artificial intelligence act", "brussels"]
        }


    def initialize_embedding_model(self):
        """Initialize the embedding model"""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Initialized embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def call_deepseek_model(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call DeepSeek model for AI analysis"""
        headers = {
            "Authorization": f"Bearer {self.together_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are an expert AI governance analyst with deep understanding of OECD, NIST, and EU AI frameworks. Provide precise, principle-based analysis."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "top_p": 0.9
        }
        
        try:
            response = requests.post(f"{self.base_url}/chat/completions", 
                                   json=payload, headers=headers, timeout=90)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error calling DeepSeek model: {e}")
            return f"AI analysis unavailable: {str(e)}"

    def detect_framework_type(self, source_file: str) -> str:
        """Detect which framework a document belongs to"""
        source_lower = source_file.lower()
        
        for framework, patterns in self.framework_patterns.items():
            if any(pattern in source_lower for pattern in patterns):
                return framework
        
        return "unknown"

    def load_and_categorize_embeddings(self, embeddings_file: str) -> Dict[str, List[Dict]]:
        """Load embeddings and categorize by framework"""
        try:
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if self.embedding_model is None:
                self.initialize_embedding_model()
            
            categorized_chunks = {"oecd": [], "nist": [], "eu": []}
            
            for item in data:
                framework_type = self.detect_framework_type(item["source_file"])
                
                if framework_type in categorized_chunks:
                    categorized_chunks[framework_type].append(item)
            
            logger.info(f"Categorized chunks - OECD: {len(categorized_chunks['oecd'])}, "
                       f"NIST: {len(categorized_chunks['nist'])}, EU: {len(categorized_chunks['eu'])}")
            
            return categorized_chunks
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return {"oecd": [], "nist": [], "eu": []}

    def extract_framework_principles(self, framework_chunks: List[Dict], 
                                   framework_name: str) -> List[ExtractedPrinciple]:
        """Extract fundamental principles from framework using embeddings clustering and AI analysis"""
        
        if not framework_chunks:
            return []
        
        logger.info(f"Extracting principles from {framework_name} framework...")
        
        # Get embeddings for clustering
        embeddings = np.array([chunk["embedding"] for chunk in framework_chunks])
        texts = [chunk["text"] for chunk in framework_chunks]
        
        # Cluster chunks to identify principle themes (6-7 clusters)
        n_clusters = min(7, len(framework_chunks))
        if n_clusters < 2:
            n_clusters = 2
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Group chunks by cluster
        clustered_chunks = {}
        for i, (chunk, label) in enumerate(zip(framework_chunks, cluster_labels)):
            if label not in clustered_chunks:
                clustered_chunks[label] = []
            clustered_chunks[label].append(chunk)
        
        # Extract principles from each cluster using AI
        extracted_principles = []
        
        for cluster_id, chunks in clustered_chunks.items():
            # Get representative texts from cluster
            cluster_texts = [chunk["text"] for chunk in chunks[:3]]  # Top 3 chunks per cluster
            combined_text = "\n\n".join(cluster_texts)
            
            prompt = f"""
            Analyze this cluster of text from the {framework_name} AI governance framework and extract the fundamental principle it represents.

            FRAMEWORK TEXT CLUSTER:
            {combined_text[:2000]}...

            Extract the core principle in this JSON format:
            {{
                "principle_title": "Clear, concise title of the principle",
                "principle_description": "Detailed description of what this principle means and why it's important",
                "key_requirements": ["requirement1", "requirement2", "requirement3", "requirement4"],
                "importance_weight": 0.X,
                "representative_quote": "Most important quote from the text that embodies this principle"
            }}

            Focus on:
            1. What fundamental governance principle does this text establish?
            2. What are the specific requirements/obligations?
            3. How important is this principle in the overall framework?
            """
            
            try:
                ai_response = self.call_deepseek_model(prompt, max_tokens=800)
                
                # Parse JSON response
                start_idx = ai_response.find('{')
                end_idx = ai_response.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = ai_response[start_idx:end_idx]
                    principle_data = json.loads(json_str)
                    
                    principle = ExtractedPrinciple(
                        principle_id=f"{framework_name.lower()}_principle_{cluster_id}",
                        title=principle_data.get("principle_title", f"Principle {cluster_id}"),
                        description=principle_data.get("principle_description", "Description not available"),
                        key_requirements=principle_data.get("key_requirements", []),
                        representative_text=principle_data.get("representative_quote", combined_text[:200]),
                        importance_weight=principle_data.get("importance_weight", 1.0/n_clusters)
                    )
                    
                    extracted_principles.append(principle)
                    
            except Exception as e:
                logger.error(f"Error extracting principle from cluster {cluster_id}: {e}")
                # Fallback principle
                principle = ExtractedPrinciple(
                    principle_id=f"{framework_name.lower()}_principle_{cluster_id}",
                    title=f"{framework_name} Principle {cluster_id}",
                    description="Principle extracted from framework analysis",
                    key_requirements=["Requirements analysis unavailable"],
                    representative_text=combined_text[:200],
                    importance_weight=1.0/n_clusters
                )
                extracted_principles.append(principle)
        
        logger.info(f"Extracted {len(extracted_principles)} principles from {framework_name}")
        return extracted_principles

    def assess_principle_compliance(self, customer_text: str, 
                                  principle: ExtractedPrinciple,
                                  framework_name: str) -> PrincipleCompliance:
        """Assess customer document compliance with a specific principle"""
        
        prompt = f"""
        Assess whether the customer AI policy document complies with this specific {framework_name} principle.

        PRINCIPLE TO ASSESS:
        Title: {principle.title}
        Description: {principle.description}
        Key Requirements: {', '.join(principle.key_requirements)}

        CUSTOMER POLICY DOCUMENT:
        {customer_text[:3000]}...

        Provide a comprehensive compliance assessment in JSON format:
        {{
            "compliance_status": "FULLY_COMPLIANT/PARTIALLY_COMPLIANT/NON_COMPLIANT/NOT_ADDRESSED",
            "compliance_score": X.X,
            "evidence_found": ["evidence1", "evidence2", "evidence3"],
            "gaps_identified": ["gap1", "gap2", "gap3"],
            "specific_recommendations": ["rec1", "rec2", "rec3"],
            "logical_analysis": "Detailed explanation of how well the customer policy addresses this principle and why"
        }}

        ASSESSMENT CRITERIA:
        - FULLY_COMPLIANT (8-10): All key requirements addressed with clear implementation
        - PARTIALLY_COMPLIANT (5-7): Some requirements addressed but gaps exist
        - NON_COMPLIANT (2-4): Requirements mentioned but inadequately addressed
        - NOT_ADDRESSED (0-1): Principle not mentioned or considered

        Be specific about what evidence you found and what's missing.
        """
        
        try:
            ai_response = self.call_deepseek_model(prompt, max_tokens=1200)
            
            # Parse JSON response
            start_idx = ai_response.find('{')
            end_idx = ai_response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = ai_response[start_idx:end_idx]
                assessment_data = json.loads(json_str)
                
                return PrincipleCompliance(
                    principle=principle,
                    compliance_status=assessment_data.get("compliance_status", "NOT_ADDRESSED"),
                    compliance_score=float(assessment_data.get("compliance_score", 0)),
                    evidence_found=assessment_data.get("evidence_found", []),
                    gaps_identified=assessment_data.get("gaps_identified", []),
                    recommendations=assessment_data.get("specific_recommendations", []),
                    logical_analysis=assessment_data.get("logical_analysis", "Analysis unavailable")
                )
                
        except Exception as e:
            logger.error(f"Error assessing principle compliance: {e}")
        
        # Fallback assessment
        return PrincipleCompliance(
            principle=principle,
            compliance_status="NOT_ASSESSED",
            compliance_score=0.0,
            evidence_found=["Assessment unavailable"],
            gaps_identified=["Could not assess compliance"],
            recommendations=["Manual review required"],
            logical_analysis="Automatic assessment failed"
        )

    def process_customer_document(self, document_text: str) -> str:
        """Process and clean customer document"""
        return re.sub(r'\s+', ' ', document_text.strip())

    def analyze_framework(self, customer_text: str, 
                         framework_chunks: List[Dict], 
                         framework_name: str) -> FrameworkAnalysis:
        """Comprehensive framework analysis"""
        
        logger.info(f"Analyzing {framework_name} framework...")
        
        # Extract fundamental principles
        extracted_principles = self.extract_framework_principles(framework_chunks, framework_name)
        
        # Assess compliance for each principle
        principle_compliance = []
        total_score = 0
        total_weight = 0
        
        for principle in extracted_principles:
            compliance = self.assess_principle_compliance(customer_text, principle, framework_name)
            principle_compliance.append(compliance)
            
            # Weight the score by principle importance
            weighted_score = compliance.compliance_score * principle.importance_weight
            total_score += weighted_score
            total_weight += principle.importance_weight
        
        # Calculate overall compliance
        overall_score = total_score / total_weight if total_weight > 0 else 0
        
        # Determine overall status
        if overall_score >= 8:
            overall_status = "EXCELLENT"
        elif overall_score >= 6:
            overall_status = "GOOD"
        elif overall_score >= 4:
            overall_status = "NEEDS_IMPROVEMENT"
        else:
            overall_status = "CRITICAL"
        
        # Compile critical gaps and strengths
        critical_gaps = []
        strengths = []
        
        for compliance in principle_compliance:
            if compliance.compliance_status in ["NON_COMPLIANT", "NOT_ADDRESSED"]:
                critical_gaps.extend(compliance.gaps_identified[:2])
            elif compliance.compliance_status == "FULLY_COMPLIANT":
                strengths.append(compliance.principle.title)
        
        return FrameworkAnalysis(
            framework_name=framework_name,
            extracted_principles=extracted_principles,
            principle_compliance=principle_compliance,
            overall_compliance_score=overall_score,
            overall_status=overall_status,
            critical_gaps=critical_gaps[:10],
            strengths=strengths[:10]
        )

    def generate_cross_framework_insights(self, oecd_analysis: FrameworkAnalysis,
                                        nist_analysis: FrameworkAnalysis,
                                        eu_analysis: FrameworkAnalysis) -> Dict[str, Any]:
        """Generate insights across all frameworks"""
        
        all_principles = (oecd_analysis.extracted_principles + 
                         nist_analysis.extracted_principles + 
                         eu_analysis.extracted_principles)
        
        prompt = f"""
        Analyze the cross-framework compliance assessment and provide strategic insights:

        OECD ASSESSMENT:
        - Overall Score: {oecd_analysis.overall_compliance_score:.1f}/10
        - Status: {oecd_analysis.overall_status}
        - Principles Assessed: {len(oecd_analysis.extracted_principles)}

        NIST ASSESSMENT:
        - Overall Score: {nist_analysis.overall_compliance_score:.1f}/10
        - Status: {nist_analysis.overall_status}
        - Principles Assessed: {len(nist_analysis.extracted_principles)}

        EU ASSESSMENT:
        - Overall Score: {eu_analysis.overall_compliance_score:.1f}/10
        - Status: {eu_analysis.overall_status}
        - Principles Assessed: {len(eu_analysis.extracted_principles)}

        Provide strategic cross-framework insights in JSON format:
        {{
            "overall_maturity_level": "ADVANCED/INTERMEDIATE/BASIC/INADEQUATE",
            "cross_framework_alignment": "HIGH/MEDIUM/LOW",
            "common_strengths": ["strength1", "strength2", "strength3"],
            "common_gaps": ["gap1", "gap2", "gap3"],
            "framework_conflicts": ["conflict1", "conflict2"],
            "prioritized_actions": ["action1", "action2", "action3"],
            "risk_level": "LOW/MEDIUM/HIGH/CRITICAL",
            "business_impact": "Brief assessment of business implications"
        }}
        """
        
        try:
            ai_response = self.call_deepseek_model(prompt, max_tokens=1000)
            
            start_idx = ai_response.find('{')
            end_idx = ai_response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = ai_response[start_idx:end_idx]
                return json.loads(json_str)
                
        except Exception as e:
            logger.error(f"Error generating cross-framework insights: {e}")
        
        # Fallback insights
        avg_score = (oecd_analysis.overall_compliance_score + 
                    nist_analysis.overall_compliance_score + 
                    eu_analysis.overall_compliance_score) / 3
        
        return {
            "overall_maturity_level": "INTERMEDIATE" if avg_score >= 5 else "BASIC",
            "cross_framework_alignment": "MEDIUM",
            "common_strengths": ["Assessment in progress"],
            "common_gaps": ["Detailed analysis required"],
            "framework_conflicts": [],
            "prioritized_actions": ["Complete principle-based assessment"],
            "risk_level": "MEDIUM",
            "business_impact": "Requires detailed review"
        }

    def create_principle_visualization(self, assessment: ComprehensiveAssessment, 
                                     output_dir: str = "reports"):
        """Create visualizations focused on principle compliance"""
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. OECD Principle Compliance
        self.create_principle_compliance_chart(axes[0, 0], assessment.oecd_analysis, "OECD")
        
        # 2. NIST Principle Compliance
        self.create_principle_compliance_chart(axes[0, 1], assessment.nist_analysis, "NIST")
        
        # 3. EU Principle Compliance
        self.create_principle_compliance_chart(axes[1, 0], assessment.eu_analysis, "EU")
        
        # 4. Cross-Framework Summary
        self.create_cross_framework_summary(axes[1, 1], assessment)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/principle_based_assessment.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Principle-based dashboard saved to {output_dir}/principle_based_assessment.png")

    def create_principle_compliance_chart(self, ax, framework_analysis: FrameworkAnalysis, 
                                        framework_name: str):
        """Create principle compliance bar chart"""
        
        principles = [p.principle.title[:30] + "..." if len(p.principle.title) > 30 else p.principle.title 
                     for p in framework_analysis.principle_compliance]
        scores = [p.compliance_score for p in framework_analysis.principle_compliance]
        statuses = [p.compliance_status for p in framework_analysis.principle_compliance]
        
        # Color mapping for compliance status
        color_map = {
            "FULLY_COMPLIANT": "#2E8B57",      # Green
            "PARTIALLY_COMPLIANT": "#FFD700",   # Yellow
            "NON_COMPLIANT": "#FF6347",         # Red
            "NOT_ADDRESSED": "#808080",         # Gray
            "NOT_ASSESSED": "#DDA0DD"           # Purple
        }
        
        colors = [color_map.get(status, "#808080") for status in statuses]
        
        bars = ax.barh(principles, scores, color=colors)
        ax.set_xlim(0, 10)
        ax.set_xlabel('Compliance Score (0-10)')
        ax.set_title(f'{framework_name} Principle Compliance\nOverall: {framework_analysis.overall_compliance_score:.1f}/10 ({framework_analysis.overall_status})',
                    fontweight='bold')
        
        # Add score labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                   f'{score:.1f}', va='center', fontweight='bold')
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=status.replace('_', ' ')) 
                          for status, color in color_map.items() if status in statuses]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    def create_cross_framework_summary(self, ax, assessment: ComprehensiveAssessment):
        """Create cross-framework summary chart"""
        
        frameworks = ['OECD\nPrinciples', 'NIST\nLifecycle', 'EU\nRisk Levels']
        scores = [
            assessment.oecd_analysis.overall_compliance_score,
            assessment.nist_analysis.overall_compliance_score,
            assessment.eu_analysis.overall_compliance_score
        ]
        
        colors = ['#4A90E2', '#50C878', '#FF9933']
        bars = ax.bar(frameworks, scores, color=colors, alpha=0.8)
        
        ax.set_ylim(0, 10)
        ax.set_ylabel('Overall Compliance Score (0-10)')
        ax.set_title(f'Cross-Framework Principle Compliance\nRisk Level: {assessment.cross_framework_insights.get("risk_level", "UNKNOWN")}',
                    fontweight='bold')
        
        # Add score labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Add average line
        avg_score = sum(scores) / len(scores)
        ax.axhline(y=avg_score, color='red', linestyle='--', alpha=0.7)
        ax.text(0.5, avg_score + 0.2, f'Average: {avg_score:.1f}', 
               transform=ax.get_yaxis_transform(), ha='center', color='red', fontweight='bold')

    def assess_customer_document(self, customer_document_path: str, 
                               embeddings_file: str) -> ComprehensiveAssessment:
        """Main assessment function with principle-based analysis"""
        
        logger.info("Starting principle-based AI governance assessment...")
        
        # Load customer document
        try:
            with open(customer_document_path, 'r', encoding='utf-8') as f:
                customer_text = f.read()
        except Exception as e:
            raise ValueError(f"Error reading customer document: {e}")
        
        customer_text = self.process_customer_document(customer_text)
        
        # Load and categorize reference embeddings
        categorized_chunks = self.load_and_categorize_embeddings(embeddings_file)
        
        # Analyze each framework
        logger.info("Extracting and analyzing OECD principles...")
        oecd_analysis = self.analyze_framework(customer_text, categorized_chunks["oecd"], "OECD")
        
        logger.info("Extracting and analyzing NIST principles...")
        nist_analysis = self.analyze_framework(customer_text, categorized_chunks["nist"], "NIST")
        
        logger.info("Extracting and analyzing EU principles...")
        eu_analysis = self.analyze_framework(customer_text, categorized_chunks["eu"], "EU")
        
        # Generate cross-framework insights
        logger.info("Generating cross-framework strategic insights...")
        cross_framework_insights = self.generate_cross_framework_insights(
            oecd_analysis, nist_analysis, eu_analysis)
        
        # Generate overall assessment
        avg_score = (oecd_analysis.overall_compliance_score + 
                    nist_analysis.overall_compliance_score + 
                    eu_analysis.overall_compliance_score) / 3
        
        if avg_score >= 8:
            overall_risk = "LOW RISK - Excellent principle-based compliance across frameworks"
        elif avg_score >= 6:
            overall_risk = "MEDIUM RISK - Good principle coverage with some gaps"
        elif avg_score >= 4:
            overall_risk = "HIGH RISK - Significant principle gaps identified"
        else:
            overall_risk = "CRITICAL RISK - Major principle deficiencies across frameworks"
        
        # Generate strategic recommendations
        strategic_recommendations = cross_framework_insights.get("prioritized_actions", [])
        
        # Create implementation roadmap
        implementation_roadmap = {
            "immediate": ["Address critical principle gaps", "Establish governance structure"],
            "short_term": ["Implement missing principles", "Create compliance monitoring"],
            "long_term": ["Regular principle-based reviews", "Continuous improvement"]
        }
        
        assessment = ComprehensiveAssessment(
            oecd_analysis=oecd_analysis,
            nist_analysis=nist_analysis,
            eu_analysis=eu_analysis,
            cross_framework_insights=cross_framework_insights,
            overall_risk_assessment=overall_risk,
            strategic_recommendations=strategic_recommendations,
            implementation_roadmap=implementation_roadmap
        )
        
        logger.info("Principle-based assessment completed successfully")
        return assessment

    def generate_comprehensive_report(self, assessment: ComprehensiveAssessment, 
                                    output_file: str = None) -> str:
        """Generate detailed principle-based assessment report"""
        
        report = f"""
# PRINCIPLE-BASED AI GOVERNANCE ASSESSMENT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## EXECUTIVE SUMMARY
**Assessment Methodology**: Principle extraction from embeddings + logical compliance analysis
**Overall Risk Assessment**: {assessment.overall_risk_assessment}
**Cross-Framework Alignment**: {assessment.cross_framework_insights.get('cross_framework_alignment', 'Unknown')}
**Maturity Level**: {assessment.cross_framework_insights.get('overall_maturity_level', 'Unknown')}

### Framework Compliance Scores:
- **OECD Values-based Principles**: {assessment.oecd_analysis.overall_compliance_score:.1f}/10 ({assessment.oecd_analysis.overall_status})
- **NIST AI Lifecycle Framework**: {assessment.nist_analysis.overall_compliance_score:.1f}/10 ({assessment.nist_analysis.overall_status})
- **EU AI Act Requirements**: {assessment.eu_analysis.overall_compliance_score:.1f}/10 ({assessment.eu_analysis.overall_status})

## DETAILED PRINCIPLE ANALYSIS

### OECD VALUES-BASED PRINCIPLES
**Overall Compliance**: {assessment.oecd_analysis.overall_compliance_score:.1f}/10 | **Status**: {assessment.oecd_analysis.overall_status}
**Principles Identified**: {len(assessment.oecd_analysis.extracted_principles)}

"""
        
        # OECD Principle Details
        for i, compliance in enumerate(assessment.oecd_analysis.principle_compliance, 1):
            status_emoji = {
                "FULLY_COMPLIANT": "✅",
                "PARTIALLY_COMPLIANT": "⚠️",
                "NON_COMPLIANT": "❌",
                "NOT_ADDRESSED": "⭕",
                "NOT_ASSESSED": "❓"
            }.get(compliance.compliance_status, "❓")
            
            report += f"""
#### {i}. {compliance.principle.title} {status_emoji}
**Compliance Score**: {compliance.compliance_score:.1f}/10 | **Status**: {compliance.compliance_status.replace('_', ' ')}

**Principle Description**: {compliance.principle.description[:300]}...

**Key Requirements**: {', '.join(compliance.principle.key_requirements[:3])}

**Evidence Found**: {', '.join(compliance.evidence_found[:3]) if compliance.evidence_found else 'None identified'}

**Gaps Identified**: {', '.join(compliance.gaps_identified[:3]) if compliance.gaps_identified else 'None identified'}

**Logical Analysis**: {compliance.logical_analysis[:400]}...

**Recommendations**: {', '.join(compliance.recommendations[:2]) if compliance.recommendations else 'None provided'}

---
"""
        
        report += f"""
### NIST AI LIFECYCLE FRAMEWORK
**Overall Compliance**: {assessment.nist_analysis.overall_compliance_score:.1f}/10 | **Status**: {assessment.nist_analysis.overall_status}
**Principles Identified**: {len(assessment.nist_analysis.extracted_principles)}

"""
        
        # NIST Principle Details
        for i, compliance in enumerate(assessment.nist_analysis.principle_compliance, 1):
            status_emoji = {
                "FULLY_COMPLIANT": "✅",
                "PARTIALLY_COMPLIANT": "⚠️",
                "NON_COMPLIANT": "❌",
                "NOT_ADDRESSED": "⭕",
                "NOT_ASSESSED": "❓"
            }.get(compliance.compliance_status, "❓")
            
            report += f"""
#### {i}. {compliance.principle.title} {status_emoji}
**Compliance Score**: {compliance.compliance_score:.1f}/10 | **Status**: {compliance.compliance_status.replace('_', ' ')}

**Principle Description**: {compliance.principle.description[:300]}...

**Key Requirements**: {', '.join(compliance.principle.key_requirements[:3])}

**Evidence Found**: {', '.join(compliance.evidence_found[:3]) if compliance.evidence_found else 'None identified'}

**Gaps Identified**: {', '.join(compliance.gaps_identified[:3]) if compliance.gaps_identified else 'None identified'}

**Logical Analysis**: {compliance.logical_analysis[:400]}...

**Recommendations**: {', '.join(compliance.recommendations[:2]) if compliance.recommendations else 'None provided'}

---
"""
        
        report += f"""
### EU AI ACT REQUIREMENTS
**Overall Compliance**: {assessment.eu_analysis.overall_compliance_score:.1f}/10 | **Status**: {assessment.eu_analysis.overall_status}
**Principles Identified**: {len(assessment.eu_analysis.extracted_principles)}

"""
        
        # EU Principle Details
        for i, compliance in enumerate(assessment.eu_analysis.principle_compliance, 1):
            status_emoji = {
                "FULLY_COMPLIANT": "✅",
                "PARTIALLY_COMPLIANT": "⚠️",
                "NON_COMPLIANT": "❌",
                "NOT_ADDRESSED": "⭕",
                "NOT_ASSESSED": "❓"
            }.get(compliance.compliance_status, "❓")
            
            report += f"""
#### {i}. {compliance.principle.title} {status_emoji}
**Compliance Score**: {compliance.compliance_score:.1f}/10 | **Status**: {compliance.compliance_status.replace('_', ' ')}

**Principle Description**: {compliance.principle.description[:300]}...

**Key Requirements**: {', '.join(compliance.principle.key_requirements[:3])}

**Evidence Found**: {', '.join(compliance.evidence_found[:3]) if compliance.evidence_found else 'None identified'}

**Gaps Identified**: {', '.join(compliance.gaps_identified[:3]) if compliance.gaps_identified else 'None identified'}

**Logical Analysis**: {compliance.logical_analysis[:400]}...

**Recommendations**: {', '.join(compliance.recommendations[:2]) if compliance.recommendations else 'None provided'}

---
"""
        
        report += f"""
## CROSS-FRAMEWORK STRATEGIC INSIGHTS

### Common Strengths Across Frameworks
{chr(10).join([f"• {strength}" for strength in assessment.cross_framework_insights.get('common_strengths', [])]) if assessment.cross_framework_insights.get('common_strengths') else "• No common strengths identified"}

### Common Gaps Across Frameworks
{chr(10).join([f"• {gap}" for gap in assessment.cross_framework_insights.get('common_gaps', [])]) if assessment.cross_framework_insights.get('common_gaps') else "• No common gaps identified"}

### Framework Conflicts Identified
{chr(10).join([f"• {conflict}" for conflict in assessment.cross_framework_insights.get('framework_conflicts', [])]) if assessment.cross_framework_insights.get('framework_conflicts') else "• No significant conflicts identified"}

### Business Impact Assessment
{assessment.cross_framework_insights.get('business_impact', 'Assessment not available')}

## STRATEGIC RECOMMENDATIONS

### Prioritized Actions
{chr(10).join([f"{i+1}. {action}" for i, action in enumerate(assessment.strategic_recommendations)]) if assessment.strategic_recommendations else "No specific recommendations available"}

## IMPLEMENTATION ROADMAP

### Immediate Actions (0-30 days)
{chr(10).join([f"• {action}" for action in assessment.implementation_roadmap.get('immediate', [])]) if assessment.implementation_roadmap.get('immediate') else "• No immediate actions defined"}

### Short-term Actions (1-6 months)
{chr(10).join([f"• {action}" for action in assessment.implementation_roadmap.get('short_term', [])]) if assessment.implementation_roadmap.get('short_term') else "• No short-term actions defined"}

### Long-term Actions (6+ months)
{chr(10).join([f"• {action}" for action in assessment.implementation_roadmap.get('long_term', [])]) if assessment.implementation_roadmap.get('long_term') else "• No long-term actions defined"}

---
*Assessment conducted using principle extraction from framework embeddings and logical compliance analysis*
*Frameworks analyzed: OECD AI Principles, NIST AI RMF, EU AI Act*
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
    """Interactive Principle-Based AI Assessment Tool"""
    
    print("="*80)
    print("PRINCIPLE-BASED AI GOVERNANCE ASSESSMENT TOOL")
    print("="*80)
    print("🧠 Extracts fundamental principles from frameworks using embeddings")
    print("🔍 Provides logical compliance analysis for each principle")
    print("📊 Assesses principle-by-principle alignment with customer policy\n")
    
    # Get API key
    api_key = input("Enter your Together AI API key: ").strip()
    if not api_key:
        print("Error: API key is required")
        return
    
    # Get customer document
    customer_doc_path = input("Enter path to your AI policy document (.txt): ").strip()
    if not os.path.exists(customer_doc_path):
        print(f"Error: Document not found at {customer_doc_path}")
        return
    
    # Get embeddings file
    embeddings_file = input("Enter path to reference embeddings JSON file: ").strip()
    if not os.path.exists(embeddings_file):
        print(f"Error: Embeddings file not found at {embeddings_file}")
        return
    
    try:
        print("\nInitializing Principle-Based Assessment Tool...")
        print("🧠 Extracting fundamental principles from each framework...")
        print("🔍 Will analyze logical compliance principle-by-principle")
        
        assessor = PrincipleBasedAIAssessment(api_key)
        
        print("\nStarting comprehensive principle extraction and assessment...")
        print("📊 Phase 1: Extracting principles from OECD framework...")
        print("📊 Phase 2: Extracting principles from NIST framework...")
        print("📊 Phase 3: Extracting principles from EU framework...")
        print("🔍 Phase 4: Assessing customer document against each principle...")
        
        assessment = assessor.assess_customer_document(customer_doc_path, embeddings_file)
        
        # Display results
        print("\n" + "="*80)
        print("PRINCIPLE-BASED ASSESSMENT RESULTS")
        print("="*80)
        
        print(f"\n🏛️  OECD VALUES-BASED PRINCIPLES")
        print(f"   Overall Compliance: {assessment.oecd_analysis.overall_compliance_score:.1f}/10")
        print(f"   Status: {assessment.oecd_analysis.overall_status}")
        print(f"   Principles Analyzed: {len(assessment.oecd_analysis.extracted_principles)}")
        
        for i, compliance in enumerate(assessment.oecd_analysis.principle_compliance[:3], 1):
            status_emoji = {"FULLY_COMPLIANT": "✅", "PARTIALLY_COMPLIANT": "⚠️", 
                           "NON_COMPLIANT": "❌", "NOT_ADDRESSED": "⭕"}.get(compliance.compliance_status, "❓")
            print(f"     {i}. {compliance.principle.title[:50]}... {status_emoji} ({compliance.compliance_score:.1f}/10)")
        
        print(f"\n🔬 NIST AI LIFECYCLE FRAMEWORK")
        print(f"   Overall Compliance: {assessment.nist_analysis.overall_compliance_score:.1f}/10")
        print(f"   Status: {assessment.nist_analysis.overall_status}")
        print(f"   Principles Analyzed: {len(assessment.nist_analysis.extracted_principles)}")
        
        for i, compliance in enumerate(assessment.nist_analysis.principle_compliance[:3], 1):
            status_emoji = {"FULLY_COMPLIANT": "✅", "PARTIALLY_COMPLIANT": "⚠️", 
                           "NON_COMPLIANT": "❌", "NOT_ADDRESSED": "⭕"}.get(compliance.compliance_status, "❓")
            print(f"     {i}. {compliance.principle.title[:50]}... {status_emoji} ({compliance.compliance_score:.1f}/10)")
        
        print(f"\n🇪🇺 EU AI ACT REQUIREMENTS")
        print(f"   Overall Compliance: {assessment.eu_analysis.overall_compliance_score:.1f}/10")
        print(f"   Status: {assessment.eu_analysis.overall_status}")
        print(f"   Principles Analyzed: {len(assessment.eu_analysis.extracted_principles)}")
        
        for i, compliance in enumerate(assessment.eu_analysis.principle_compliance[:3], 1):
            status_emoji = {"FULLY_COMPLIANT": "✅", "PARTIALLY_COMPLIANT": "⚠️", 
                           "NON_COMPLIANT": "❌", "NOT_ADDRESSED": "⭕"}.get(compliance.compliance_status, "❓")
            print(f"     {i}. {compliance.principle.title[:50]}... {status_emoji} ({compliance.compliance_score:.1f}/10)")
        
        print(f"\n🎯 OVERALL ASSESSMENT")
        risk_emoji = "🔴" if "CRITICAL" in assessment.overall_risk_assessment else \
                    "🟠" if "HIGH" in assessment.overall_risk_assessment else \
                    "🟡" if "MEDIUM" in assessment.overall_risk_assessment else "🟢"
        print(f"   {risk_emoji} {assessment.overall_risk_assessment}")
        
        print(f"\n🧠 CROSS-FRAMEWORK INSIGHTS")
        print(f"   Maturity Level: {assessment.cross_framework_insights.get('overall_maturity_level', 'Unknown')}")
        print(f"   Framework Alignment: {assessment.cross_framework_insights.get('cross_framework_alignment', 'Unknown')}")
        print(f"   Risk Level: {assessment.cross_framework_insights.get('risk_level', 'Unknown')}")
        
        # Generate outputs
        save_outputs = input(f"\nGenerate detailed principle-based report and visualizations? (y/n): ").strip().lower()
        if save_outputs == 'y':
            base_name = os.path.splitext(os.path.basename(customer_doc_path))[0]
            
            # Create visualizations
            print("📊 Generating principle-based compliance dashboard...")
            assessor.create_principle_visualization(assessment)
            
            # Generate report
            print("📄 Generating comprehensive principle-based report...")
            report_file = f"{base_name}_principle_based_assessment.md"
            assessor.generate_comprehensive_report(assessment, report_file)
            
            # Save JSON results
            json_file = f"{base_name}_principle_assessment_results.json"
            results_data = {
                "timestamp": datetime.now().isoformat(),
                "methodology": "Principle extraction + logical compliance analysis",
                "document_analyzed": customer_doc_path,
                "overall_risk": assessment.overall_risk_assessment,
                "cross_framework_insights": assessment.cross_framework_insights,
                "oecd_analysis": {
                    "overall_score": assessment.oecd_analysis.overall_compliance_score,
                    "status": assessment.oecd_analysis.overall_status,
                    "principles_count": len(assessment.oecd_analysis.extracted_principles),
                    "principle_titles": [p.principle.title for p in assessment.oecd_analysis.principle_compliance]
                },
                "nist_analysis": {
                    "overall_score": assessment.nist_analysis.overall_compliance_score,
                    "status": assessment.nist_analysis.overall_status,
                    "principles_count": len(assessment.nist_analysis.extracted_principles),
                    "principle_titles": [p.principle.title for p in assessment.nist_analysis.principle_compliance]
                },
                "eu_analysis": {
                    "overall_score": assessment.eu_analysis.overall_compliance_score,
                    "status": assessment.eu_analysis.overall_status,
                    "principles_count": len(assessment.eu_analysis.extracted_principles),
                    "principle_titles": [p.principle.title for p in assessment.eu_analysis.principle_compliance]
                },
                "strategic_recommendations": assessment.strategic_recommendations
            }
            
            with open(json_file, "w", encoding='utf-8') as f:
                json.dump(results_data, f, indent=2)
            
            print(f"\n✅ Principle-Based Assessment Outputs Generated:")
            print(f"   📊 Compliance Dashboard: reports/principle_based_assessment.png")
            print(f"   📄 Detailed Report: {report_file}")
            print(f"   📋 JSON Results: {json_file}")
        
        print(f"\n🎉 Principle-based assessment completed!")
        print(f"🧠 Extracted and analyzed {len(assessment.oecd_analysis.extracted_principles) + len(assessment.nist_analysis.extracted_principles) + len(assessment.eu_analysis.extracted_principles)} total principles")
        print(f"🔍 Provided logical compliance analysis for each principle")
        
    except Exception as e:
        print(f"\n❌ Error during assessment: {e}")
        print("Please check your inputs and API key.")

if __name__ == "__main__":
    main()
