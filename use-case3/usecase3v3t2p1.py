import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dataclasses import dataclass
from datetime import datetime
import re
import matplotlib.pyplot as plt
from math import pi
from google import genai

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
class FastPrincipleCompliance:
    """Simplified data class for fast principle compliance assessment"""
    principle: ExtractedPrinciple
    compliance_score: float  # 0-10
    evidence_found: List[str]
    improvement_sources: List[str]  # Specific document sections for improvement
    similarity_scores: List[float]  # Embedding similarity scores
    quick_recommendation: str

@dataclass
class FastFrameworkAnalysis:
    """Simplified framework analysis"""
    framework_name: str
    principle_compliance: List[FastPrincipleCompliance]
    overall_compliance_score: float
    overall_status: str

@dataclass
class FastAssessment:
    """Simplified assessment results"""
    document_name: str
    oecd_analysis: FastFrameworkAnalysis
    nist_analysis: FastFrameworkAnalysis
    eu_analysis: FastFrameworkAnalysis
    overall_score: float
    overall_status: str

class RadarChartGenerator:
    """
    Radar Chart Generator for TXT Assessment Reports
    """
    
    def __init__(self):
        """Initialize the radar chart generator"""
        plt.style.use('default')
        
        # Framework colors
        self.colors = {
            "OECD": "#4A90E2", 
            "NIST": "#50C878", 
            "EU": "#FF9933"
        }
    
    def parse_txt_report(self, txt_file_path: str) -> Dict[str, List[Tuple[str, float]]]:
        """Parse TXT report and extract framework principles and scores"""
        frameworks_data = {
            "OECD": [],
            "NIST": [],
            "EU": []
        }
        
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Parsing TXT report: {txt_file_path}")
            
            # Extract all principles using TXT format patterns
            all_principles = self._extract_principles_from_txt(content)
            
            # Categorize by framework section
            frameworks_data = self._categorize_txt_principles_by_section(content, all_principles)
            
            # Log results
            for framework, principles in frameworks_data.items():
                logger.info(f"FINAL: {framework} has {len(principles)} principles")
                    
        except Exception as e:
            logger.error(f"Error parsing TXT report: {e}")
            
        return frameworks_data
    
    def _extract_principles_from_txt(self, content: str) -> List[Tuple[str, float, int]]:
        """Extract all principles from TXT format"""
        all_principles = []
        
        # Pattern for TXT format: "1. Title [symbol]" followed by "Score: X.X/10"
        # Look for numbered principles with symbols
        principle_pattern = r'(\d+)\.\s*([^[]+?)\s*\[([✓⚠✗])\]'
        score_pattern = r'Score:\s*(\d+\.?\d*)/10'
        
        principle_matches = list(re.finditer(principle_pattern, content))
        score_matches = list(re.finditer(score_pattern, content))
        
        logger.info(f"Found {len(principle_matches)} principle headers in TXT format")
        logger.info(f"Found {len(score_matches)} scores in TXT format")
        
        # Match each principle to its closest following score
        for principle_match in principle_matches:
            principle_num = principle_match.group(1)
            principle_title = principle_match.group(2).strip()
            principle_symbol = principle_match.group(3)
            principle_position = principle_match.start()
            
            # Find the closest score after this principle
            closest_score = None
            min_distance = float('inf')
            
            for score_match in score_matches:
                score_position = score_match.start()
                if score_position > principle_position:
                    distance = score_position - principle_position
                    if distance < min_distance and distance < 1000:  # Reasonable distance
                        min_distance = distance
                        closest_score = float(score_match.group(1))
            
            if closest_score is not None:
                all_principles.append((principle_title, closest_score, principle_position))
                logger.info(f"✅ Matched TXT: '{principle_title}' = {closest_score}/10")
        
        return all_principles
    
    def _categorize_txt_principles_by_section(self, content: str, all_principles: List[Tuple[str, float, int]]) -> Dict[str, List[Tuple[str, float]]]:
        """Categorize principles by framework section for TXT format"""
        frameworks_data = {
            "OECD": [],
            "NIST": [],
            "EU": []
        }
        
        # Find section boundaries for TXT format
        oecd_start = content.find("--- OECD VALUES-BASED PRINCIPLES ---")
        nist_start = content.find("--- NIST AI RISK MANAGEMENT FRAMEWORK ---")
        eu_start = content.find("--- EU AI ACT REQUIREMENTS ---")
        priority_start = content.find("PRIORITY IMPROVEMENT ACTIONS")
        
        logger.info(f"TXT Section positions - OECD: {oecd_start}, NIST: {nist_start}, EU: {eu_start}")
        
        # Categorize principles based on their position
        for title, score, position in all_principles:
            if oecd_start != -1 and nist_start != -1 and oecd_start < position < nist_start:
                frameworks_data["OECD"].append((title, score))
            elif nist_start != -1 and eu_start != -1 and nist_start < position < eu_start:
                frameworks_data["NIST"].append((title, score))
            elif eu_start != -1 and priority_start != -1 and eu_start < position < priority_start:
                frameworks_data["EU"].append((title, score))
            elif eu_start != -1 and priority_start == -1 and position > eu_start:
                frameworks_data["EU"].append((title, score))
        
        return frameworks_data
    
    def create_single_radar_chart(self, principles: List[Tuple[str, float]], 
                                 framework_name: str, 
                                 save_path: str = None) -> plt.Figure:
        """Create a radar chart for a single framework"""
        
        if not principles:
            logger.warning(f"No principles found for {framework_name}")
            return None
        
        if len(principles) == 1:
            return self._create_single_principle_chart(principles[0], framework_name, save_path)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        # Prepare data
        principle_names = [p[0] for p in principles]
        scores = [p[1] for p in principles]
        
        # Number of principles
        N = len(principles)
        
        # Calculate angles for each principle
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add first score to end to complete the circle
        scores += scores[:1]
        
        # Set up the radar chart
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw the plot
        line = ax.plot(angles, scores, 'o-', linewidth=3, markersize=8, 
                      color=self.colors.get(framework_name, '#333333'))
        ax.fill(angles, scores, alpha=0.25, color=self.colors.get(framework_name, '#333333'))
        
        # Customize the chart
        ax.set_xticks(angles[:-1])
        
        # Shorten long principle names
        display_names = []
        for name in principle_names:
            if len(name) > 25:
                words = name.split()
                if len(words) > 3:
                    # Break into 2 lines
                    mid = len(words) // 2
                    display_name = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
                else:
                    display_name = name[:22] + "..."
            else:
                display_name = name
            display_names.append(display_name)
        
        ax.set_xticklabels(display_names, fontsize=9, fontweight='bold')
        
        # Set y-axis (scores)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9, alpha=0.7)
        ax.grid(True, alpha=0.3)
        
        # Calculate average score
        avg_score = sum(scores[:-1]) / len(scores[:-1])
        
        # Determine status
        if avg_score >= 8:
            status = "EXCELLENT"
            status_color = "green"
        elif avg_score >= 6:
            status = "GOOD"
            status_color = "blue"
        elif avg_score >= 4:
            status = "NEEDS_IMPROVEMENT"
            status_color = "orange"
        else:
            status = "CRITICAL"
            status_color = "red"
        
        # Add title
        title = f'{framework_name} Framework Compliance\nAverage Score: {avg_score:.1f}/10 ({status})\n{len(principles)} Principles Assessed'
        ax.set_title(title, size=14, fontweight='bold', pad=30, color=status_color)
        
        # Add score labels for all principles
        for angle, score, name in zip(angles[:-1], scores[:-1], principle_names):
            ax.annotate(f'{score:.1f}', xy=(angle, score), xytext=(angle, score + 0.8),
                       ha='center', va='center', fontweight='bold', fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                logger.info(f"Saved {framework_name} radar chart to {save_path}")
            except Exception as e:
                logger.error(f"Error saving {framework_name} radar chart: {e}")
        
        return fig
    
    def _create_single_principle_chart(self, principle: Tuple[str, float], 
                                     framework_name: str, save_path: str = None) -> plt.Figure:
        """Create a simple chart for frameworks with only 1 principle"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        title, score = principle
        
        # Create a simple bar chart
        bars = ax.bar([framework_name], [score], color=self.colors.get(framework_name, '#333333'), alpha=0.8)
        
        ax.set_ylim(0, 10)
        ax.set_ylabel('Compliance Score (0-10)', fontweight='bold')
        ax.set_title(f'{framework_name} Framework: {title}\nScore: {score:.1f}/10', 
                    fontweight='bold', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        # Add score label
        ax.text(bars[0].get_x() + bars[0].get_width()/2, bars[0].get_height() + 0.1,
               f'{score:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Add threshold lines
        ax.axhline(y=8, color='green', linestyle='--', alpha=0.5, label='Excellent')
        ax.axhline(y=6, color='blue', linestyle='--', alpha=0.5, label='Good') 
        ax.axhline(y=4, color='orange', linestyle='--', alpha=0.5, label='Needs Improvement')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                logger.info(f"Saved {framework_name} single principle chart to {save_path}")
            except Exception as e:
                logger.error(f"Error saving {framework_name} chart: {e}")
        
        return fig
    
    def create_comparison_radar_chart(self, frameworks_data: Dict[str, List[Tuple[str, float]]], 
                                    document_name: str = "",
                                    save_path: str = None) -> plt.Figure:
        """Create a comparison radar chart showing average scores for each framework"""
        
        # Calculate average scores
        framework_averages = {}
        framework_counts = {}
        
        for framework, principles in frameworks_data.items():
            if principles:
                avg_score = sum(score for _, score in principles) / len(principles)
                framework_averages[framework] = avg_score
                framework_counts[framework] = len(principles)
        
        # Only include frameworks with data
        valid_frameworks = {k: v for k, v in framework_averages.items() if framework_counts[k] > 0}
        
        if not valid_frameworks:
            logger.error("No framework data available for comparison")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Prepare data
        frameworks = list(valid_frameworks.keys())
        averages = list(valid_frameworks.values())
        
        # Create angles
        N = len(frameworks)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        averages += averages[:1]
        
        # Plot comparison
        ax.plot(angles, averages, 'o-', linewidth=4, markersize=12, color='#E74C3C')
        ax.fill(angles, averages, alpha=0.3, color='#E74C3C')
        
        # Customize
        ax.set_xticks(angles[:-1])
        
        # Create labels with principle counts
        labels = [f"{framework}\n({framework_counts[framework]} principles)" 
                 for framework in frameworks]
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
        
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=10, alpha=0.7)
        ax.grid(True, alpha=0.3)
        
        # Add score labels
        for angle, avg, framework in zip(angles[:-1], averages[:-1], frameworks):
            ax.text(angle, avg + 0.5, f'{avg:.1f}', ha='center', va='center', 
                   fontweight='bold', fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        # Calculate overall average
        overall_avg = sum(averages[:-1]) / len(averages[:-1])
        
        # Determine status
        if overall_avg >= 8:
            overall_status = "EXCELLENT"
            title_color = "green"
        elif overall_avg >= 6:
            overall_status = "GOOD"
            title_color = "blue"
        elif overall_avg >= 4:
            overall_status = "NEEDS_IMPROVEMENT"
            title_color = "orange"
        else:
            overall_status = "CRITICAL"
            title_color = "red"
        
        # Add title
        doc_display = f"{document_name}\n" if document_name else ""
        title = f'{doc_display}Cross-Framework Compliance Comparison\nOverall Average: {overall_avg:.1f}/10 ({overall_status})'
        ax.set_title(title, size=16, fontweight='bold', pad=30, color=title_color)
        
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                logger.info(f"Saved comparison radar chart to {save_path}")
            except Exception as e:
                logger.error(f"Error saving comparison radar chart: {e}")
        
        return fig
    
    def generate_charts_from_assessment(self, assessment: FastAssessment, 
                                      txt_report_path: str,
                                      output_dir: str = "radar_charts") -> None:
        """Generate radar charts from assessment data and TXT report"""
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info(f"Generating radar charts from TXT report: {txt_report_path}")
        
        # Parse the TXT report
        frameworks_data = self.parse_txt_report(txt_report_path)
        
        # Check if any data was found
        total_principles = sum(len(principles) for principles in frameworks_data.values())
        
        if total_principles == 0:
            logger.warning("No principles extracted from TXT report for charts")
            # Fallback: create charts directly from assessment data
            frameworks_data = self._extract_from_assessment_data(assessment)
        
        # Generate individual charts
        for framework, principles in frameworks_data.items():
            if principles:
                logger.info(f"Creating {framework} radar chart ({len(principles)} principles)")
                chart_file = os.path.join(output_dir, f"{assessment.document_name}_{framework.lower()}_radar.png")
                fig = self.create_single_radar_chart(principles, framework, chart_file)
                if fig:
                    plt.close(fig)
        
        # Generate comparison chart
        if any(frameworks_data.values()):
            logger.info("Creating framework comparison chart")
            comparison_file = os.path.join(output_dir, f"{assessment.document_name}_comparison.png")
            fig = self.create_comparison_radar_chart(frameworks_data, assessment.document_name, comparison_file)
            if fig:
                plt.close(fig)
        
        logger.info(f"Radar charts generated in: {output_dir}")
    
    def _extract_from_assessment_data(self, assessment: FastAssessment) -> Dict[str, List[Tuple[str, float]]]:
        """Fallback: Extract chart data directly from assessment object"""
        
        frameworks_data = {
            "OECD": [(c.principle.title, c.compliance_score) for c in assessment.oecd_analysis.principle_compliance],
            "NIST": [(c.principle.title, c.compliance_score) for c in assessment.nist_analysis.principle_compliance],
            "EU": [(c.principle.title, c.compliance_score) for c in assessment.eu_analysis.principle_compliance]
        }
        
        logger.info("Using fallback chart data from assessment object")
        return frameworks_data

class HybridFastAIAssessment:
    """
    Hybrid Fast AI Assessment Tool - Embeddings + Gemini (30 second execution)
    Uses embeddings for semantic section finding + Gemini for scoring
    """
    
    def __init__(self, api_key: str, embedding_model: str = "all-mpnet-base-v2"):
        """Initialize the hybrid assessment tool"""
        self.api_key = api_key
        os.environ['GEMINI_API_KEY'] = api_key
        self.client = genai.Client()
        self.model_name = "gemini-2.5-pro"
        
        # Initialize embedding model (match your 768-dim embeddings)
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.framework_embeddings = {}  # Cache for framework embeddings
        
        # Directory for saved principles
        self.saved_principles_dir = "saved_principles"
        
        # Framework identification patterns
        self.framework_patterns = {
            "oecd": ["oecd", "organisation for economic", "council on artificial intelligence"],
            "nist": ["nist", "national institute", "ai risk management", "rmf"],
            "eu": ["eu", "european", "ai act", "artificial intelligence act", "brussels"]
        }

    def initialize_embedding_model(self):
        """Initialize embedding model to match your 768-dim embeddings"""
        if self.embedding_model is None:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Initialized embedding model: {self.embedding_model_name}")
                
                # Test embedding dimension
                test_embedding = self.embedding_model.encode(["test"])
                logger.info(f"Embedding model dimension: {test_embedding.shape[1]}")
                
            except Exception as e:
                logger.error(f"Error initializing embedding model {self.embedding_model_name}: {e}")
                # Try fallback models that produce 768 dimensions
                try:
                    self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
                    logger.info("Fallback: Using all-mpnet-base-v2 (768 dimensions)")
                except Exception as e2:
                    logger.error(f"Fallback model also failed: {e2}")
                    self.embedding_model = SentenceTransformer("all-MiniLM-L12-v2")
                    logger.info("Final fallback: Using all-MiniLM-L12-v2")

    def load_framework_embeddings(self, embeddings_file: str):
        """Load and cache framework embeddings for semantic matching"""
        try:
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Categorize embeddings by framework
            for item in data:
                framework_type = self.detect_framework_type(item["source_file"])
                if framework_type not in self.framework_embeddings:
                    self.framework_embeddings[framework_type] = []
                
                self.framework_embeddings[framework_type].append({
                    'text': item['text'],
                    'embedding': np.array(item['embedding']),
                    'source': item['source_file']
                })
            
            logger.info(f"Loaded embeddings - OECD: {len(self.framework_embeddings.get('oecd', []))}, "
                       f"NIST: {len(self.framework_embeddings.get('nist', []))}, "
                       f"EU: {len(self.framework_embeddings.get('eu', []))}")
                       
        except Exception as e:
            logger.error(f"Error loading framework embeddings: {e}")

    def detect_framework_type(self, source_file: str) -> str:
        """Detect which framework a document belongs to"""
        source_lower = source_file.lower()
        
        for framework, patterns in self.framework_patterns.items():
            if any(pattern in source_lower for pattern in patterns):
                return framework
        
        return "unknown"

    def call_gemini_model_fast(self, prompt: str, max_tokens: int = 400) -> str:
        """Fast Gemini model call with improved error handling"""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                if response and response.text:
                    return response.text
                else:
                    logger.warning(f"Empty response from Gemini API (attempt {attempt + 1})")
                    
            except Exception as e:
                logger.error(f"Gemini API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)  # Wait 1 second before retry
                    
        return None  # Return None instead of error string

    def load_extracted_principles(self, framework_name: str) -> List[ExtractedPrinciple]:
        """Load extracted principles from JSON file"""
        filename = f"{self.saved_principles_dir}/{framework_name.lower()}_principles.json"
        if not os.path.exists(filename):
            logger.error(f"Principles file not found: {filename}")
            return []
            
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                principles_data = json.load(f)
            
            principles = []
            for data in principles_data:
                principles.append(ExtractedPrinciple(
                    principle_id=data["principle_id"],
                    title=data["title"],
                    description=data["description"],
                    key_requirements=data["key_requirements"],
                    representative_text=data["representative_text"],
                    importance_weight=data["importance_weight"]
                ))
            
            logger.info(f"Loaded {len(principles)} principles for {framework_name}")
            return principles
            
        except Exception as e:
            logger.error(f"Error loading principles for {framework_name}: {e}")
            return []

    def find_most_relevant_framework_sections(self, customer_text: str, 
                                            principle: ExtractedPrinciple,
                                            framework_name: str) -> Tuple[List[str], List[float]]:
        """Use embeddings to find most relevant FRAMEWORK sections for improvement guidance"""
        
        if self.embedding_model is None:
            self.initialize_embedding_model()
        
        # Create embedding for principle (combine title + description + requirements)
        principle_text = f"{principle.title} {principle.description} {' '.join(principle.key_requirements)}"
        
        try:
            principle_embedding = self.embedding_model.encode([principle_text])
            
            # First, try to find relevant sections from the framework document embeddings
            if framework_name.lower() in self.framework_embeddings:
                framework_chunks = self.framework_embeddings[framework_name.lower()]
                if framework_chunks:
                    # Check dimension compatibility
                    framework_embed_dim = framework_chunks[0]['embedding'].shape[0]
                    principle_embed_dim = principle_embedding.shape[1]
                    
                    if framework_embed_dim != principle_embed_dim:
                        logger.error(f"Dimension mismatch: Framework embeddings ({framework_embed_dim}D) vs Principle embeddings ({principle_embed_dim}D)")
                        return self.find_framework_sections_fallback(principle, framework_name), [0.0]
                    
                    # Calculate similarities between principle and framework chunks
                    framework_embeddings_array = np.array([chunk['embedding'] for chunk in framework_chunks])
                    framework_similarities = cosine_similarity(principle_embedding, framework_embeddings_array)[0]
                    
                    # Get top 3 most relevant framework sections
                    top_indices = np.argsort(framework_similarities)[-3:][::-1]  # Top 3, descending
                    
                    relevant_sections = []
                    similarity_scores = []
                    
                    for idx in top_indices:
                        if framework_similarities[idx] > 0.2:  # Minimum similarity threshold for framework matching
                            chunk = framework_chunks[idx]
                            # Extract source document name for reference
                            source_name = os.path.basename(chunk.get('source', 'Framework Document'))
                            chunk_text = chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text']
                            
                            relevant_sections.append(f"📋 {framework_name} Reference ({source_name}): {chunk_text}")
                            similarity_scores.append(float(framework_similarities[idx]))
                    
                    if relevant_sections:
                        return relevant_sections[:3], similarity_scores[:3]
            
            # Fallback: Create generic framework guidance based on principle
            return self.find_framework_sections_fallback(principle, framework_name), [0.5]
            
        except Exception as e:
            logger.error(f"Error in framework section finding: {e}")
            return self.find_framework_sections_fallback(principle, framework_name), [0.0]

    def find_framework_sections_fallback(self, principle: ExtractedPrinciple, framework_name: str) -> List[str]:
        """Fallback method to provide framework guidance when embeddings fail"""
        
        guidance_sections = []
        
        # Create framework-specific guidance based on principle requirements
        for i, requirement in enumerate(principle.key_requirements[:2], 1):
            guidance_sections.append(
                f"📋 {framework_name} Guidance {i}: To address '{principle.title}', implement: {requirement}"
            )
        
        # Add generic principle description as guidance
        if len(guidance_sections) < 2:
            guidance_sections.append(
                f"📋 {framework_name} Principle: {principle.description[:200]}..."
            )
        
        return guidance_sections[:3]

    def find_relevant_text_sections_fallback(self, customer_text: str, principle: ExtractedPrinciple) -> List[str]:
        """Fallback method using keyword matching in customer document - kept for compatibility"""
        paragraphs = [p.strip() for p in customer_text.split('\n\n') if p.strip()]
        
        # Extract keywords from principle requirements
        keywords = []
        for req in principle.key_requirements:
            words = re.findall(r'\w+', req.lower())
            keywords.extend([w for w in words if len(w) > 3])
        
        # Find paragraphs with keyword matches
        relevant_sections = []
        for i, paragraph in enumerate(paragraphs):
            paragraph_lower = paragraph.lower()
            keyword_matches = sum(1 for keyword in keywords if keyword in paragraph_lower)
            
            if keyword_matches > 0:
                section_ref = f"Your Document Section {i+1}"
                truncated = paragraph[:200] + "..." if len(paragraph) > 200 else paragraph
                relevant_sections.append(f"{section_ref}: {truncated}")
        
        return relevant_sections[:3]

    def hybrid_assess_principle_compliance(self, customer_text: str, 
                                         principle: ExtractedPrinciple,
                                         framework_name: str) -> FastPrincipleCompliance:
        """Hybrid assessment: embeddings for framework section finding + Gemini for scoring"""
        
        # Step 1: Use embeddings to find most relevant FRAMEWORK sections for improvement guidance
        improvement_sources, similarity_scores = self.find_most_relevant_framework_sections(
            customer_text, principle, framework_name)
        
        # Step 2: Use Gemini for compliance assessment on customer text (first 1000 chars for speed)
        customer_summary = customer_text[:1000] + "..." if len(customer_text) > 1000 else customer_text
        
        # Include framework guidance in the prompt to provide context
        framework_context = "\n".join([src.split(": ", 1)[1] if ": " in src else src 
                                     for src in improvement_sources[:1]])[:400]  # Limit context for speed
        
        prompt = f"""
        HYBRID COMPLIANCE ASSESSMENT for {framework_name} principle:
        
        PRINCIPLE: {principle.title}
        KEY REQUIREMENTS: {', '.join(principle.key_requirements[:2])}
        
        CUSTOMER POLICY DOCUMENT:
        {customer_summary}
        
        FRAMEWORK REFERENCE FOR IMPROVEMENT:
        {framework_context}
        
        Rate how well the customer policy addresses this principle compared to the framework requirements.
        
        Provide ONLY this JSON:
        {{
            "score": X.X,
            "evidence": ["specific evidence found in customer policy", "additional evidence"],
            "quick_rec": "One actionable improvement recommendation based on framework guidance"
        }}
        
        Score based on principle coverage in customer policy:
        8-10: Comprehensive implementation of principle
        5-7: Partial implementation with gaps
        2-4: Minimal coverage of principle
        0-1: Principle not addressed
        """
        
        try:
            ai_response = self.call_gemini_model_fast(prompt, max_tokens=300)
            
            # Handle None response from API failures
            if ai_response is None:
                logger.warning("Gemini API unavailable, using fallback assessment")
                fallback_score = max(similarity_scores) * 8 if similarity_scores else 2.0
                return FastPrincipleCompliance(
                    principle=principle,
                    compliance_score=fallback_score,
                    evidence_found=["Assessment based on framework similarity matching"],
                    improvement_sources=improvement_sources,
                    similarity_scores=similarity_scores,
                    quick_recommendation="Review framework sections provided and implement missing requirements"
                )
            
            # Improved JSON parsing with multiple attempts
            json_data = None
            
            # Try to find JSON in response
            start_idx = ai_response.find('{')
            end_idx = ai_response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = ai_response[start_idx:end_idx]
                try:
                    json_data = json.loads(json_str)
                except json.JSONDecodeError as je:
                    logger.warning(f"JSON parsing failed: {je}")
                    # Try to clean the JSON string
                    import re
                    cleaned_json = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
                    try:
                        json_data = json.loads(cleaned_json)
                    except:
                        logger.error("Failed to parse cleaned JSON")
            
            if json_data:
                return FastPrincipleCompliance(
                    principle=principle,
                    compliance_score=float(json_data.get("score", 0)),
                    evidence_found=json_data.get("evidence", ["Evidence parsing failed"]),
                    improvement_sources=improvement_sources,
                    similarity_scores=similarity_scores,
                    quick_recommendation=json_data.get("quick_rec", "Review framework requirements and implement missing elements")
                )
                
        except Exception as e:
            logger.error(f"Error in hybrid assessment: {e}")
        
        # Fallback assessment with similarity-based scoring
        fallback_score = max(similarity_scores) * 6 if similarity_scores else 0.0
        
        return FastPrincipleCompliance(
            principle=principle,
            compliance_score=fallback_score,
            evidence_found=["Assessment based on framework similarity analysis"],
            improvement_sources=improvement_sources,
            similarity_scores=similarity_scores,
            quick_recommendation="Manual review recommended - compare your policy against framework sections provided"
        )

    def hybrid_analyze_framework(self, customer_text: str, framework_name: str) -> FastFrameworkAnalysis:
        """Hybrid framework analysis using embeddings + Gemini"""
        
        logger.info(f"Hybrid analyzing {framework_name} framework...")
        
        # Load saved principles
        extracted_principles = self.load_extracted_principles(framework_name)
        
        if not extracted_principles:
            logger.warning(f"No principles found for {framework_name}")
            return FastFrameworkAnalysis(
                framework_name=framework_name,
                principle_compliance=[],
                overall_compliance_score=0.0,
                overall_status="NO_PRINCIPLES_LOADED"
            )
        
        # Hybrid assess compliance for each principle
        principle_compliance = []
        total_score = 0
        
        for principle in extracted_principles:
            compliance = self.hybrid_assess_principle_compliance(customer_text, principle, framework_name)
            principle_compliance.append(compliance)
            total_score += compliance.compliance_score
        
        # Calculate overall compliance
        overall_score = total_score / len(extracted_principles) if extracted_principles else 0
        
        # Determine overall status
        if overall_score >= 8:
            overall_status = "EXCELLENT"
        elif overall_score >= 6:
            overall_status = "GOOD"
        elif overall_score >= 4:
            overall_status = "NEEDS_IMPROVEMENT"
        else:
            overall_status = "CRITICAL"
        
        return FastFrameworkAnalysis(
            framework_name=framework_name,
            principle_compliance=principle_compliance,
            overall_compliance_score=overall_score,
            overall_status=overall_status
        )

    def hybrid_assess_customer_document(self, customer_document_path: str, 
                                      embeddings_file: str = None) -> FastAssessment:
        """Hybrid assessment function - embeddings + Gemini for 30 second execution"""
        
        logger.info("Starting hybrid AI governance assessment...")
        
        # Initialize embedding model
        self.initialize_embedding_model()
        
        # Load framework embeddings if provided
        if embeddings_file and os.path.exists(embeddings_file):
            self.load_framework_embeddings(embeddings_file)
        
        # Load customer document
        try:
            with open(customer_document_path, 'r', encoding='utf-8') as f:
                customer_text = f.read()
        except Exception as e:
            raise ValueError(f"Error reading customer document: {e}")
        
        # Clean text
        customer_text = re.sub(r'\s+', ' ', customer_text.strip())
        document_name = os.path.splitext(os.path.basename(customer_document_path))[0]
        
        # Hybrid analyze each framework
        oecd_analysis = self.hybrid_analyze_framework(customer_text, "OECD")
        nist_analysis = self.hybrid_analyze_framework(customer_text, "NIST") 
        eu_analysis = self.hybrid_analyze_framework(customer_text, "EU")
        
        # Calculate overall assessment
        overall_score = (oecd_analysis.overall_compliance_score + 
                        nist_analysis.overall_compliance_score + 
                        eu_analysis.overall_compliance_score) / 3
        
        if overall_score >= 8:
            overall_status = "EXCELLENT"
        elif overall_score >= 6:
            overall_status = "GOOD"
        elif overall_score >= 4:
            overall_status = "NEEDS_IMPROVEMENT"
        else:
            overall_status = "CRITICAL"
        
        assessment = FastAssessment(
            document_name=document_name,
            oecd_analysis=oecd_analysis,
            nist_analysis=nist_analysis,
            eu_analysis=eu_analysis,
            overall_score=overall_score,
            overall_status=overall_status
        )
        
        logger.info("Hybrid assessment completed successfully")
        return assessment

    def generate_hybrid_txt_report(self, assessment: FastAssessment, output_file: str = None) -> str:
        """Generate hybrid TXT report with embedding similarity scores and source references"""
        
        report = f"""AI GOVERNANCE HYBRID COMPLIANCE ASSESSMENT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Document: {assessment.document_name}
Method: Embeddings (framework guidance identification) + Gemini AI (compliance scoring)

=============================================================================
EXECUTIVE SUMMARY
=============================================================================
Overall Score: {assessment.overall_score:.1f}/10
Overall Status: {assessment.overall_status}

Framework Scores:
- OECD Values-based Principles: {assessment.oecd_analysis.overall_compliance_score:.1f}/10 ({assessment.oecd_analysis.overall_status})
- NIST AI Risk Management: {assessment.nist_analysis.overall_compliance_score:.1f}/10 ({assessment.nist_analysis.overall_status})
- EU AI Act Requirements: {assessment.eu_analysis.overall_compliance_score:.1f}/10 ({assessment.eu_analysis.overall_status})

=============================================================================
DETAILED HYBRID ANALYSIS WITH FRAMEWORK GUIDANCE
=============================================================================

"""
        
        # Helper function to add framework analysis
        def add_framework_analysis(analysis: FastFrameworkAnalysis, framework_title: str):
            nonlocal report
            
            report += f"""
--- {framework_title} ---
Overall Score: {analysis.overall_compliance_score:.1f}/10
Status: {analysis.overall_status}
Principles Assessed: {len(analysis.principle_compliance)}

"""
            
            for i, compliance in enumerate(analysis.principle_compliance, 1):
                status_symbol = "✓" if compliance.compliance_score >= 7 else "⚠" if compliance.compliance_score >= 4 else "✗"
                
                # Show similarity scores if available
                similarity_info = ""
                if compliance.similarity_scores:
                    max_similarity = max(compliance.similarity_scores)
                    similarity_info = f" (Semantic Match: {max_similarity:.2f})"
                
                report += f"""
{i}. {compliance.principle.title} [{status_symbol}]
   Score: {compliance.compliance_score:.1f}/10{similarity_info}
   
   Evidence Found (AI Analysis):
   {chr(10).join([f"   - {evidence}" for evidence in compliance.evidence_found[:2]])}
   
   Improvement Recommendation:
   {compliance.quick_recommendation}
   
   Framework Sections to Implement (Embedding-Based):
   {chr(10).join([f"   - {source}" for source in compliance.improvement_sources[:2]]) if compliance.improvement_sources else "   - No relevant framework sections found"}
   
   Key Requirements:
   {chr(10).join([f"   - {req}" for req in compliance.principle.key_requirements[:2]])}
   
   Framework Match Scores: {', '.join([f'{score:.2f}' for score in compliance.similarity_scores[:3]]) if compliance.similarity_scores else 'N/A'}

"""
        
        # Add each framework analysis
        add_framework_analysis(assessment.oecd_analysis, "OECD VALUES-BASED PRINCIPLES")
        add_framework_analysis(assessment.nist_analysis, "NIST AI RISK MANAGEMENT FRAMEWORK")
        add_framework_analysis(assessment.eu_analysis, "EU AI ACT REQUIREMENTS")
        
        # Summary recommendations with semantic insights
        report += f"""
=============================================================================
PRIORITY IMPROVEMENT ACTIONS (Based on Hybrid Analysis)
=============================================================================

Critical Issues (Score < 4):
"""
        
        critical_issues = []
        for analysis in [assessment.oecd_analysis, assessment.nist_analysis, assessment.eu_analysis]:
            for compliance in analysis.principle_compliance:
                if compliance.compliance_score < 4:
                    similarity_note = f" [Low semantic match: {max(compliance.similarity_scores):.2f}]" if compliance.similarity_scores else ""
                    critical_issues.append(f"- {analysis.framework_name}: {compliance.principle.title} (Score: {compliance.compliance_score:.1f}){similarity_note}")
        
        if critical_issues:
            report += "\n" + "\n".join(critical_issues[:5])
        else:
            report += "\nNo critical issues identified."
        
        report += f"""

High Framework Match, Low Score (Quick Implementation Opportunities):
"""
        
        quick_wins = []
        for analysis in [assessment.oecd_analysis, assessment.nist_analysis, assessment.eu_analysis]:
            for compliance in analysis.principle_compliance:
                if compliance.similarity_scores and max(compliance.similarity_scores) > 0.5 and compliance.compliance_score < 6:
                    quick_wins.append(f"- {analysis.framework_name}: {compliance.principle.title} (Score: {compliance.compliance_score:.1f}, Framework Match: {max(compliance.similarity_scores):.2f})")
        
        if quick_wins:
            report += "\n" + "\n".join(quick_wins[:3])
        else:
            report += "\nNo quick implementation opportunities identified."
        
        report += f"""

Strengths (Score 7+):
"""
        
        strengths = []
        for analysis in [assessment.oecd_analysis, assessment.nist_analysis, assessment.eu_analysis]:
            for compliance in analysis.principle_compliance:
                if compliance.compliance_score >= 7:
                    match_note = f" [Framework guidance available: {max(compliance.similarity_scores):.2f}]" if compliance.similarity_scores else ""
                    strengths.append(f"- {analysis.framework_name}: {compliance.principle.title} (Score: {compliance.compliance_score:.1f}){match_note}")
        
        if strengths:
            report += "\n" + "\n".join(strengths[:5])
        else:
            report += "\nNo high-scoring principles identified."
        
        report += f"""

=============================================================================
HYBRID METHODOLOGY SUMMARY
=============================================================================

Framework Guidance Identification:
- Used sentence embeddings to find most relevant framework document sections
- Calculated cosine similarity between principles and actual OECD/NIST/EU content
- Identified top-matching framework sections for implementation guidance

AI Compliance Scoring:
- Gemini AI analyzed customer policy against principle requirements
- Provided evidence-based scoring with specific recommendations
- Used framework context to generate targeted improvement suggestions

Benefits of Framework-Guided Approach:
- Shows specific framework sections to implement (not just what you have)
- Provides authoritative source material for improvements
- Gives similarity scores to prioritize implementation efforts
- Links compliance gaps directly to official framework guidance

Assessment completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"Hybrid report saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving report: {e}")
        
        return report


def main():
    """Hybrid Fast AI Assessment Tool with Integrated Radar Charts - Embeddings + Gemini (30 Second Execution)"""
    print("="*70)
    print("HYBRID FAST AI GOVERNANCE ASSESSMENT TOOL")
    print("="*70)
    print("🔄 Hybrid Approach: Embeddings + Gemini AI")
    print("🎯 Embeddings: Framework guidance identification")
    print("🤖 Gemini: Compliance scoring & recommendations")
    print("⚡ Optimized for 30-second execution")
    print("📋 Shows relevant framework sections for implementation")
    print("📊 Includes automatic radar chart generation\n")
    
    # Get API key
    api_key = input("Enter your Google AI API key: ").strip()
    if not api_key:
        print("Error: API key is required")
        return
    
    if not api_key.startswith("AIza"):
        print("Warning: API key format may be incorrect (should start with 'AIza')")
        continue_anyway = input("Continue anyway? (y/n): ").strip().lower()
        if continue_anyway != 'y':
            return
    
    # Get document path
    doc_path = input("Enter path to your AI policy document (.txt): ").strip()
    if not os.path.exists(doc_path):
        print(f"Error: Document not found at {doc_path}")
        return
    
    # Get embeddings file (optional for enhanced accuracy)
    embeddings_file = input("Enter path to reference embeddings JSON file (optional, press Enter to skip): ").strip()
    if embeddings_file and not os.path.exists(embeddings_file):
        print(f"Warning: Embeddings file not found at {embeddings_file}, proceeding without framework embeddings")
        embeddings_file = None
    
    # Ask about radar chart generation
    generate_charts = input("Generate radar charts and visualizations? (y/n): ").strip().lower()
    create_visualizations = generate_charts == 'y'
    
    # Check if saved principles exist
    saved_dir = "saved_principles"
    required_files = ["oecd_principles.json", "nist_principles.json", "eu_principles.json"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(saved_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"\nError: Missing principle files in {saved_dir}/:")
        for file in missing_files:
            print(f"  - {file}")
        print("Please run the principle extraction process first.")
        return
    
    try:
        print("\n🔄 Starting hybrid assessment (Embeddings + Gemini)...")
        start_time = datetime.now()
        
        # Use embedding model that matches your 768-dim embeddings
        assessor = HybridFastAIAssessment(api_key, embedding_model="all-mpnet-base-v2")
        assessment = assessor.hybrid_assess_customer_document(doc_path, embeddings_file)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Display results
        print(f"\n🎯 HYBRID ASSESSMENT RESULTS (Completed in {execution_time:.1f}s)")
        print("="*70)
        print(f"Document: {assessment.document_name}")
        print(f"Overall Score: {assessment.overall_score:.1f}/10 ({assessment.overall_status})")
        print(f"OECD: {assessment.oecd_analysis.overall_compliance_score:.1f}/10 ({assessment.oecd_analysis.overall_status})")
        print(f"NIST: {assessment.nist_analysis.overall_compliance_score:.1f}/10 ({assessment.nist_analysis.overall_status})")
        print(f"EU: {assessment.eu_analysis.overall_compliance_score:.1f}/10 ({assessment.eu_analysis.overall_status})")
        
        # Show API success rate
        total_principles = (len(assessment.oecd_analysis.principle_compliance) + 
                           len(assessment.nist_analysis.principle_compliance) + 
                           len(assessment.eu_analysis.principle_compliance))
        
        api_failures = sum(1 for analysis in [assessment.oecd_analysis, assessment.nist_analysis, assessment.eu_analysis]
                          for compliance in analysis.principle_compliance 
                          if "API unavailable" in compliance.quick_recommendation or 
                             "semantic similarity only" in str(compliance.evidence_found))
        
        if api_failures > 0:
            print(f"⚠️  {api_failures}/{total_principles} assessments used fallback due to API issues")
        
        # Generate hybrid report
        output_file = f"{assessment.document_name}_hybrid_assessment.txt"
        report = assessor.generate_hybrid_txt_report(assessment, output_file)
        
        # Generate radar charts if requested
        charts_generated = False
        if create_visualizations:
            try:
                print(f"\n📊 Generating radar charts and visualizations...")
                chart_generator = RadarChartGenerator()
                chart_dir = f"{assessment.document_name}_charts"
                chart_generator.generate_charts_from_assessment(assessment, output_file, chart_dir)
                
                print(f"📊 Charts saved to: {chart_dir}/")
                charts_generated = True
                
            except Exception as chart_error:
                print(f"⚠️  Error generating charts: {chart_error}")
                print("📄 TXT report still available without visualizations")
        
        print(f"\n✅ Hybrid Assessment Complete!")
        print(f"📄 Report saved to: {output_file}")
        print(f"⏱️  Execution time: {execution_time:.1f} seconds")
        print(f"🔄 Method: Embeddings (framework guidance) + Gemini (scoring)")
        
        if charts_generated:
            print(f"📊 Radar charts: {assessment.document_name}_charts/")
        
        # Show sample insights
        high_framework_match_low_score = 0
        for analysis in [assessment.oecd_analysis, assessment.nist_analysis, assessment.eu_analysis]:
            for compliance in analysis.principle_compliance:
                if compliance.similarity_scores and max(compliance.similarity_scores) > 0.5 and compliance.compliance_score < 6:
                    high_framework_match_low_score += 1
        
        critical_count = sum(1 for analysis in [assessment.oecd_analysis, assessment.nist_analysis, assessment.eu_analysis]
                           for compliance in analysis.principle_compliance 
                           if compliance.compliance_score < 4)
        
        if critical_count > 0:
            print(f"⚠️  {critical_count} critical issues identified")
        
        if high_framework_match_low_score > 0:
            print(f"🎯 {high_framework_match_low_score} quick implementation opportunities (clear framework guidance available)")
        
        if critical_count == 0 and high_framework_match_low_score == 0:
            print("✅ Strong overall compliance with good framework alignment")
        
        # Show file outputs summary
        print(f"\n📁 OUTPUT SUMMARY:")
        print(f"   📄 TXT Report: {output_file}")
        if charts_generated:
            print(f"   📊 Individual Charts: {assessment.document_name}_charts/{assessment.document_name}_[framework]_radar.png")
            print(f"   📊 Comparison Chart: {assessment.document_name}_charts/{assessment.document_name}_comparison.png")
        
    except Exception as e:
        print(f"\n❌ Error during hybrid assessment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()