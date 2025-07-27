# GovernAIce ML Service Layer
# --------------------------
# This module provides a service layer for integrating ML models with the backend API.
# It handles model initialization, API calls, and result processing.

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add ML module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml'))

try:
    from models.compliance_checker import AIComplianceChecker
    from models.policy_comparator import AIPolicyComparator
    from models.principle_assessor import PrincipleBasedAIAssessment
    ML_MODELS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML models not available: {e}")
    ML_MODELS_AVAILABLE = False

logger = logging.getLogger(__name__)

class MLService:
    """
    Service layer for ML model integration with the backend API.
    Handles model initialization, API calls, and result processing.
    """
    
    def __init__(self):
        """Initialize ML service with all available models."""
        self.compliance_checker = None
        self.policy_comparator = None
        self.principle_assessor = None
        
        if ML_MODELS_AVAILABLE:
            self._initialize_models()
        else:
            logger.warning("ML models not available - using fallback methods")
    
    def _initialize_models(self):
        """Initialize ML models with proper configuration."""
        try:
            api_key = os.getenv('TOGETHER_API_KEY')
            if not api_key:
                logger.warning("TOGETHER_API_KEY not set - ML models may not work properly")
            
            # Initialize compliance checker
            self.compliance_checker = AIComplianceChecker(
                together_api_key=api_key,
                embedding_model="all-MiniLM-L6-v2"
            )
            logger.info("Compliance checker initialized")
            
            # Initialize policy comparator
            self.policy_comparator = AIPolicyComparator(
                together_api_key=api_key
            )
            logger.info("Policy comparator initialized")
            
            # Initialize principle assessor
            self.principle_assessor = PrincipleBasedAIAssessment(
                together_api_key=api_key
            )
            logger.info("Principle assessor initialized")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
            ML_MODELS_AVAILABLE = False
    
    def analyze_compliance(self, document_text: str, country: str, input_type: str = "text") -> Dict[str, Any]:
        """
        Analyze document for compliance using the compliance checker model.
        
        Args:
            document_text: Text content to analyze
            country: Target country for compliance analysis
            input_type: Type of input ("text" or "file")
            
        Returns:
            Dictionary containing compliance analysis results
        """
        if not self.compliance_checker:
            return self._fallback_compliance_analysis(document_text, country)
        
        try:
            logger.info(f"Running compliance analysis for {country}")
            result = self.compliance_checker.run_compliance_check(
                document_input=document_text,
                country=country,
                input_type=input_type
            )
            return self._format_compliance_result(result)
        except Exception as e:
            logger.error(f"Error in compliance analysis: {e}")
            return self._fallback_compliance_analysis(document_text, country)
    
    def compare_policies(self, user_document: str, reference_file: str, country: str = None) -> Dict[str, Any]:
        """
        Compare user document against reference policies using the policy comparator model.
        
        Args:
            user_document: User's policy document text
            reference_file: Path to reference embeddings file
            country: Target country for comparison
            
        Returns:
            Dictionary containing policy comparison results
        """
        if not self.policy_comparator:
            return self._fallback_policy_comparison(user_document, country)
        
        try:
            logger.info(f"Running policy comparison for {country or 'all countries'}")
            result = self.policy_comparator.compare_policy_document(
                user_document_text=user_document,
                reference_embeddings_file=reference_file,
                target_country=country
            )
            return self._format_policy_comparison_result(result)
        except Exception as e:
            logger.error(f"Error in policy comparison: {e}")
            return self._fallback_policy_comparison(user_document, country)
    
    def assess_principles(self, document_path: str, embeddings_file: str) -> Dict[str, Any]:
        """
        Assess document against AI principles using the principle assessor model.
        
        Args:
            document_path: Path to document file
            embeddings_file: Path to embeddings file
            
        Returns:
            Dictionary containing principle assessment results
        """
        if not self.principle_assessor:
            return self._fallback_principle_assessment(document_path)
        
        try:
            logger.info("Running principle assessment")
            result = self.principle_assessor.assess_customer_document(
                customer_document_path=document_path,
                embeddings_file=embeddings_file
            )
            return self._format_principle_assessment_result(result)
        except Exception as e:
            logger.error(f"Error in principle assessment: {e}")
            return self._fallback_principle_assessment(document_path)
    
    def _format_compliance_result(self, result) -> Dict[str, Any]:
        """Format compliance analysis result for API response."""
        return {
            'overall_score': getattr(result, 'overall_score', 0),
            'major_gaps': getattr(result, 'major_gaps', []),
            'excellencies': getattr(result, 'excellencies', []),
            'improvement_strategy': getattr(result, 'improvement_strategy', []),
            'detailed_analysis': getattr(result, 'detailed_analysis', {}),
            'referenced_policies': getattr(result, 'referenced_policies', []),
            'analysis_type': 'compliance'
        }
    
    def _format_policy_comparison_result(self, result) -> Dict[str, Any]:
        """Format policy comparison result for API response."""
        return {
            'overall_score': getattr(result, 'overall_score', 0),
            'risk_level': getattr(result, 'risk_level', 'Unknown'),
            'detailed_scores': getattr(result, 'detailed_scores', {}),
            'alignment_analysis': getattr(result, 'alignment_analysis', {}),
            'recommendations': getattr(result, 'recommendations', []),
            'gaps_identified': getattr(result, 'gaps_identified', []),
            'analysis_type': 'policy_comparison'
        }
    
    def _format_principle_assessment_result(self, result) -> Dict[str, Any]:
        """Format principle assessment result for API response."""
        return {
            'oecd_analysis': getattr(result, 'oecd_analysis', {}),
            'nist_analysis': getattr(result, 'nist_analysis', {}),
            'eu_analysis': getattr(result, 'eu_analysis', {}),
            'cross_framework_insights': getattr(result, 'cross_framework_insights', {}),
            'overall_risk_assessment': getattr(result, 'overall_risk_assessment', 'Unknown'),
            'strategic_recommendations': getattr(result, 'strategic_recommendations', []),
            'implementation_roadmap': getattr(result, 'implementation_roadmap', {}),
            'analysis_type': 'principle_assessment'
        }
    
    def _fallback_compliance_analysis(self, document_text: str, country: str) -> Dict[str, Any]:
        """Fallback compliance analysis when ML models are not available."""
        return {
            'overall_score': 50,  # Neutral score
            'major_gaps': ['ML model not available for detailed analysis'],
            'excellencies': ['Basic text analysis completed'],
            'improvement_strategy': ['Enable ML models for advanced analysis'],
            'detailed_analysis': {'status': 'fallback_mode'},
            'referenced_policies': [],
            'analysis_type': 'compliance_fallback'
        }
    
    def _fallback_policy_comparison(self, user_document: str, country: str) -> Dict[str, Any]:
        """Fallback policy comparison when ML models are not available."""
        return {
            'overall_score': 50,
            'risk_level': 'Medium',
            'detailed_scores': {'basic_analysis': 50},
            'alignment_analysis': {'status': 'fallback_mode'},
            'recommendations': ['Enable ML models for detailed comparison'],
            'gaps_identified': ['ML model not available'],
            'analysis_type': 'policy_comparison_fallback'
        }
    
    def _fallback_principle_assessment(self, document_path: str) -> Dict[str, Any]:
        """Fallback principle assessment when ML models are not available."""
        return {
            'oecd_analysis': {'status': 'fallback_mode'},
            'nist_analysis': {'status': 'fallback_mode'},
            'eu_analysis': {'status': 'fallback_mode'},
            'cross_framework_insights': {'status': 'fallback_mode'},
            'overall_risk_assessment': 'Unknown',
            'strategic_recommendations': ['Enable ML models for principle assessment'],
            'implementation_roadmap': {'status': 'fallback_mode'},
            'analysis_type': 'principle_assessment_fallback'
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of ML models."""
        return {
            'ml_models_available': ML_MODELS_AVAILABLE,
            'compliance_checker': self.compliance_checker is not None,
            'policy_comparator': self.policy_comparator is not None,
            'principle_assessor': self.principle_assessor is not None,
            'together_api_key': bool(os.getenv('TOGETHER_API_KEY'))
        }

# Global ML service instance
ml_service = MLService() 
