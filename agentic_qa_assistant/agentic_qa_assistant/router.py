"""Router for intelligent tool selection between SQL, RAG, and Hybrid modes."""

import re
import time
import logging
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ToolChoice(Enum):
    """Available tool choices."""
    SQL = "SQL"
    RAG = "RAG"
    HYBRID = "HYBRID"


@dataclass
class RoutingDecision:
    """Routing decision with metadata."""
    decision: ToolChoice
    confidence: float  # 0.0 to 1.0
    reasons: List[str]
    matched_keywords: Dict[str, List[str]]
    latency_ms: int


class LLMRouterResponse(BaseModel):
    """Pydantic model for LLM router response."""
    decision: str
    confidence: float
    reasons: List[str]


class RuleBasedRouter:
    """Rule-based router using keyword matching."""
    
    def __init__(self):
        """Initialize rule-based router with keyword patterns."""
        
        # SQL keywords (data/numeric/analytical queries)
        self.sql_keywords = {
            'time': ['monthly', 'yearly', 'quarterly', 'annual', 'period', 'time', 'temporal'],
            'location': ['country', 'region', 'germany', 'france', 'italy', 'spain', 'uk', 'poland', 'romania', 
                        'europe', 'european', 'western', 'southern', 'eastern'],
            'models': ['model', 'rav4', 'yaris', 'c-hr', 'ux', 'nx', 'rx', 'lexus', 'toyota'],
            'metrics': ['sales', 'contracts', 'volume', 'numbers', 'total', 'sum', 'count', 'average',
                       'trend', 'compare', 'comparison', 'vs', 'versus', 'against'],
            'powertrain': ['hev', 'phev', 'petrol', 'hybrid', 'electric', 'powertrain', 'engine'],
            'ordertype': ['private', 'fleet', 'demo', 'b2c', 'b2b', 'dealer']
        }
        
        # RAG keywords (policy/manual/warranty queries)  
        self.rag_keywords = {
            'warranty': ['warranty', 'guarantee', 'coverage', 'policy', 'terms', 'conditions'],
            'contract': ['contract', 'agreement', 'clause', 'appendix', 'legal', 'retailer'],
            'manual': ['manual', 'owner', 'maintenance', 'service', 'repair', 'feature', 'function',
                      'tire', 'dashboard', 'indicator', 'location', 'where', 'how to'],
            'policy': ['policy', 'rule', 'regulation', 'standard', 'procedure', 'guideline'],
            'duration': ['years', 'months', 'km', 'kilometers', 'mileage', 'unlimited', 'perforation', 'corrosion']
        }
        
        # Hybrid keywords (mixed queries needing both data and policy)
        self.hybrid_keywords = {
            'comparison': ['compare', 'vs', 'versus', 'difference', 'different', 'both', 'each'],
            'summary': ['summarize', 'summary', 'overview', 'key', 'main', 'important'],
            'analysis': ['analysis', 'analyze', 'explain', 'why', 'reason', 'because']
        }
        
    def route(self, question: str) -> RoutingDecision:
        """Route question using rule-based keyword matching.
        
        Args:
            question: User question to route
            
        Returns:
            Routing decision with confidence and reasoning
        """
        start_time = time.time()
        question_lower = question.lower()
        
        # Track matched keywords
        matched_sql = self._match_keywords(question_lower, self.sql_keywords)
        matched_rag = self._match_keywords(question_lower, self.rag_keywords) 
        matched_hybrid = self._match_keywords(question_lower, self.hybrid_keywords)
        
        # Calculate scores
        sql_score = len(matched_sql)
        rag_score = len(matched_rag)
        hybrid_score = len(matched_hybrid)
        
        # Decision logic
        reasons = []
        decision = None
        confidence = 0.0
        
        # Strong indicators for hybrid
        if hybrid_score > 0 and (sql_score > 0 or rag_score > 0):
            decision = ToolChoice.HYBRID
            confidence = min(0.9, 0.6 + (hybrid_score + min(sql_score, rag_score)) * 0.1)
            reasons.append(f"Hybrid indicators: {list(matched_hybrid.keys())}")
            if sql_score > 0:
                reasons.append(f"Also has SQL indicators: {list(matched_sql.keys())}")
            if rag_score > 0:
                reasons.append(f"Also has RAG indicators: {list(matched_rag.keys())}")
                
        # Clear SQL preference
        elif sql_score >= 2 and rag_score <= 1:
            decision = ToolChoice.SQL
            confidence = min(0.95, 0.7 + sql_score * 0.1)
            reasons.append(f"Strong SQL indicators: {list(matched_sql.keys())}")
            
        # Clear RAG preference  
        elif rag_score >= 2 and sql_score <= 1:
            decision = ToolChoice.RAG
            confidence = min(0.95, 0.7 + rag_score * 0.1)
            reasons.append(f"Strong RAG indicators: {list(matched_rag.keys())}")
            
        # Mixed signals - prefer hybrid if both have some score
        elif sql_score > 0 and rag_score > 0:
            decision = ToolChoice.HYBRID
            confidence = 0.8
            reasons.append(f"Mixed SQL ({list(matched_sql.keys())}) and RAG ({list(matched_rag.keys())}) indicators")
            
        # Weak signals - pick the stronger one
        elif sql_score > rag_score:
            decision = ToolChoice.SQL
            confidence = 0.6
            reasons.append(f"Weak SQL preference: {list(matched_sql.keys())}")
            
        elif rag_score > sql_score:
            decision = ToolChoice.RAG
            confidence = 0.6
            reasons.append(f"Weak RAG preference: {list(matched_rag.keys())}")
            
        # No clear indicators - default to hybrid as safest
        else:
            decision = ToolChoice.HYBRID
            confidence = 0.4
            reasons.append("No clear indicators - defaulting to hybrid for safety")
            
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Combine all matched keywords
        all_matched = {**matched_sql, **matched_rag, **matched_hybrid}
        
        return RoutingDecision(
            decision=decision,
            confidence=confidence,
            reasons=reasons,
            matched_keywords=all_matched,
            latency_ms=latency_ms
        )
        
    def _match_keywords(self, text: str, keyword_groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Match keywords in text and return grouped matches."""
        matches = {}
        
        for group, keywords in keyword_groups.items():
            found_keywords = []
            for keyword in keywords:
                if keyword in text:
                    found_keywords.append(keyword)
                    
            if found_keywords:
                matches[group] = found_keywords
                
        return matches


class LLMRouter:
    """LLM-based router for ambiguous cases."""
    
    def __init__(self, openai_client: OpenAI):
        """Initialize LLM router.
        
        Args:
            openai_client: OpenAI client instance
        """
        self.client = openai_client
        
    def route(self, question: str) -> RoutingDecision:
        """Route question using LLM classification.
        
        Args:
            question: User question to route
            
        Returns:
            Routing decision with confidence and reasoning
        """
        start_time = time.time()
        
        system_prompt = """You are a routing assistant that categorizes user questions into three types:

1. SQL: Questions about numerical data, sales figures, trends, comparisons between models/regions/time periods
   - Examples: "Monthly sales of RAV4 in Germany", "Compare Toyota vs Lexus sales in 2024"

2. RAG: Questions about policies, warranties, contracts, manuals, procedures, or factual information
   - Examples: "What is Toyota warranty coverage?", "Where is the tire repair kit located?"

3. HYBRID: Questions that need both data analysis AND policy/manual information
   - Examples: "Compare sales and warranty differences between Toyota and Lexus"

Respond with a JSON object containing:
- decision: "SQL", "RAG", or "HYBRID" 
- confidence: float between 0.0 and 1.0
- reasons: list of strings explaining the classification

Be conservative - if unsure between two options, choose HYBRID."""

        user_prompt = f"Classify this question: {question}"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            import json
            result = json.loads(response.choices[0].message.content)
            
            # Validate and convert to our format
            llm_response = LLMRouterResponse(**result)
            
            # Convert to ToolChoice enum
            decision = ToolChoice(llm_response.decision.upper())
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return RoutingDecision(
                decision=decision,
                confidence=llm_response.confidence,
                reasons=llm_response.reasons,
                matched_keywords={"llm_classified": [llm_response.decision.lower()]},
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            
            # Fallback to hybrid with low confidence
            latency_ms = int((time.time() - start_time) * 1000)
            return RoutingDecision(
                decision=ToolChoice.HYBRID,
                confidence=0.3,
                reasons=[f"LLM routing failed: {e}"],
                matched_keywords={},
                latency_ms=latency_ms
            )


class SmartRouter:
    """Smart router that combines rule-based and LLM routing."""
    
    def __init__(self, openai_client: OpenAI, confidence_threshold: float = 0.6):
        """Initialize smart router.
        
        Args:
            openai_client: OpenAI client instance  
            confidence_threshold: Minimum confidence for rule-based routing
        """
        self.rule_router = RuleBasedRouter()
        self.llm_router = LLMRouter(openai_client)
        self.confidence_threshold = confidence_threshold
        
    def route(self, question: str) -> RoutingDecision:
        """Route question using hybrid rule-based + LLM approach.
        
        Args:
            question: User question to route
            
        Returns:
            Final routing decision
        """
        logger.info(f"Routing question: {question[:50]}...")
        
        # Try rule-based first
        rule_decision = self.rule_router.route(question)
        
        # If confident enough, use rule-based decision
        if rule_decision.confidence >= self.confidence_threshold:
            logger.info(f"Rule-based routing: {rule_decision.decision.value} (confidence: {rule_decision.confidence:.2f})")
            return rule_decision
            
        # Otherwise, use LLM fallback
        logger.info(f"Rule-based confidence too low ({rule_decision.confidence:.2f}), using LLM fallback")
        llm_decision = self.llm_router.route(question)
        
        # Combine insights from both approaches
        combined_reasons = rule_decision.reasons + ["LLM fallback used"] + llm_decision.reasons
        combined_keywords = {**rule_decision.matched_keywords, **llm_decision.matched_keywords}
        
        final_decision = RoutingDecision(
            decision=llm_decision.decision,
            confidence=llm_decision.confidence,
            reasons=combined_reasons,
            matched_keywords=combined_keywords,
            latency_ms=rule_decision.latency_ms + llm_decision.latency_ms
        )
        
        logger.info(f"Final routing: {final_decision.decision.value} (confidence: {final_decision.confidence:.2f})")
        return final_decision
        
    def batch_route(self, questions: List[str]) -> List[RoutingDecision]:
        """Route multiple questions.
        
        Args:
            questions: List of questions to route
            
        Returns:
            List of routing decisions
        """
        return [self.route(q) for q in questions]