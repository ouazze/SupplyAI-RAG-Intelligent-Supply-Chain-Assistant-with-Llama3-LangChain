"""
SupplyAI Prompt Templates
Professional, structured prompts for supply chain reasoning.
"""

from langchain.prompts import PromptTemplate


class SupplyChainPrompts:
    """
    Collection of specialized prompts for the SupplyAI assistant.
    """
    
    @staticmethod
    def get_rag_prompt() -> PromptTemplate:
        """
        Main RAG prompt template for supply chain Q&A.
        Structures the LLM as a supply chain expert with clear reasoning rules.
        """
        template = """You are SupplyAI, an expert supply chain management assistant with 20+ years of experience in logistics, inventory optimization, and procurement.

Your task is to analyze the provided supply chain context and answer the user's question with precision, actionable insights, and professional recommendations.

## CONTEXT (Retrieved Supply Chain Records):
{context}

## USER QUESTION:
{question}

## INSTRUCTIONS:
1. **Analyze** the context carefully. Pay attention to stock levels, reorder points, lead times, and supplier information.
2. **Detect Risks**: Identify low stock, potential stockouts, supplier delays, or cost anomalies.
3. **Provide Recommendations**: Suggest specific actions (restock quantities, supplier negotiations, safety stock adjustments).
4. **Explain Clearly**: Use professional but accessible language. Quantify recommendations where possible.
5. **Be Honest**: If the context lacks sufficient information, state what data is missing rather than guessing.

## RESPONSE FORMAT:
- **Direct Answer**: Concise answer to the question
- **Risk Analysis**: Bullet points of identified risks (if any)
- **Recommendations**: Numbered, actionable next steps
- **Confidence**: High/Medium/Low based on context quality

SupplyAI Response:
"""
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    @staticmethod
    def get_restock_recommendation_prompt() -> PromptTemplate:
        """
        Specialized prompt for restock recommendation engine.
        """
        template = """Based on the following inventory data, provide a detailed restock recommendation.

Product Context:
{context}

Generate a restock plan including:
1. Recommended order quantity
2. Priority level (Critical/High/Medium/Low)
3. Suggested order date
4. Supplier considerations
5. Budget estimate

Format as a professional procurement brief.
"""
        return PromptTemplate(
            template=template,
            input_variables=["context"]
        )
    
    @staticmethod
    def get_risk_assessment_prompt() -> PromptTemplate:
        """
        Prompt for comprehensive risk assessment.
        """
        template = """Analyze the following supply chain data for operational risks:

{context}

Identify and categorize risks:
- **Stockout Risk**: Probability and impact
- **Supplier Risk**: Single-source dependencies, lead time volatility
- **Cost Risk**: Price fluctuation exposure
- **Operational Risk**: Warehouse capacity, turnover issues

Provide a risk matrix and mitigation strategies.
"""
        return PromptTemplate(
            template=template,
            input_variables=["context"]
        )