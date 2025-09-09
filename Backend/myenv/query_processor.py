

import google.generativeai as genai
from typing import List, Dict, Any
import re

class QueryProcessor:
    """Process queries and generate responses with proper citations and conversation context"""
    
    def __init__(self, gemini_api_key: str):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def analyze_query(self, query: str, conversation_context: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze the query to determine optimal retrieval strategy with conversation context"""
        query_lower = query.lower()
        
        # Check conversation context for drug mentions
        current_drug = None
        if conversation_context:
            for exchange in reversed(conversation_context):
                if exchange.get('detected_drug'):
                    current_drug = exchange['detected_drug']
                    break
        
        # Drug-specific queries
        drug_keywords = {
    'orencia': 'orencia', 
    'simponi': 'simponi', 
    'aria': 'aria',
    'humira': 'humira', 
    'enbrel': 'enbrel', 
    'remicade': 'remicade', 
    'keytruda': 'keytruda',
    'alora_pi': 'alora_pi',
    'augtyro': 'augtyro',
    'blujepa': 'blujepa',
    'cinbinqo': 'cinbinqo',
    'dalvance_pi': 'dalvance_pi',
    'herceptin': 'herceptin',
    'ibuprofen': 'ibuprofen',
    'jakafi': 'jakafi',
    'methadone': 'methadone',
    'olumiant': 'olumiant',
    'opzelura-prescribing-infor': 'opzelura-prescribing-infor',
    'prilosec': 'prilosec',
    'rinvoq_pi': 'rinvoq_pi',
    'sotyktu': 'sotyktu',
    'stelara': 'stelara',
    'xeljanx': 'xeljanx'
}

        
        mentioned_drugs = []
        pdf_filter = current_drug  # Start with context drug
        
        for drug, filter_term in drug_keywords.items():
            if drug in query_lower:
                mentioned_drugs.append(drug)
                pdf_filter = filter_term  # Use the most specific match from current query
        
        # Medical content queries
        medical_keywords = {
            'dose', 'dosage', 'side effect', 'adverse', 'warning', 
            'contraindication', 'interaction', 'administration', 'treatment',
            'safety', 'efficacy', 'pharmacokinetics', 'indication'
        }
        is_medical_query = any(keyword in query_lower for keyword in medical_keywords)
        
        # Visual content queries
        visual_keywords = {'graph', 'chart', 'diagram', 'figure', 'image', 'picture', 'visual'}
        is_visual_query = any(keyword in query_lower for keyword in visual_keywords)
        
        # Follow-up queries
        is_follow_up = self._is_follow_up_query(query, conversation_context)
        
        return {
            "mentioned_drugs": mentioned_drugs,
            "is_medical_query": is_medical_query,
            "is_visual_query": is_visual_query,
            "is_fda_specific": is_medical_query,
            "pdf_filter": pdf_filter,
            "prefer_medical": is_medical_query,
            "prefer_visual": is_visual_query,
            "is_follow_up": is_follow_up,
            "current_drug": current_drug
        }
    
    def _is_follow_up_query(self, query: str, conversation_context: List[Dict[str, Any]]) -> bool:
        """Check if this is a follow-up query based on conversation context"""
        if not conversation_context:
            return False
        
        # Common follow-up patterns
        follow_up_patterns = [
            r'what about',
            r'how about',
            r'what are (?:its|their)',
            r'what is (?:its|their)',
            r'what does (?:it|they)',
            r'can you tell me more',
            r'and (?:its|their)',
            r'also',
            r'too'
        ]
        
        query_lower = query.lower()
        
        # Check for follow-up patterns
        for pattern in follow_up_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Check if query seems to reference previous context but doesn't mention a specific drug
        last_exchange = conversation_context[-1] if conversation_context else None
        if last_exchange and last_exchange.get('detected_drug'):
            # If previous exchange was about a drug and current query doesn't mention any specific drug
            drug_mentioned = any(drug in query_lower for drug in[
  'orencia',
  'simponi',
  'aria',
  'humira',
  'enbrel',
  'remicade',
  'keytruda',
  'alora_pi',
  'Augtyro',
  'BLUJEPA',
  'Cinbinqo',
  'dalvance_pi',
  'Herceptin',
  'Ibuprofen',
  'jakafi',
  'methadone',
  'Olumiant',
  'opzelura-prescribing-infor',
  'PRILOSEC',
  'rinvoq_pi',
  'Sotyktu',
  'STELARA',
  'XELJANZ'
]
)
            if not drug_mentioned:
                return True
        
        return False
    
    def format_retrieved_info(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved information with proper citations"""
        if not retrieved_chunks:
            return "No relevant information found in the documents."
        
        formatted_info = "RETRIEVED INFORMATION WITH SOURCE CITATIONS:\n\n"
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            # Build citation based on document type
            citation_parts = [f"Source: {chunk.get('pdf_name', 'Unknown PDF')}"]
            
            # Add section information for text documents
            if chunk.get('section') and chunk['section'] not in ["TABULAR_DATA", "VISUAL_DATA", "TABULAR_DATA_ROW"]:
                citation_parts.append(f"Section: {chunk['section']}")
            
            # Add page information
            page_info = self._get_page_info(chunk)
            if page_info:
                citation_parts.append(f"Page: {page_info}")
            
            # Add specific context for tables and images
            if chunk.get('doc_type') == 'table':
                if chunk.get('table_index'):
                    citation_parts.append(f"Table: {chunk['table_index']}")
            elif chunk.get('doc_type') == 'image':
                if chunk.get('image_index'):
                    citation_parts.append(f"Figure: {chunk['image_index']}")
            
            citation = ", ".join(citation_parts)
            
            formatted_info += f"{i}. {citation}\nContent: {chunk['chunk_text']}\n\n"
        
        return formatted_info
    
    def _get_page_info(self, chunk: Dict[str, Any]) -> str:
        """Get appropriate page information for citation"""
        # Prefer PDF page number if available
        if chunk.get('pdf_page_number'):
            return str(chunk['pdf_page_number'])
        elif chunk.get('pdf_page_start') and chunk.get('pdf_page_end'):
            if chunk['pdf_page_start'] == chunk['pdf_page_end']:
                return str(chunk['pdf_page_start'])
            else:
                return f"{chunk['pdf_page_start']}-{chunk['pdf_page_end']}"
        elif chunk.get('page_start') and chunk.get('page_end'):
            if chunk['page_start'] == chunk['page_end']:
                return str(chunk['page_start'])
            else:
                return f"{chunk['page_start']}-{chunk['page_end']}"
        return "Unknown"
        
    def generate_response(self, user_query: str, retrieved_info: str, conversation_context: List[Dict[str, Any]] = None) -> str:
        """Generate natural, conversational response using Gemini with proper source consolidation and context"""
        context_prompt = ""
        if conversation_context:
            # Add relevant context from previous exchanges
            context_prompt = "\n*CONVERSATION CONTEXT:*\n"
            for exchange in conversation_context[-5:]:  # Last 3 exchanges
                context_prompt += f"User: {exchange['user_query']}\n"
                context_prompt += f"Assistant: {exchange['assistant_response'][:200]}...\n\n"
        
        prompt = f"""
You are a helpful and precise medical information assistant. Your knowledge comes strictly from the provided excerpts from official drug labels.

*USER'S QUESTION:* {user_query}
{context_prompt}
*RELEVANT INFORMATION FROM THE DRUG LABEL:*
{retrieved_info}

*YOUR TASK:*
Answer the user's question clearly and conversationally using ONLY the information provided above.

*HOW TO STRUCTURE YOUR RESPONSE:*
1.  *Direct Answer:* Start with a clear, direct answer to the user's question.
2.  *Key Details:* Provide the necessary details in a simple, easy-to-read paragraph.
3.  *Context (If Important):* Briefly mention any critical context if it's important for understanding the answer.
4.  *Precise Citation:* On a new line, state: "This information comes from:" followed by a CONSOLIDATED and precise list of sources.
    -   *CONSOLIDATE:* Identify the PRIMARY source where the core answer is definitively stated. Cite ONLY that primary, definitive source.
    -   *FORMAT:* Use the format: "DocumentName, Section Name (Page X)" or "DocumentName, (Page X)".
    -   *CRITICAL:* You MUST include the page number from the provided context for the primary source.
    -   DO NOT mention or reference table numbers (e.g., "Table 1", "Table 2") in the response

*CRITICAL RULES:*
-   *DO NOT* use markdown, headings like "Summary:", or any special formatting in the body.
-   *DO NOT* list every page number from the context. Cite only the single most relevant and authoritative source.
-   *DO NOT* make up any information not present in the provided context.
-   Write in a friendly, professional, and conversational tone.
-   If this seems like a follow-up question, maintain continuity with the previous conversation.

*RESPONSE:*
"""
        try:
            response = self.model.generate_content(prompt)
            # Clean up the response to ensure natural formatting
            response_text = response.text
            # Remove any residual markdown formatting that might appear
            response_text = response_text.replace("**", "").replace("*", "")
            return response_text
            
        except Exception as e:
            return f"I apologize, but I encountered an error while generating the response. Please try again later. Error: {str(e)}"