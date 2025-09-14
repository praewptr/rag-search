import os
from typing import Dict, List, Optional
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from services.vector_store import VectorStore

class QAEngine:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
        # Initialize OpenAI LLM (you can replace with other models)
        self.llm = OpenAI(
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create custom prompt template
        self.prompt_template = PromptTemplate(
            template="""
            Use the following pieces of context to answer the question at the end. 
            If you don't know the answer based on the context, just say that you don't know, 
            don't try to make up an answer.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer: """,
            input_variables=["context", "question"]
        )
    
    async def get_answer(self, question: str, document_ids: Optional[List[str]] = None) -> Dict:
        """
        Get answer for a question based on the document context
        """
        try:
            # Search for relevant documents
            search_results = await self.vector_store.search_similar(
                query=question,
                n_results=5,
                document_ids=document_ids
            )
            
            if not search_results['documents'] or not search_results['documents'][0]:
                return {
                    "answer": "I don't have enough information to answer this question based on the uploaded documents.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Combine relevant text chunks as context
            context_chunks = search_results['documents'][0]
            context = "\n\n".join(context_chunks)
            
            # Generate answer using the LLM
            prompt = self.prompt_template.format(context=context, question=question)
            answer = await self._generate_answer_with_fallback(prompt, context_chunks, question)
            
            # Extract sources
            sources = []
            for metadata in search_results['metadatas'][0]:
                if metadata['filename'] not in sources:
                    sources.append(metadata['filename'])
            
            # Calculate confidence based on similarity scores
            distances = search_results.get('distances', [[]])[0]
            if distances:
                # Convert distances to similarities (assuming cosine distance)
                similarities = [1 - dist for dist in distances]
                confidence = max(similarities) if similarities else 0.0
            else:
                confidence = 0.5
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": min(confidence, 1.0)
            }
        
        except Exception as e:
            # Fallback to simple context-based answer
            try:
                search_results = await self.vector_store.search_similar(
                    query=question,
                    n_results=3,
                    document_ids=document_ids
                )
                
                if search_results['documents'] and search_results['documents'][0]:
                    context_chunks = search_results['documents'][0]
                    answer = await self._generate_simple_answer(context_chunks, question)
                    
                    sources = []
                    for metadata in search_results['metadatas'][0]:
                        if metadata['filename'] not in sources:
                            sources.append(metadata['filename'])
                    
                    return {
                        "answer": answer,
                        "sources": sources,
                        "confidence": 0.3
                    }
            except Exception:
                pass
            
            raise Exception(f"Error generating answer: {str(e)}")
    
    async def _generate_answer_with_fallback(self, prompt: str, context_chunks: List[str], question: str) -> str:
        """
        Generate answer with OpenAI API, fallback to simple method if API fails
        """
        try:
            # Try to use OpenAI API
            if os.getenv("OPENAI_API_KEY"):
                response = self.llm(prompt)
                return response.strip()
            else:
                # Fallback to simple context-based answer
                return await self._generate_simple_answer(context_chunks, question)
        except Exception:
            # Fallback to simple context-based answer
            return await self._generate_simple_answer(context_chunks, question)
    
    async def _generate_simple_answer(self, context_chunks: List[str], question: str) -> str:
        """
        Generate a simple answer by finding the most relevant context chunk
        """
        try:
            # Simple keyword matching approach
            question_lower = question.lower()
            question_words = set(question_lower.split())
            
            best_chunk = ""
            best_score = 0
            
            for chunk in context_chunks:
                chunk_lower = chunk.lower()
                chunk_words = set(chunk_lower.split())
                
                # Calculate simple similarity score
                common_words = question_words.intersection(chunk_words)
                score = len(common_words) / len(question_words) if question_words else 0
                
                if score > best_score:
                    best_score = score
                    best_chunk = chunk
            
            if best_chunk and best_score > 0.1:
                # Return the most relevant chunk with a prefix
                return f"Based on the document: {best_chunk[:500]}{'...' if len(best_chunk) > 500 else ''}"
            else:
                return "I couldn't find a specific answer to your question in the uploaded documents. Please try rephrasing your question or upload more relevant documents."
        
        except Exception as e:
            return f"I encountered an error while processing your question: {str(e)}"
