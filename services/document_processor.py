import uuid
from typing import List
import PyPDF2
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    async def process_document(self, content: bytes, filename: str) -> str:
        """
        Process a document and return a unique document ID
        """
        document_id = str(uuid.uuid4())
        return document_id
    
    async def extract_text_chunks(self, content: bytes) -> List[str]:
        """
        Extract text from PDF and split into chunks
        """
        try:
            # Create a BytesIO object from the content
            pdf_file = BytesIO(content)
            
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            return chunks
        
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove special characters but keep punctuation
        # You can add more cleaning rules here
        
        return text
