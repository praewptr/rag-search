import os
import base64
from typing import List, Dict
from io import BytesIO
import uuid
from PIL import Image
from pdf2image import convert_from_bytes
from openai import AzureOpenAI

class ImageProcessor:
    def __init__(self):
        """Initialize the Azure OpenAI client for vision processing"""
        self.azure_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.vision_model = os.getenv("AZURE_OPENAI_VISION_MODEL", "gpt-4-vision-preview")
    
    async def process_pdf_with_vision(self, content: bytes, filename: str) -> Dict:
        """
        Convert PDF to images and process with Azure OpenAI Vision
        """
        try:
            # Convert PDF to images
            images = await self._pdf_to_images(content)
            
            # Process each image with Azure OpenAI Vision
            extracted_data = []
            for i, image in enumerate(images):
                # Convert image to base64
                image_b64 = await self._image_to_base64(image)
                
                # Extract text and understand content using Azure OpenAI Vision
                page_data = await self._analyze_image_with_azure_vision(
                    image_b64, 
                    page_number=i+1,
                    filename=filename
                )
                
                extracted_data.append(page_data)
            
            # Generate document ID
            document_id = str(uuid.uuid4())
            
            return {
                "document_id": document_id,
                "filename": filename,
                "total_pages": len(images),
                "extracted_data": extracted_data,
                "processing_method": "azure_openai_vision"
            }
        
        except Exception as e:
            raise Exception(f"Error processing PDF with vision: {str(e)}")
    
    async def _pdf_to_images(self, pdf_content: bytes) -> List[Image.Image]:
        """
        Convert PDF bytes to a list of PIL Images
        """
        try:
            # Convert PDF to images
            images = convert_from_bytes(
                pdf_content,
                dpi=200,  # Good quality for text extraction
                fmt='PNG'
            )
            return images
        except Exception as e:
            raise Exception(f"Error converting PDF to images: {str(e)}")
    
    async def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string
        """
        try:
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            return image_b64
        except Exception as e:
            raise Exception(f"Error converting image to base64: {str(e)}")
    
    async def _analyze_image_with_azure_vision(self, image_b64: str, page_number: int, filename: str) -> Dict:
        """
        Analyze image using Azure OpenAI Vision API
        """
        try:
            # Prepare the vision prompt
            vision_prompt = """
            You are an expert document analyzer. Please analyze this image and provide:
            
            1. **Extracted Text**: All text content in the image, maintaining structure and formatting
            2. **Document Understanding**: 
               - What type of document is this?
               - What are the main topics or subjects?
               - Key information, data points, or insights
               - Any tables, charts, or visual elements and their meaning
            3. **Content Summary**: A brief summary of the page content
            4. **Key Entities**: Important names, dates, numbers, locations, or concepts
            
            Please format your response as JSON with the following structure:
            {
                "extracted_text": "...",
                "document_type": "...",
                "main_topics": [...],
                "key_insights": [...],
                "content_summary": "...",
                "key_entities": {...},
                "visual_elements": [...]
            }
            """
            
            # Make the API call to Azure OpenAI Vision
            response = self.azure_client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vision_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0.1
            )
            
            # Parse the response
            vision_result = response.choices[0].message.content
            
            # Try to parse as JSON, fallback to text if not valid JSON
            try:
                import json
                parsed_result = json.loads(vision_result)
            except json.JSONDecodeError:
                # If not valid JSON, create structured response from text
                parsed_result = {
                    "extracted_text": vision_result,
                    "document_type": "unknown",
                    "main_topics": [],
                    "key_insights": [],
                    "content_summary": vision_result[:200] + "..." if len(vision_result) > 200 else vision_result,
                    "key_entities": {},
                    "visual_elements": []
                }
            
            return {
                "page_number": page_number,
                "filename": filename,
                "vision_analysis": parsed_result,
                "raw_response": vision_result
            }
        
        except Exception as e:
            # Fallback response if Azure OpenAI fails
            return {
                "page_number": page_number,
                "filename": filename,
                "vision_analysis": {
                    "extracted_text": f"Error processing page {page_number}: {str(e)}",
                    "document_type": "error",
                    "main_topics": [],
                    "key_insights": [],
                    "content_summary": f"Failed to analyze page {page_number}",
                    "key_entities": {},
                    "visual_elements": []
                },
                "raw_response": f"Error: {str(e)}"
            }
    
    async def extract_text_chunks_from_vision_data(self, extracted_data: List[Dict]) -> List[str]:
        """
        Extract and chunk text from vision analysis data
        """
        try:
            text_chunks = []
            
            for page_data in extracted_data:
                vision_analysis = page_data.get("vision_analysis", {})
                page_number = page_data.get("page_number", 1)
                
                # Get extracted text
                extracted_text = vision_analysis.get("extracted_text", "")
                
                # Get content summary and insights
                content_summary = vision_analysis.get("content_summary", "")
                key_insights = vision_analysis.get("key_insights", [])
                main_topics = vision_analysis.get("main_topics", [])
                
                # Combine information into comprehensive chunks
                page_content = f"Page {page_number}:\n\n"
                
                if extracted_text:
                    page_content += f"Text Content:\n{extracted_text}\n\n"
                
                if content_summary:
                    page_content += f"Summary: {content_summary}\n\n"
                
                if main_topics:
                    page_content += f"Main Topics: {', '.join(main_topics)}\n\n"
                
                if key_insights:
                    insights_text = '\n'.join([f"- {insight}" for insight in key_insights])
                    page_content += f"Key Insights:\n{insights_text}\n\n"
                
                # Split into smaller chunks if the page content is too long
                if len(page_content) > 1500:
                    # Split into smaller chunks
                    chunk_size = 1000
                    for i in range(0, len(page_content), chunk_size):
                        chunk = page_content[i:i + chunk_size]
                        if chunk.strip():
                            text_chunks.append(chunk.strip())
                else:
                    if page_content.strip():
                        text_chunks.append(page_content.strip())
            
            return text_chunks
        
        except Exception as e:
            raise Exception(f"Error extracting text chunks from vision data: {str(e)}")
