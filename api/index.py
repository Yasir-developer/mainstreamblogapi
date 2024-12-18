import os
import uvicorn
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the Gemini API client
genai.configure(api_key=GEMINI_API_KEY)

# Define the request body schema
class GenerateRequest(BaseModel):
    title: str
    keywords: List[str]
    description: Optional[str] = None
    tone: Optional[str] = "neutral"
    word_count: Optional[int] = 1000
    language: Optional[str] = "en"

# Define the response schema
class GenerateResponse(BaseModel):
    content: str
    meta_title: str
    meta_description: str
    tags: List[str]

app = FastAPI()

@app.get("/")
def main():
    return "Home Page"

@app.post("/generate", response_model=GenerateResponse)
def generate_blog_post(req: GenerateRequest):
    """
    Generates a blog post using Gemini API based on the provided input.
    """
    # Construct the prompt
    # The prompt can be structured to guide the AI model more effectively.
    # For example, we can include instructions and the input parameters clearly.
    prompt = (
        f"Generate a well-structured, SEO-friendly blog post.\n"
        f"Title: {req.title}\n"
        f"Keywords: {', '.join(req.keywords)}\n"
        f"Language: {req.language}\n"
        f"Tone: {req.tone}\n"
        f"Desired word count: approximately {req.word_count} words.\n"
    )
    if req.description:
        prompt += f"Additional context: {req.description}\n"
    prompt += (
        "The blog post should include a clear introduction, organized sections with headings, "
        "HTML formatting (like <h2>, <p>, <ul>, <li>) and a conclusion."
        "After the blog post content, provide meta information in the following format:\n\n"
        "**Meta Information:**\n\n"
        "**Meta Title:** Your suggested meta title\n" # meta title 55-60 char
        "**Meta Description:** Your suggested meta description\n" # description 160
        "**Tags:** comma-separated tags\n"
    )

    # Initialize the model
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    chat_session = model.start_chat(history=[])

    # Send the prompt to the model
    response = chat_session.send_message(prompt)
    generated_text = response.text.strip()
    

    # Here, we assume that the model returns content and then at the end
    # provides meta title, meta description, and tags.
    # You need to parse these from the model’s response.
    # The parsing logic below is a simplistic example and may need adjustments
    # depending on how the model is instructed or how it formats the output.

    # Let's assume the model ends the content section and then outputs meta info in a clear manner.
    # For example, the model might produce something like:
    # "<h2>Introduction</h2>...<p>Content</p> ... 
    # META_TITLE: Some Title
    # META_DESCRIPTION: Some description
    # TAGS: tag1, tag2, tag3"
    # We'll search for these markers.

    if "**Meta Information:**" in generated_text:
        parts = generated_text.split("**Meta Information:**")
        content_part = parts[0].strip()
        meta_part = parts[1].strip()
    else:
        # If there's no meta information, raise an error or provide defaults
        raise HTTPException(status_code=500, detail="No meta information found.")

    # Initialize defaults
    meta_title = "Default Meta Title"
    meta_description = "Default Meta Description"
    tags = []

    # Parse the meta_part line by line
    # Expected lines (as per model output):
    # **Meta Title:** The Future of AI: Emerging Trends and Ethical Implications
    # **Meta Description:** Explore the future...
    # **Tags:** artificial intelligence, AI, ...
    meta_lines = meta_part.split("\n")
    for line in meta_lines:
        line_stripped = line.strip()
        if line_stripped.upper().startswith("**META TITLE:**"):
            meta_title = line_stripped.split("**Meta Title:**", 1)[1].strip()
        elif line_stripped.upper().startswith("**META DESCRIPTION:**"):
            meta_description = line_stripped.split("**Meta Description:**", 1)[1].strip()
        elif line_stripped.upper().startswith("**TAGS:**"):
            raw_tags = line_stripped.split("**Tags:**", 1)[1].strip()
            tags = [t.strip() for t in raw_tags.split(",") if t.strip()]

    # Validate that we have content
    if not content_part:
        raise HTTPException(status_code=500, detail="No content generated.")

    return GenerateResponse(
        content=content_part.replace('\n',''),
        meta_title=meta_title,
        meta_description=meta_description,
        tags=tags
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)