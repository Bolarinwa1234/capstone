from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests
import json
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable is required")

app = FastAPI(title="AI Career Advisor API")

# frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class PromptRequest(BaseModel):
    prompt: str

class UserSkillsRequest(BaseModel):
    skills: str  # user skills as comma seperated strings

class DesiredJobRequest(BaseModel):
    desired_job: str 
    skills: UserSkillsRequest

class Question(BaseModel):
    question: str

def call_llm(prompt: str, retries: int = 3) -> str:
    """Helper function to call the LLM API using only DeepSeek model"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",  # Optional: helps with rate limiting
        "X-Title": "AI Career Advisor"  # Optional: for OpenRouter analytics
    }
    
    payload = {
        "model": "deepseek/deepseek-r1:free",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    for attempt in range(retries + 1):
        try:
            logger.info(f"Attempting API call to deepseek/deepseek-r1:free (attempt {attempt + 1})")
            logger.info(f"Payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            logger.info(f"API Response Status: {response.status_code}")
            logger.info(f"Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                results = response.json()
                logger.info(f"Full API Response: {json.dumps(results, indent=2)}")
                
                # Handle DeepSeek R1 specific response structure
                if "choices" in results and len(results["choices"]) > 0:
                    choice = results["choices"][0]
                    logger.info(f"Choice structure: {json.dumps(choice, indent=2)}")
                    
                    # Try different response structures
                    content = None
                    
                    # Standard OpenAI format
                    if "message" in choice and choice["message"] and "content" in choice["message"]:
                        content = choice["message"]["content"]
                    
                    # Delta format (for streaming responses)
                    elif "delta" in choice and choice["delta"] and "content" in choice["delta"]:
                        content = choice["delta"]["content"]
                    
                    # Text format (some models use this)
                    elif "text" in choice:
                        content = choice["text"]
                    
                    # Direct content (fallback)
                    elif isinstance(choice, str):
                        content = choice
                    
                    if content and content.strip():
                        logger.info(f"Successfully extracted content: {content[:100]}...")
                        return content.strip()
                    else:
                        logger.warning(f"No valid content found in choice: {choice}")
                
                else:
                    logger.error(f"No choices in response or empty choices: {results}")
                    
            else:
                error_detail = response.text
                logger.error(f"API Error {response.status_code}: {error_detail}")
                
                # Handle specific error codes
                if response.status_code == 401:
                    raise HTTPException(status_code=401, detail="Invalid API key")
                elif response.status_code == 402:
                    raise HTTPException(status_code=402, detail="Insufficient credits")
                elif response.status_code == 404:
                    raise HTTPException(status_code=404, detail="Model not found - deepseek/deepseek-r1:free may not be available")
                elif response.status_code == 429:
                    logger.warning(f"Rate limited, attempt {attempt + 1}/{retries + 1}")
                    if attempt < retries:
                        import time
                        wait_time = (attempt + 1) * 2  # Exponential backoff
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
                elif response.status_code >= 500:
                    logger.warning(f"Server error {response.status_code}, attempt {attempt + 1}/{retries + 1}")
                    if attempt < retries:
                        import time
                        time.sleep(2)  # Wait before retry
                        continue
                else:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"API request failed: {error_detail}"
                    )
                    
        except requests.Timeout:
            logger.error(f"Timeout on attempt {attempt + 1}/{retries + 1}")
            if attempt < retries:
                continue
            else:
                raise HTTPException(status_code=504, detail="Request timed out after multiple attempts")
                
        except requests.ConnectionError:
            logger.error(f"Connection error on attempt {attempt + 1}/{retries + 1}")
            if attempt < retries:
                import time
                time.sleep(1)
                continue
            else:
                raise HTTPException(status_code=503, detail="Unable to connect to AI service")
                
        except requests.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            if attempt < retries:
                continue
            else:
                raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            logger.error(f"Response text: {response.text}")
            if attempt < retries:
                continue
            else:
                raise HTTPException(status_code=500, detail="Invalid JSON response from AI service")
    
    # If we get here, all attempts failed
    raise HTTPException(status_code=500, detail="DeepSeek model failed to respond after multiple attempts")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/test-connection")
def test_api_connection():
    """Test endpoint to check if the API connection works"""
    try:
        test_prompt = "Say 'Hello World' and nothing else."
        logger.info("Testing API connection...")
        response = call_llm(test_prompt)
        logger.info(f"Test successful: {response}")
        return {
            "status": "success",
            "test_prompt": test_prompt,
            "api_response": response
        }
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }

@app.post("/career-advice")
def get_career_advice(request: DesiredJobRequest):
    """Get career advice based on skills list"""
    try:
        logger.info(f"Processing career advice request: {request.desired_job}")
        
        skills_list = [s.strip().title() for s in request.skills.skills.split(",") if s.strip()]
        skills_text = ", ".join(skills_list)
        job_text = request.desired_job
        
        logger.info(f"Skills: {skills_text}, Job: {job_text}")
        
        prompt = f"""You are an AI career advisor.
Based on the user's skills: {skills_text}, suggest maximum of 4 missing skills that the user must have in order to be ready for a job in {job_text}.
Return ONLY the key skills in a numbered list, no explanations.

Example format:
1. Skill One
2. Skill Two
3. Skill Three
4. Skill Four"""
        
        skill_response = call_llm(prompt)
        logger.info(f"LLM Response: {skill_response}")
        
        # Process the response to extract skills
        skill_lines = skill_response.split("\n")
        skills = []
        
        for line in skill_lines:
            line = line.strip()
            if line and (any(char.isdigit() for char in line) or line.startswith('-') or line.startswith('•')):
                # Remove numbering (1., 2., etc.) and clean up
                if '.' in line:
                    skill_title = line.split('.', 1)[-1].strip()
                elif line.startswith('-') or line.startswith('•'):
                    skill_title = line[1:].strip()
                else:
                    skill_title = line
                
                if skill_title and len(skill_title) > 2:
                    skills.append(skill_title)
        
        # If no skills found, try to extract from any non-empty lines
        if not skills:
            skills = [line.strip() for line in skill_lines if line.strip() and len(line.strip()) > 3][:4]
        
        logger.info(f"Extracted skills: {skills}")
        
        return {
            "skills": skills_list,
            "job": request.desired_job,
            "recommended_skills": skills[:4],  # Ensure max 4 skills
            # "raw_response": skill_response
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Unexpected error in career advice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing career advice: {str(e)}")

@app.post("/get-careers")
def get_careers_from_user_input(request: UserSkillsRequest):
    """User-friendly endpoint - accepts skills as comma-separated string"""
    try:
        logger.info(f"Processing career recommendations for skills: {request.skills}")
        
        # Clean and parse the skills string
        skills_list = [skill.strip().title() for skill in request.skills.split(",") if skill.strip()]
        
        if not skills_list:
            raise HTTPException(status_code=400, detail="Please provide at least one skill")
        
        skills_text = ", ".join(skills_list)
        
        prompt = f"""You are an AI career advisor.
Based on the user's skills: {skills_text}, suggest 3 relevant career roles.
Return ONLY the job titles in a numbered list, no explanations.

Example format:
1. Job Title One
2. Job Title Two  
3. Job Title Three"""
        
        careers_response = call_llm(prompt)
        logger.info(f"LLM Response: {careers_response}")
        
        # Processing the response to extract job titles
        career_lines = careers_response.split("\n")
        jobs = []
        
        for line in career_lines:
            line = line.strip()
            if line and (any(char.isdigit() for char in line) or line.startswith('-') or line.startswith('•')):
                # Handling different formats: "1. Job Title", "- Job Title", etc.
                if '.' in line:
                    job_title = line.split('.', 1)[-1].strip()
                elif line.startswith('-') or line.startswith('•'):
                    job_title = line[1:].strip()
                else:
                    job_title = line
                
                if job_title and job_title not in jobs and len(job_title) > 2:  # Avoiding duplicates
                    jobs.append(job_title)
        
        # If no jobs found, try to extract from any non-empty lines
        if not jobs:
            jobs = [line.strip() for line in career_lines if line.strip() and len(line.strip()) > 5][:3]
        
        logger.info(f"Extracted careers: {jobs}")
        
        return {
            "user_input": request.skills,
            "parsed_skills": skills_list,
            "recommended_careers": jobs[:3],  # Ensure max 3 careers
            "message": f"Based on your skills in {skills_text}, here are 3 career recommendations:"
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Unexpected error in get careers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing career advice: {str(e)}")
    
@app.post("/ask-question")
def ask_question(request: Question):
    """Answer career-related questions from users"""
    try:
        logger.info(f"Processing question: {request.question}")
        
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        prompt = f"""You are an AI career advisor with expertise in career development, job searching, skill building, and professional growth.

User's question: {request.question}

Provide helpful, actionable career advice. Keep your response concise but informative (1 paragraph maximum). Focus on practical steps the user can take."""
        
        answer = call_llm(prompt)
        logger.info(f"Generated answer for question: {request.question}")
        
        return {
            "question": request.question,
            "answer": answer,
            "status": "success"
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing your question: {str(e)}")

@app.post("/quick-career-check")
def quick_career_check():
    """Quick test with predefined skills"""
    try:
        logger.info("Running quick career check...")
        test_skills = "python, javascript, tableau"
        desired_job = 'machine learning'
        skills_request = UserSkillsRequest(skills=test_skills)
        request = DesiredJobRequest(skills=skills_request, desired_job=desired_job)
        return get_career_advice(request)
    except Exception as e:
        logger.error(f"Quick career check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quick test failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
