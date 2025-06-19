import os
import re
import json
import uuid
import faiss
import pickle
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from sentence_transformers import SentenceTransformer
import ollama
from jira import JIRA
from dotenv import load_dotenv
import logging
import mimetypes
import requests
from typing import Dict, List, Optional
import spacy
from pydantic import BaseModel
import time
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add SessionMiddleware for conversation ID
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", str(uuid.uuid4())),
    session_cookie="session_id",
    max_age=3600  # 1 hour session expiry
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory for file uploads
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# File paths for FAISS index and metadata
INDEX_FILE = "ticket_index.faiss"
META_FILE = "ticket_metadata.pkl"
CONVERSATION_STATE_FILE = "conversation_state.pkl"

# Jira configuration
JIRA_DOMAIN = os.getenv("JIRA_DOMAIN")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")
JIRA_ISSUE_TYPE = os.getenv("JIRA_ISSUE_TYPE", "Task")
JIRA_URL = f"https://{JIRA_DOMAIN}" if JIRA_DOMAIN else None

# Debug environment variables
logger.info(f"JIRA_URL: {JIRA_URL}")
logger.info(f"JIRA_EMAIL: {JIRA_EMAIL}")
logger.info(f"JIRA_API_TOKEN: {JIRA_API_TOKEN[:4] + '...' if JIRA_API_TOKEN else None}")
logger.info(f"JIRA_PROJECT_KEY: {JIRA_PROJECT_KEY}")
logger.info(f"JIRA_ISSUE_TYPE: {JIRA_ISSUE_TYPE}")

# Initialize Jira client
def init_jira_client(retries=3, delay=5):
    if not all([JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY]):
        logger.error("Missing required Jira configuration in .env file")
        raise RuntimeError("Jira configuration incomplete")
    
    for attempt in range(retries):
        try:
            client = JIRA(
                server=JIRA_URL,
                basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN),
                options={"verify": True}
            )
            server_info = client.server_info()
            logger.info(f"Connected to Jira. Server version: {server_info['version']}")
            return client
        except Exception as e:
            logger.warning(f"Jira connection attempt {attempt + 1}/{retries} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            raise RuntimeError(f"Jira connection failed: {str(e)}")

try:
    jira_client = init_jira_client()
    logger.info("Jira client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Jira client: {str(e)}")
    raise RuntimeError(f"Jira initialization failed: {str(e)}")

# Models for validation and responses
class Ticket(BaseModel):
    id: str
    title: str
    issue: str
    priority: str
    component: str
    description: Optional[str] = None
    resolution: Optional[str] = None
    attachments: Optional[List[str]] = None
    issuetype: Optional[str] = None

class TicketCreateResponse(BaseModel):
    jira_key: str
    jira_url: str
    status: str

# Global state storage for multiple conversations
conversation_states = {}

# Load conversation state from disk
def load_conversation_state() -> Dict:
    try:
        if os.path.exists(CONVERSATION_STATE_FILE):
            with open(CONVERSATION_STATE_FILE, "rb") as f:
                state = pickle.load(f)
                logger.info("Conversation state loaded from disk")
                # Clean up stale conversations (older than 24 hours)
                cutoff = datetime.now() - timedelta(hours=24)
                for cid in list(state.keys()):
                    if state[cid].get("last_updated", datetime.min) < cutoff:
                        del state[cid]
                return state
        return {}
    except Exception as e:
        logger.error(f"Failed to load conversation state: {str(e)}")
        return {}

# Save conversation state to disk
def save_conversation_state():
    try:
        for state in conversation_states.values():
            state["last_updated"] = datetime.now()
        with open(CONVERSATION_STATE_FILE, "wb") as f:
            pickle.dump(conversation_states, f)
        logger.info("Conversation state saved to disk")
    except Exception as e:
        logger.error(f"Failed to save conversation state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save conversation state: {str(e)}")

# Initialize conversation state for a new conversation
def init_conversation_state(conversation_id: str):
    conversation_states[conversation_id] = {
        "context_history": [],
        "greeted": False,
        "ticket_started": False,
        "awaiting_attachment": False,
        "awaiting_upload_confirmation": False,
        "similar_checked": False,
        "awaiting_ticket_confirmation": False,
        "awaiting_validation_fix": False,
        "awaiting_no_similarity_confirmation": False,  # New state for no similarity confirmation
        "pending_fields": {
            "title": None,
            "issue": None,
            "priority": None,
            "component": None,
            "issuetype": None,
            "attachments": []
        },
        "finalized_ticket": None,
        "jira_response": None,
        "last_updated": datetime.now()
    }
    save_conversation_state()

# Load embedding model and FAISS index
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
try:
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)
except FileNotFoundError:
    logger.warning("FAISS index or metadata not found. Initializing empty index.")
    index = faiss.IndexFlatL2(384)  # Assuming 384-dim embeddings from all-MiniLM-L6-v2
    metadata = []

# Initialize spaCy for ticket analysis
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("spaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'.")
    raise RuntimeError("spaCy model not found")

# Get or create conversation ID
def get_conversation_id(request: Request) -> str:
    if "conversation_id" not in request.session:
        conversation_id = str(uuid.uuid4())
        request.session["conversation_id"] = conversation_id
        init_conversation_state(conversation_id)
    return request.session["conversation_id"]

# Mark greeted status
def mark_greeted_if_needed(text, state):
    if not state["greeted"]:
        if re.search(r"\b(hi+|hello+|hey+|good (morning|afternoon|evening))\b", text.lower()):
            state["greeted"] = True

# Search similar tickets
def search_similar_tickets(query, top_k=3):
    if index is None or not metadata:
        return []
    embedding = embedding_model.encode([query])
    scores, indices = index.search(np.array(embedding).astype("float32"), top_k)
    results = []
    for i, score in zip(indices[0], scores[0]):
        if i < len(metadata):
            results.append({"score": float(score), "ticket": metadata[i]})
    return results

# Infer component and issue type
def infer_component_and_issuetype(issue_text):
    issue_lower = issue_text.lower()
    bug_keywords = ["crash", "error", "fail", "failed", "failure", "bug", "defect", "issue", "broken", "not working", "timeout", "denied", "exception", "glitch"]
    task_keywords = ["add", "implement", "create", "new", "feature", "enhance", "improvement", "update", "change", "modify", "request", "configure", "setup"]
    components = ["ui", "authentication", "payment gateway", "login", "dashboard", "notification", "search", "analytics", "reporting", "deployment", "backend"]
    issuetype = "Bug" if any(k in issue_lower for k in bug_keywords) else JIRA_ISSUE_TYPE
    for c in components:
        if c in issue_lower:
            return c.title(), issuetype
    return "General", issuetype

# Build prompt for Mistral
def build_prompt(state) -> str:
    context = json.dumps(state["context_history"], indent=2)
    greeted = state.get("greeted", False)
    prompt = (
        "You are a professional, concise support assistant.\n"
        f"{'Start with a polite greeting if this is the first message.' if not greeted else ''}\n"
        "Only ask one thing at a time. Start by asking what problem the user is facing.\n"
        "Do NOT mention priority, component, issue type, or attachments until the problem is clearly described.\n"
        "If the issue involves visuals (like UI bugs or webpage errors), THEN and ONLY THEN ask: 'Would you like to attach any screenshots or links?'\n"
        "Keep replies short (under 2 sentences). Speak naturally.\n"
        "Avoid bullet points, technical jargon, or long instructions.\n"
        "Never use 'role' or 'content' in your output — speak as a human.\n"
        "If the user expresses that a suggested solution didn’t work (in any wording), treat that as a rejection.\n"
        "After the first rejection, stop showing similar tickets. Ask the user to describe their issue again and treat it as a new request.\n"
        "Only return the JSON ticket when the user has provided enough real-world detail to raise a support case. This includes: the service provider name (e.g., RedBus), date of booking, date of cancellation, refund method (e.g., UPI), refund amount, and confirmation that the refund is overdue.\n"
        "DO NOT return the JSON if these details are missing. Ask the user for them first, one at a time, if needed.\n"
        "Once the user has provided complete details, return ONLY a clean JSON object with fields: title, issue, priority, component, issuetype.\n"
        "After the JSON, ask: 'Would you like to raise this ticket?'\n"
        "If the user says yes, then ask for an attachment. This attachment is REQUIRED — it can be a PDF, image, or link. If user refuses, explain it's mandatory.\n"
        "Once the attachment is provided, respond: 'Ticket created. Thank you.' and do not ask anything else.\n"
        f"\nConversation so far:\n{context}\n"
    )
    return prompt


# Reset ticket state
def reset_ticket(state):
    state["context_history"] = []
    state["greeted"] = False
    state["ticket_started"] = False
    state["awaiting_attachment"] = False
    state["awaiting_upload_confirmation"] = False
    state["awaiting_ticket_confirmation"] = False
    state["awaiting_validation_fix"] = False
    state["similar_checked"] = False
    state["awaiting_no_similarity_confirmation"] = False
    state["finalized_ticket"] = None
    state["jira_response"] = None
    state["pending_fields"] = {
        "title": None,
        "issue": None,
        "priority": None,
        "component": None,
        "issuetype": None,
        "attachments": []
    }
    save_conversation_state()


# Get valid JIRA issue types
def get_valid_issue_types(project_key: str) -> List[str]:
    try:
        issue_types = jira_client.issue_types()
        project = jira_client.project(project_key)
        available_issue_types = [t.name for t in issue_types if t.id in [it.id for it in project.issueTypes]]
        logger.info(f"Valid issue types for project {project_key}: {available_issue_types}")
        return available_issue_types
    except Exception as e:
        logger.error(f"Failed to fetch issue types: {str(e)}")
        return ["Task", "Bug", "Story"]  # Fallback to common issue types

# Create JIRA ticket
def create_jira_ticket(ticket: Dict, retries=3, delay=5) -> Dict:
    try:
        priorities = {p.name.lower(): p.id for p in jira_client.priorities()}
        components = {c.name.lower(): c.id for c in jira_client.project_components(JIRA_PROJECT_KEY)}
        valid_issue_types = get_valid_issue_types(JIRA_PROJECT_KEY)
        
        logger.info(f"Available priorities: {priorities}")
        logger.info(f"Available components: {components}")
        logger.info(f"Available issue types: {valid_issue_types}")

        description = ticket.get("issue", "")
        attachments = ticket.get("attachments", [])

        priority_name = ticket.get("priority", "medium").lower()
        priority_id = priorities.get(priority_name)
        if not priority_id:
            logger.warning(f"Priority '{priority_name}' not found, defaulting to 'Medium'")
            priority_id = priorities.get("medium", list(priorities.values())[0])

        component_name = ticket.get("component", "").lower()
        component_id = components.get(component_name)
        components_field = [{"id": component_id}] if component_id else []

        # Validate issue type
        issue_type = ticket.get("issuetype", JIRA_ISSUE_TYPE).capitalize()
        if issue_type not in valid_issue_types:
            logger.warning(f"Invalid issue type '{issue_type}'. Defaulting to '{JIRA_ISSUE_TYPE}'")
            issue_type = JIRA_ISSUE_TYPE
            if issue_type not in valid_issue_types:
                issue_type = valid_issue_types[0] if valid_issue_types else "Task"
                logger.warning(f"Fallback issue type '{JIRA_ISSUE_TYPE}' also invalid. Using '{issue_type}'")

        issue_dict = {
            "project": {"key": JIRA_PROJECT_KEY},
            "summary": ticket.get("title", "New Ticket"),
            "description": description,
            "issuetype": {"name": issue_type},
            "priority": {"id": priority_id}
        }
        if components_field:
            issue_dict["components"] = components_field

        logger.info(f"Sending Jira payload: {json.dumps(issue_dict, indent=2)}")

        # Create the Jira ticket
        new_issue = None
        for attempt in range(retries):
            try:
                new_issue = jira_client.create_issue(fields=issue_dict)
                logger.info(f"Created Jira ticket: {new_issue.key}")
                break
            except Exception as e:
                logger.warning(f"Jira ticket creation attempt {attempt + 1}/{retries} failed: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    raise HTTPException(status_code=500, detail=f"Failed to create Jira ticket: {str(e)}")

        # Handle attachments
        failed_attachments = []
        for attachment in attachments:
            # Resolve absolute path relative to project directory
            filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), attachment.lstrip('/')))
            if not os.path.exists(filepath):
                logger.error(f"Attachment file not found: {filepath}")
                failed_attachments.append(attachment)
                continue
            if not os.access(filepath, os.R_OK):
                logger.error(f"Attachment file not readable: {filepath}")
                failed_attachments.append(attachment)
                continue
            for attempt in range(retries):
                try:
                    with open(filepath, "rb") as f:
                        jira_client.add_attachment(issue=new_issue, attachment=f, filename=os.path.basename(filepath))
                        logger.info(f"Successfully attached {os.path.basename(filepath)} to Jira ticket {new_issue.key}")
                        break
                except Exception as e:
                    logger.error(f"Attachment upload attempt {attempt + 1}/{retries} failed for {filepath}: {str(e)}")
                    if attempt == retries - 1:
                        logger.error(f"Failed to attach {filepath} to Jira ticket {new_issue.key}")
                        failed_attachments.append(attachment)

        response = {
            "jira_key": new_issue.key,
            "jira_url": f"{JIRA_URL}/browse/{new_issue.key}",
            "status": "created"
        }
        if failed_attachments:
            response["attachment_warnings"] = f"Failed to attach the following files: {', '.join(failed_attachments)}"
            logger.warning(response["attachment_warnings"])

        return response

    except Exception as e:
        logger.error(f"Failed to create Jira ticket: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create Jira ticket: {str(e)}")

# Ticket Analysis Agent
class TicketAnalysisAgent:
    def __init__(self):
        self.required_fields = ["title", "issue", "priority", "component"]
        self.priority_options = ["low", "medium", "high", "very high"]
        self.valid_components = ["ui", "authentication", "payment gateway", "login", "dashboard", "notification", "search"]
        self.nlp = nlp

    def analyze_ticket(self, ticket: Dict) -> Dict:
        missing_fields = []
        quality_issues = []

        for field in self.required_fields:
            if field not in ticket or not ticket[field]:
                missing_fields.append(field)

        if ticket.get("priority") and ticket["priority"].lower() not in self.priority_options:
            quality_issues.append(f"Invalid priority: {ticket['priority']}. Must be one of {self.priority_options}")
        if ticket.get("title") and len(ticket["title"]) < 5:
            quality_issues.append("Title is too short. Provide a descriptive title (min 5 characters).")
        if ticket.get("issue") and len(ticket["issue"]) < 20:
            quality_issues.append("Issue description is too brief. Provide more details (min 20 characters).")
        if ticket.get("component") and not isinstance(ticket["component"], str):
            quality_issues.append("Component must be a valid text string.")

        field_values = {field: ticket.get(field, "").strip().lower() for field in self.required_fields}
        for i, field1 in enumerate(self.required_fields):
            for field2 in self.required_fields[i + 1:]:
                if field_values[field1] and field_values[field2]:
                    if field1 == "priority" and field_values[field1] in self.priority_options:
                        continue
                    if field2 == "priority" and field_values[field2] in self.priority_options:
                        continue
                    if field_values[field1] == field_values[field2]:
                        flagged_field = self._choose_field_to_flag(field1, field2)
                        other_field = field2 if flagged_field == field1 else field1
                        quality_issues.append(
                            f"{flagged_field.capitalize()} is identical to {other_field}. Please provide a more distinct {flagged_field}."
                        )
                    else:
                        doc1 = self.nlp(field_values[field1])
                        doc2 = self.nlp(field_values[field2])
                        similarity = doc1.similarity(doc2)
                        if similarity > 0.9:
                            flagged_field = self._choose_field_to_flag(field1, field2)
                            other_field = field2 if flagged_field == field1 else field1
                            quality_issues.append(
                                f"{flagged_field.capitalize()} is too similar to {other_field}. Please provide a more distinct {flagged_field}."
                            )

        if ticket.get("issue"):
            issue_doc = self.nlp(ticket["issue"].strip())
            meaningful_tokens = [token.text for token in issue_doc if not token.is_stop and not token.is_punct]
            if len(meaningful_tokens) < 5:
                quality_issues.append(
                    "Issue description lacks sufficient detail. Please include specific information."
                )

        status = "success" if not missing_fields and not quality_issues else "incomplete"
        return {
            "status": status,
            "missing_fields": missing_fields,
            "quality_issues": quality_issues
        }

    def _choose_field_to_flag(self, field1: str, field2: str) -> str:
        priority_order = ["issue", "title", "component", "priority"]
        return field1 if priority_order.index(field1) < priority_order.index(field2) else field2

    def generate_guidance(self, analysis: Dict) -> Dict:
        guidance = []
        for field in analysis.get("missing_fields", []):
            guidance.append(f"Please provide the '{field}' field. It's required for ticket submission.")
        for issue in analysis.get("quality_issues", []):
            guidance.append(issue)
        return {"guidance": guidance}

    def summarize_resolutions(self, similar_tickets: List[Dict]) -> str:
        resolutions = [ticket.get("resolution", "").strip() for ticket in similar_tickets if ticket.get("resolution", "").strip()]
        if not resolutions:
            return "No resolution information available for similar tickets."
        combined = " ".join(resolutions)
        doc = self.nlp(combined)
        summary = " ".join([sent.text for sent in doc.sents][:2])
        return summary or "Unable to generate summary from resolutions."

# Main chat logic
def chat_with_mistral(user_input: str, conversation_id: str):
    state = conversation_states.get(conversation_id)
    if not state:
        init_conversation_state(conversation_id)
        state = conversation_states[conversation_id]

    ticket_analyzer = TicketAnalysisAgent()
    user_input_clean = user_input.strip().lower()

    # Save user input
    state["context_history"].append({"role": "user", "content": user_input.strip()})
    mark_greeted_if_needed(user_input, state)

    # --- SIMILARITY CHECK FIRST TIME ONLY ---
    if not state["similar_checked"] and not state["finalized_ticket"]:
        results = search_similar_tickets(user_input)
        if results and results[0]["score"] > 0.75:
            top = results[0]["ticket"]
            state["similar_checked"] = True
            summarized = ticket_analyzer.summarize_resolutions([top])
            state["context_history"].append({
                "role": "assistant",
                "content": (
                    f"I found a similar issue:\n- Title: {top.get('title')}\n"
                    f"- Issue: {top.get('issue')}\n- Resolution: {top.get('resolution', 'No resolution available')}"
                )
            })
            save_conversation_state()
            return {
                "response": (
                    f"I found a similar issue:\n- Title: {top.get('title')}\n"
                    f"- Issue: {top.get('issue')}\n- Resolution: {top.get('resolution', 'No resolution available')}\n\n"
                    "If this doesn't solve your issue, please describe your problem again."
                ),
                "status": "suggestion"
            }

    # --- MISTRAL LLM HANDLING ---
    prompt = build_prompt(state)
    try:
        response = ollama.chat(
            model="mistral",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Respond naturally."},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.4}
        )
        response_text = response["message"]["content"].strip()
        state["context_history"].append({"role": "assistant", "content": response_text})

        # -- Check for JSON ticket --
        match = re.search(r"{.*}", response_text, re.DOTALL)
        if match:
            try:
                ticket_data = json.loads(match.group())
                ticket_data["id"] = f"TICKET-{hash(conversation_id) % 10000:04}"
                ticket_data["description"] = ticket_data.get("issue", "")
                state["finalized_ticket"] = ticket_data

                analysis = ticket_analyzer.analyze_ticket(ticket_data)
                if analysis["status"] != "success":
                    state["awaiting_validation_fix"] = True
                    guidance = ticket_analyzer.generate_guidance(analysis)
                    return {
                        "response": f"Ticket validation failed:\n" + "\n".join(guidance["guidance"]),
                        "status": "needs_fix"
                    }

                state["awaiting_ticket_confirmation"] = True
                save_conversation_state()
                return {
                    "response": "Would you like to raise this ticket?",
                    "ticket": ticket_data,
                    "status": "awaiting_confirmation"
                }
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in LLM output")

        # --- Ticket confirmation (yes/no) ---
        if state["awaiting_ticket_confirmation"]:
            if user_input_clean in ["yes", "y", "create"]:
                state["awaiting_ticket_confirmation"] = False
                state["awaiting_attachment"] = True
                save_conversation_state()
                return {
                    "response": "Would you like to upload a screenshot, PDF, or link related to the issue? (yes/no)"
                }

            elif user_input_clean in ["no", "n"]:
                reset_ticket(state)
                return {"response": "Okay, ticket creation cancelled."}
            else:
                return {"response": "Please say 'yes' to raise the ticket or 'no' to cancel."}

        # --- Optional attachment handling ---
        if state["awaiting_attachment"]:
            if user_input_clean in ["no", "n"]:
                try:
                    jira_response = create_jira_ticket(state["finalized_ticket"])
                    state["jira_response"] = jira_response
                    reset_ticket(state)
                    return {
                        "response": "Ticket created. Thank you.",
                        "status": "created",
                        "jira": jira_response
                    }
                except HTTPException as e:
                    return {
                        "response": f"Failed to create Jira ticket: {str(e.detail)}",
                        "error": str(e.detail),
                        "status": "error"
                    }

            elif user_input_clean in ["yes", "y"]:
                state["awaiting_attachment"] = False
                state["awaiting_upload_confirmation"] = True
                save_conversation_state()
                return {"response": "Great! Please upload the file now. Once done, type 'done' to continue."}
            else:
                return {"response": "Please reply with 'yes' to upload or 'no' to skip attachment."}

        # --- Final confirmation after file upload ---
        if state["awaiting_upload_confirmation"]:
            if user_input_clean in ["done", "yes"]:
                try:
                    jira_response = create_jira_ticket(state["finalized_ticket"])
                    state["jira_response"] = jira_response
                    reset_ticket(state)
                    return {
                        "response": "Ticket created. Thank you.",
                        "status": "created",
                        "jira": jira_response
                    }
                except HTTPException as e:
                    return {
                        "response": f"Failed to create Jira ticket: {str(e.detail)}",
                        "error": str(e.detail),
                        "status": "error"
                    }
            else:
                return {"response": "Let me know once you're done uploading."}

        # -- Final fallback if none of the above --
        save_conversation_state()
        return {"response": response_text}

    except Exception as e:
        logger.error(f"Mistral error: {str(e)}")
        return {"error": f"Mistral failed: {str(e)}"}



# API Endpoints
@app.get("/chat")
async def chat(request: Request, user_input: str):
    conversation_id = get_conversation_id(request)
    response = chat_with_mistral(user_input, conversation_id)
    return JSONResponse(response)

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    conversation_id = get_conversation_id(request)
    state = conversation_states.get(conversation_id)
    if not state:
        raise HTTPException(status_code=404, detail="Conversation not found")

    ext = os.path.splitext(file.filename)[-1]
    filename = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(await file.read())

    url = f"/{UPLOAD_DIR}/{filename}"
    state["pending_fields"]["attachments"].append(url)
    if state["finalized_ticket"]:
        state["finalized_ticket"]["attachments"] = state["pending_fields"]["attachments"]

    # ✅ Mark that upload has happened and we can ask for "done" confirmation
    state["awaiting_upload_confirmation"] = True
    save_conversation_state()

    return JSONResponse({
        "message": "File uploaded successfully. Type 'done' or 'yes' in chat when you're ready to continue.",
        "file_url": url,
        "conversation_id": conversation_id
    })


# Exception handler
@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error at {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Initialize conversation states
conversation_states = load_conversation_state()
