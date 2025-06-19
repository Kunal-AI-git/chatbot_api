## 🤖 AI Chatbot for JIRA Ticketing
This project is a smart conversational assistant built with FastAPI that helps users report issues, checks for similar historical tickets using FAISS, and creates JIRA tickets automatically. It supports file uploads, field validation, and multi-turn conversation with memory.

## 🚀 Features
🔍 FAISS-powered Similarity Search to detect duplicate issues

🧠 Mistral model via Ollama for natural dialogue generation

🛠️ Auto-field inference: Component, Issue Type, Priority

📁 Attachment support via /upload endpoint

✅ Ticket quality validation with field guidance

🧾 Automatic JIRA ticket creation with attachments

💬 Session-based memory for natural back-and-forth chat

🧠 Semantic summarization of resolutions using spaCy

## 📂 Project Structure
bash

chatbot_api.py            # Main FastAPI application
ticket_index.faiss        # FAISS index of previous ticket embeddings
ticket_metadata.pkl       # Metadata associated with indexed tickets
uploads/                  # Directory for user-uploaded files
.env                      # Environment variables for JIRA

## ⚙️ Environment Variables (.env)
ini
JIRA_DOMAIN=your-domain.atlassian.net
JIRA_EMAIL=your-email@example.com
JIRA_API_TOKEN=your-api-token
JIRA_PROJECT_KEY=PROJ
JIRA_ISSUE_TYPE=Task
SESSION_SECRET_KEY=your-random-secret-key

## 🧪 Setup & Run
1. Start Ollama (Mistral model)
bash

ollama run mistral

2. Launch API Server
bash

uvicorn chatbot_api:app --reload --port 8000

## 🔄 API Endpoints

/chat - Chat Interface
GET /chat?user_input=your+message

Responds to the user input and handles conversation context, field extraction, and ticket creation logic.

## 📎 /upload - File Upload
POST /upload
Body: multipart/form-data with key file

Uploads attachments (images, PDFs, etc.) linked to the current conversation's ticket.

## 📦 Sample Request Flow
http

GET /chat?user_input=hi
→ "Hi! How can I assist you today?"

GET /chat?user_input=The login page is broken on mobile
→ Searches for similar issues

GET /chat?user_input=no
→ Proceeds to create new ticket

GET /chat?user_input=high
→ Sets priority

GET /chat?user_input=yes
→ Prompt to upload files via /upload

POST /upload
→ Upload attachment

GET /chat?user_input=done
→ Finalizes and creates ticket

## ✅ Example Ticket Output
json

{
  "title": "Dockerfile build fails with permission denied",
  "issue": "Dockerfile build fails with permission denied during apt-get",
  "priority": "Low",
  "component": "Backend",
  "issuetype": "Bug",
  "attachments": ["/uploads/example.png"]
}

## 🧠 How It Works
Ollama Mistral generates natural responses and extracts missing fields

FAISS checks for semantically similar tickets

TicketAnalysisAgent validates the ticket before submission

Attachments are uploaded first, then finalized via user confirmation

JIRA Integration uses the official Python SDK
