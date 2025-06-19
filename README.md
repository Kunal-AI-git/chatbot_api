# 🤖 Chatbot API with JIRA Integration

This project is a FastAPI-based smart assistant that:
- Responds to user issues conversationally
- Checks similarity with past tickets (via FAISS)
- Validates ticket quality and fields
- Creates JIRA tickets (with attachments)
- Uses an LLM (Mistral via Ollama) for natural dialogue

---

## 🚀 Features

- 💬 Conversational flow using Mistral + FAISS
- 🧠 NLP-based ticket analysis with spaCy
- 🧾 JIRA ticket creation with priority/component inference
- 📎 Attachment upload support
- 🧠 Session-based memory & validation
- 🗃️ Auto-saves conversation history & states

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
Make sure to download the spaCy model:

bash

python -m spacy download en_core_web_sm

## 🔐 .env Configuration
Create a .env file with the following:

env

JIRA_DOMAIN=your-domain.atlassian.net
JIRA_EMAIL=your-email@example.com
JIRA_API_TOKEN=your-api-token
JIRA_PROJECT_KEY=PROJ
JIRA_ISSUE_TYPE=Task
SESSION_SECRET_KEY=your-random-secret-key

## 🧠 Preload FAISS Index (Optional)
If you want similar ticket suggestions:

Store ticket metadata in ticket_metadata.pkl

Build FAISS index and save it as ticket_index.faiss

## 🛠 Run the App
bash
uvicorn chatbot_api:app --reload
The API will be available at:
📍 http://127.0.0.1:8000

## 📂 Endpoints
Method	Endpoint	Description
GET	/chat	Start or continue chat session
POST	/upload	Upload attachment for the ticket

## 📁 Upload Directory
Uploaded files are stored in the /uploads/ folder and used as attachments when creating JIRA tickets.

## ✅ Flow Summary
User chats with assistant

Bot checks for similar issues using FAISS

If unresolved, collects details → validates → formats JSON

Asks user to confirm and upload file

Sends issue to JIRA and confirms ticket creation

## 🧪 Example Input
text
I’m unable to login to the dashboard. I enter my credentials, but nothing happens.
→ Assistant will extract issue
→ Ask for confirmation
→ Then create a JIRA ticket with file attachment.

## 👨‍💻 Credits
Developed by Kunal J — powered by:

🧠 Ollama + Mistral

🧮 FAISS

🌐 FastAPI

📌 JIRA Cloud API

🔍 spaCy NLP
