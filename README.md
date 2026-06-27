# ✈️ VoyageAI Backend

VoyageAI Backend is an **AI-powered travel planning service** built with **FastAPI**, **LangGraph**, and **Groq LLMs**. It leverages a **graph-based multi-agent architecture** with **8 specialized AI agents**, real-time API integrations, and intelligent tool orchestration to generate personalized travel recommendations, itineraries, hotels, attractions, restaurants, weather insights, and transport guidance.

---

## 🚀 Features

- 🤖 Graph-based multi-agent system using **LangGraph**
- 🧠 Powered by **Groq LLMs**
- 🗺️ Personalized destination recommendations
- 🏨 Intelligent hotel recommendations
- 🎯 Attraction discovery and ranking
- 🍽️ Restaurant and local cuisine recommendations
- 🚗 Transport planning
- 🌤️ Real-time weather information
- 📅 AI-generated day-wise itineraries
- 🖼️ Destination, hotel, restaurant, and attraction image retrieval
- ⚡ Real-time travel data through external APIs

---

## 🏗️ System Architecture

The backend consists of **8 specialized AI agents** working collaboratively through a graph-based workflow.

1. Destination Agent
2. Hotel Agent
3. Attraction Agent
4. Food Agent
5. Transport Agent
6. Weather Agent
7. Itinerary Agent
8. Budget Agent

Each agent specializes in a specific planning task and exchanges structured information to generate a complete travel plan.

---

## 🛠️ Tech Stack

### Backend
- FastAPI
- Python

### AI Framework
- LangGraph
- LangChain
- Groq LLM
- RAG

### Databases
- ChromaDB

### External APIs
- OpenTripMap API
- Geoapify API
- OpenWeather API
- Pexels API

---

## 📁 Project Structure

```
app/
│
├── agents/
├── api/
├── config/
├── graph/
├── tools/
├── utils/
└── app.py
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/satyabrata-mishra/VoyageAI-Backend.git
```

Move into the project

```bash
cd VoyageAI-Backend
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file and configure the following keys:

```env
GROQ_API_KEY=
OPENTRIPMAP_API_KEY=
OPENWEATHER_API_KEY=
GEOAPIFY_API_KEY=
PEXELS_API_KEY=
MONGODB_URI=
DATABASE_URL=
```

---

## ▶️ Running the Server

```bash
uvicorn app.app:app --reload
```

Server runs on

```
http://localhost:8000
```

Interactive API documentation

```
http://localhost:8000/docs
```

---

## 📌 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | Root endpoint |
| `/api/health` | Health check |
| `/api/travel` | Generate personalized travel plan |

---

## 🌍 Real-Time Integrations

- OpenTripMap for destinations and attractions
- Geoapify for hotels and restaurants
- OpenWeather for weather forecasts
- Pexels for destination and place images

---

## 🎯 Highlights

- Graph-based multi-agent orchestration
- Autonomous AI workflow execution
- Intelligent tool calling
- Shared state management
- Dynamic task routing
- Real-time API integration
- Modular and scalable architecture

---

## 👨‍💻 Author

**Satyabrata Mishra**

---