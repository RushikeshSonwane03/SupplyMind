# SupplyMind: Agentic AI Operating System for Supply Chain Intelligence

<div align="center">

![SupplyMind Logo](https://img.shields.io/badge/SupplyMind-Agentic%20AI%20OS-6366f1?style=for-the-badge&logo=robot&logoColor=white)

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=react&logoColor=black)](https://reactjs.org/)
[![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=flat-square&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.2:latest.2:latest.2:latest-FF6B6B?style=flat-square)](https://ollama.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

**Transform reactive supply chains into autonomous decision-making ecosystems**

[Live Demo](https://agentic-supply.preview.emergentagent.com) · [Documentation](#documentation) · [API Reference](#api-reference) · [Contributing](#contributing)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Multi-Agent System](#multi-agent-system)
- [Supply Chain Workflows](#supply-chain-workflows)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Frontend Pages](#frontend-pages)
- [Database Schema](#database-schema)
- [Validation Layers](#validation-layers)
- [LLM Integration](#llm-integration)
- [Academic Deliverables](#academic-deliverables)
- [ROI Analysis](#roi-analysis)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**SupplyMind** is an Agentic AI Operating System that autonomously orchestrates global supply chains using multi-agent AI systems. Built as a comprehensive university capstone project, it demonstrates AI-First OS thinking, Agentic AI architecture, and multi-agent orchestration.

### Vision
Transform reactive supply chains into autonomous decision-making ecosystems that can predict, adapt, and optimize in real-time.

### Mission
Address critical supply chain challenges including:
- Demand volatility and forecasting accuracy
- Inventory imbalance across warehouses
- Supplier risk detection and mitigation
- Logistics delays and route optimization
- Rising operational costs
- Sustainability tracking

### Core Innovation
- **LLM-orchestrated multi-agent architecture** with specialized agents for demand, inventory, supplier, and action management
- **Hybrid ML + LLM decision systems** combining classical forecasting with generative AI reasoning
- **Real-time supply chain visibility** with autonomous exception handling
- **15-25% cost reduction** through optimized operations

---

## Key Features

### 1. Multi-Agent AI System
Five specialized AI agents working in coordination:
- **Orchestrator Agent**: Coordinates all agents, decomposes complex tasks
- **Demand Agent**: Forecasting, stockout prediction, anomaly detection
- **Inventory Agent**: Reorder points, safety stock, warehouse balancing
- **Supplier Agent**: Risk scoring, shipping optimization, vendor monitoring
- **Action Agent**: Purchase orders, alerts, reports, decision logging

### 2. 10 Supply Chain Workflows
Comprehensive coverage of supply chain operations:
1. Demand Forecasting
2. Inventory Optimization
3. Warehouse Automation
4. Route Optimization
5. Shipment Delay Prediction
6. Supplier Risk Detection
7. Procurement Automation
8. Logistics Cost Optimization
9. Production Planning
10. Sustainability Tracking

### 3. Interactive Supplier Risk Demo
- Real-time supplier risk analysis
- LLM-powered reasoning with Ollama llama3.2:latest.2:latest.2:latest
- Step-by-step agent activity visualization
- Risk scoring with actionable recommendations

### 4. Academic Deliverables
- Master Report with executive summary, problem statement, solution architecture
- 10-slide Pitch Deck for presentations
- ROI Analysis with detailed cost savings breakdown
- Patentability documentation

### 5. Analytics Dashboard
- Real-time KPIs with trend indicators
- Demand trend visualization
- Inventory levels by warehouse
- Supplier performance matrix
- Cost breakdown analysis
- ROI calculator

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SUPPLYMIND ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                      FRONTEND (React)                          │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │  │
│  │  │Dashboard│ │Workflows│ │ Agents  │ │Analytics│ │ Reports │ │  │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ │  │
│  └───────┼───────────┼───────────┼───────────┼───────────┼───────┘  │
│          │           │           │           │           │           │
│          └───────────┴───────────┼───────────┴───────────┘           │
│                                  │                                    │
│                          ┌───────▼───────┐                           │
│                          │   REST API    │                           │
│                          │   (FastAPI)   │                           │
│                          └───────┬───────┘                           │
│                                  │                                    │
│  ┌───────────────────────────────┼───────────────────────────────┐  │
│  │                    BACKEND SERVICES                            │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │              MULTI-AGENT ORCHESTRATOR                    │  │  │
│  │  │  ┌─────────────────────────────────────────────────┐    │  │  │
│  │  │  │              MESSAGE BUS (JSON State)            │    │  │  │
│  │  │  └──────┬──────┬──────┬──────┬──────┬──────────────┘    │  │  │
│  │  │         │      │      │      │      │                    │  │  │
│  │  │    ┌────▼──┐┌──▼───┐┌─▼────┐┌▼─────┐┌▼──────┐          │  │  │
│  │  │    │Orchest││Demand││Invent││Suppli││Action │          │  │  │
│  │  │    │rator  ││Agent ││ory   ││er    ││Agent  │          │  │  │
│  │  │    │Agent  ││      ││Agent ││Agent ││       │          │  │  │
│  │  │    └───────┘└──────┘└──────┘└──────┘└───────┘          │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │                              │                                  │  │
│  │  ┌───────────────────────────▼───────────────────────────────┐│  │
│  │  │                   VALIDATION LAYER                         ││  │
│  │  │  • Request Validation  • State Transition  • LLM Output   ││  │
│  │  └───────────────────────────────────────────────────────────┘│  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                  │                                    │
│          ┌───────────────────────┼───────────────────────┐           │
│          │                       │                       │           │
│   ┌──────▼──────┐        ┌───────▼───────┐       ┌──────▼──────┐   │
│   │   MongoDB   │        │  Ollama LLM   │       │   External  │   │
│   │  Database   │        │   (llama3.2:latest.2:latest.2:latest)    │       │    APIs     │   │
│   └─────────────┘        └───────────────┘       └─────────────┘   │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Multi-Agent System

### Agent Roles and Responsibilities

| Agent | Role | Tools | Responsibilities |
|-------|------|-------|------------------|
| **Orchestrator** | Coordinator | `coordinate_agents()`, `decompose_task()`, `manage_workflow()` | Task coordination & planning, Agent selection, Result aggregation |
| **Demand** | Forecaster | `forecast_product()`, `get_stockout_risk()`, `detect_demand_anomaly()` | Demand forecasting, Stockout prediction, Anomaly detection |
| **Inventory** | Optimizer | `compute_reorder()`, `check_warehouse_levels()`, `calculate_safety_stock()` | Reorder point calculation, Safety stock optimization, Warehouse balancing |
| **Supplier** | Risk Analyst | `score_supplier_risk()`, `get_best_shipping_mode()`, `flag_vendor()` | Supplier risk scoring, Shipping optimization, Vendor monitoring |
| **Action** | Executor | `raise_purchase_order()`, `generate_po_pdf()`, `send_alert()`, `log_decision()` | Execute purchase orders, Send alerts, Generate reports, Log decisions |

### Agent Communication Flow

```
User Request
     │
     ▼
┌─────────────┐
│ Orchestrator│──────────────────────────────────────┐
│    Agent    │                                      │
└──────┬──────┘                                      │
       │ Analyzes task, creates coordination plan    │
       │                                             │
       ▼                                             │
┌──────────────────────────────────────────────┐    │
│              MESSAGE BUS                      │    │
│  ┌─────────────────────────────────────────┐ │    │
│  │ {                                        │ │    │
│  │   "from": "orchestrator",               │ │    │
│  │   "to": "demand",                       │ │    │
│  │   "type": "analyze_request",            │ │    │
│  │   "content": {...}                      │ │    │
│  │ }                                        │ │    │
│  └─────────────────────────────────────────┘ │    │
└──────────────────────────────────────────────┘    │
       │                                             │
       ├─────────────┬─────────────┬────────────────┤
       ▼             ▼             ▼                │
┌─────────────┐┌─────────────┐┌─────────────┐      │
│   Demand    ││  Inventory  ││  Supplier   │      │
│   Agent     ││   Agent     ││   Agent     │      │
└──────┬──────┘└──────┬──────┘└──────┬──────┘      │
       │              │              │              │
       └──────────────┼──────────────┘              │
                      ▼                             │
               ┌─────────────┐                      │
               │   Action    │◄─────────────────────┘
               │   Agent     │
               └──────┬──────┘
                      │
                      ▼
              Final Decision/Action
```

### State Transitions

Valid agent state transitions:
- `IDLE` → `THINKING`, `WAITING`
- `THINKING` → `ACTING`, `IDLE`, `WAITING`
- `ACTING` → `IDLE`, `THINKING`
- `WAITING` → `IDLE`, `THINKING`

---

## Supply Chain Workflows

### Workflow Classification Matrix

| Workflow | Category | Type | Agents Involved |
|----------|----------|------|-----------------|
| Demand Forecasting | Customer | Efficiency | Orchestrator, Demand |
| Inventory Optimization | Operations | Efficiency | Orchestrator, Inventory, Demand |
| Warehouse Automation | Infrastructure | Execution | Orchestrator, Inventory, Action |
| Route Optimization | Operations | Efficiency | Orchestrator, Supplier, Action |
| Shipment Delay Prediction | Operations | Exception | Orchestrator, Supplier |
| Supplier Risk Detection | Business | Exception | Orchestrator, Supplier, Action |
| Procurement Automation | Business | Execution | Orchestrator, Supplier, Action |
| Logistics Cost Optimization | Business | Efficiency | Orchestrator, Supplier, Inventory |
| Production Planning | Operations | Execution | Orchestrator, Demand, Inventory |
| Sustainability Tracking | Sustainability | Expansion | Orchestrator, Supplier, Action |

### Deep Dive: Key Workflows

#### 1. Demand Forecasting
- **Baseline**: Manual forecasting using spreadsheets and historical averages
- **Static Data**: Historical sales data, product catalog, pricing information
- **Dynamic Data**: Real-time POS data, market trends, promotional calendars
- **ML Models**: ARIMA, Prophet, LSTM neural networks, Gradient Boosting
- **LLM Role**: Interpret forecast results, explain anomalies, suggest actions
- **Metrics**: MAPE, WMAPE, Forecast Bias, Forecast Value Added (FVA)

#### 2. Supplier Risk Detection
- **Baseline**: Annual supplier reviews and manual monitoring
- **Static Data**: Supplier profiles, financial data, certification records
- **Dynamic Data**: News feeds, social media, market data, performance metrics
- **ML Models**: NLP for sentiment analysis, anomaly detection, risk scoring
- **LLM Role**: Analyze news, summarize risks, recommend actions, draft alerts
- **Metrics**: Risk Score Accuracy, Early Warning Rate, Supplier Retention

---

## Tech Stack

### Backend
| Technology | Purpose | Version |
|------------|---------|---------|
| FastAPI | Web Framework | 0.110.1 |
| MongoDB | Database | 4.5.0 (pymongo) |
| Motor | Async MongoDB Driver | 3.3.1 |
| Pydantic | Data Validation | 2.6.4+ |
| Ollama | LLM Integration | 0.6.1 |
| HTTPX | Async HTTP Client | 0.28.1 |

### Frontend
| Technology | Purpose | Version |
|------------|---------|---------|
| React | UI Framework | 19.0.0 |
| React Router | Navigation | 7.5.1 |
| Recharts | Data Visualization | 3.6.0 |
| Framer Motion | Animations | 12.36.0 |
| Tailwind CSS | Styling | 3.4.17 |
| Shadcn/UI | Component Library | Latest |
| Axios | HTTP Client | 1.8.4 |
| Sonner | Toast Notifications | 2.0.3 |

### Infrastructure
| Technology | Purpose |
|------------|---------|
| Docker | Containerization |
| Nginx | Reverse Proxy |
| Supervisor | Process Management |

---

## Project Structure

```
supplymind/
├── backend/
│   ├── server.py              # Main FastAPI application
│   │   ├── Models             # Pydantic models for all entities
│   │   ├── OllamaService      # LLM integration service
│   │   ├── ValidationLayer    # Request/response validation
│   │   ├── AgentMessageBus    # Inter-agent communication
│   │   ├── MultiAgentOrch     # Agent orchestration logic
│   │   └── API Routes         # REST endpoints
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Environment variables
│
├── frontend/
│   ├── public/                # Static assets
│   ├── src/
│   │   ├── App.js             # Main app with routing
│   │   ├── App.css            # Component styles
│   │   ├── index.css          # Global styles & CSS variables
│   │   ├── index.js           # Entry point
│   │   ├── pages/
│   │   │   ├── Dashboard.js       # KPIs, charts, system status
│   │   │   ├── Workflows.js       # 10 workflows with details
│   │   │   ├── Agents.js          # Agent network visualization
│   │   │   ├── Analytics.js       # Deep analytics & ROI
│   │   │   ├── Reports.js         # Academic deliverables
│   │   │   └── SupplierRiskDemo.js # Interactive LLM demo
│   │   └── components/
│   │       └── ui/            # Shadcn UI components
│   ├── package.json           # Node dependencies
│   ├── tailwind.config.js     # Tailwind configuration
│   └── .env                   # Frontend environment
│
├── tests/                     # Test files
├── scripts/                   # Utility scripts
├── memory/
│   └── PRD.md                 # Product Requirements Document
└── README.md                  # This file
```

---

## Installation & Setup

### Prerequisites

- **Node.js** >= 18.x
- **Python** >= 3.11
- **MongoDB** >= 6.0
- **Yarn** >= 1.22
- **Ollama** (optional, for LLM features)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/supplymind.git
cd supplymind
```

### Step 2: Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file and configure
cp .env.example .env
# Edit .env with your configuration
```

### Step 3: Frontend Setup

```bash
# Navigate to frontend directory
cd ../frontend

# Install dependencies
yarn install

# Copy environment file
cp .env.example .env
# Edit .env with your backend URL
```

### Step 4: Database Setup

```bash
# Start MongoDB (if not running)
mongod --dbpath /path/to/data/db

# Or using Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

### Step 5: Ollama Setup (Optional)

```bash
# Install Ollama from https://ollama.com

# Pull llama3.2:latest.2:latest.2:latest model
ollama pull llama3.2:latest.2:latest.2:latest

# Start Ollama server
ollama serve
```

---

## Configuration

### Backend Environment Variables (.env)

```env
# MongoDB Configuration
MONGO_URL="mongodb://localhost:27017"
DB_NAME="supplymind_db"

# CORS Configuration
CORS_ORIGINS="*"

# Ollama LLM Configuration
OLLAMA_HOST="http://localhost:11434"
OLLAMA_MODEL="llama3.2:latest.2:latest.2:latest"
```

### Frontend Environment Variables (.env)

```env
# Backend API URL
REACT_APP_BACKEND_URL=http://localhost:8001

# WebSocket Port (for development)
WDS_SOCKET_PORT=443
```

---

## Running the Application

### Development Mode

#### Terminal 1: Start Backend
```bash
cd backend
source venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

#### Terminal 2: Start Frontend
```bash
cd frontend
yarn start
```

#### Terminal 3: Start Ollama (Optional)
```bash
ollama serve
```

### Production Mode

#### Using Docker Compose
```bash
docker-compose up -d
```

#### Manual Production Build
```bash
# Build frontend
cd frontend
yarn build

# Serve with nginx or serve package
npx serve -s build -l 3000

# Start backend with production server
cd ../backend
gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8001
```

### Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8001
- **API Documentation**: http://localhost:8001/docs
- **Ollama**: http://localhost:11434

---

## API Reference

### Base URL
```
http://localhost:8001/api
```

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API welcome message |
| GET | `/health` | System health check |
| GET | `/agents/states` | Get all agent states |
| GET | `/agents/messages` | Get agent message history |
| GET | `/workflows` | List all workflows |
| GET | `/workflows/{id}` | Get workflow details |
| POST | `/orchestrator/process` | Process task through agents |
| POST | `/supplier-risk/analyze` | Analyze supplier risk |
| GET | `/metrics/system` | Get system metrics |
| GET | `/metrics/kpis` | Get KPI data |
| GET | `/reports/master` | Get master report content |
| GET | `/reports/slides` | Get slide deck content |
| GET | `/inventory` | Get inventory items |
| GET | `/forecasts` | Get demand forecasts |
| GET | `/analytics/demand-trend` | Get demand trend data |
| GET | `/analytics/inventory-levels` | Get inventory levels |
| GET | `/analytics/supplier-performance` | Get supplier metrics |
| GET | `/analytics/cost-breakdown` | Get cost breakdown |

### Detailed API Documentation

#### Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "ollama_connected": true,
  "model": "llama3.2:latest.2:latest.2:latest",
  "timestamp": "2026-03-16T10:47:21.982259+00:00"
}
```

#### Analyze Supplier Risk
```http
POST /api/supplier-risk/analyze
Content-Type: application/json

{
  "supplier_name": "Acme Manufacturing",
  "include_news": true
}
```

**Response:**
```json
{
  "supplier_name": "Acme Manufacturing",
  "risk_level": "medium",
  "risk_score": 55.0,
  "risk_factors": [
    "Market volatility",
    "Supply chain disruptions",
    "Geopolitical factors"
  ],
  "news_summary": "Analysis based on simulated market news and AI reasoning",
  "llm_reasoning": "{\"analysis\": \"Based on simulated market analysis...\"}",
  "recommendation": "Maintain close monitoring and consider diversification",
  "timestamp": "2026-03-16T10:47:55.123456+00:00"
}
```

#### Process Orchestrator Task
```http
POST /api/orchestrator/process
Content-Type: application/json

{
  "task": "Analyze low inventory for Widget A and recommend reorder",
  "context": {
    "product_id": "SKU-001",
    "current_stock": 150,
    "reorder_point": 500
  }
}
```

**Response:**
```json
{
  "task_id": "uuid-here",
  "status": "completed",
  "orchestrator_reasoning": "...",
  "agent_responses": {
    "orchestrator": "...",
    "inventory": "...",
    "demand": "...",
    "action": "..."
  },
  "final_decision": "Task processed successfully",
  "actions_taken": ["Analysis complete", "Recommendations generated"],
  "timestamp": "2026-03-16T10:48:00.000000+00:00"
}
```

---

## Frontend Pages

### 1. Dashboard (`/`)
- **KPI Cards**: Cost Savings, On-Time Delivery, Inventory Turnover, etc.
- **Demand Trend Chart**: Actual vs Forecast comparison
- **System Status**: Active agents, workflows, decisions today
- **Inventory by Warehouse**: Bar chart visualization
- **Quick Actions**: Run Forecast, Check Inventory, etc.

### 2. Workflows (`/workflows`)
- **Workflow List**: 10 supply chain workflows with icons
- **Category Badges**: Customer, Operations, Infrastructure, Business, Sustainability
- **Detail Panel**: Description, metrics, agents involved
- **Technical Deep Dive**: Baseline, data sources, ML models, LLM role

### 3. Agents (`/agents`)
- **Agent Network Visualization**: Hierarchical view of 5 agents
- **Real-time Status**: Idle, Thinking, Acting, Waiting states
- **Message Bus**: Live feed of inter-agent communications
- **Agent Details**: Responsibilities, available tools

### 4. Analytics (`/analytics`)
- **Overview Tab**: Demand trends, inventory health
- **Suppliers Tab**: Performance matrix, radar chart
- **Costs Tab**: Pie chart breakdown, cost trends
- **ROI Tab**: Direct cost savings, operational improvements

### 5. Reports (`/reports`)
- **Master Report**: Collapsible sections with full content
- **Pitch Deck**: 10-slide presentation viewer
- **Download Options**: Export as text files
- **Table of Contents**: Quick navigation

### 6. Supplier Risk Demo (`/supplier-risk`)
- **Supplier Input**: Search with quick select options
- **Agent Activity**: Step-by-step analysis visualization
- **Risk Results**: Score, factors, recommendations
- **LLM Reasoning**: Raw output from Ollama llama3.2:latest.2:latest.2:latest

---

## Database Schema

### Collections

#### `orchestrator_tasks`
```javascript
{
  task_id: String,
  status: String,
  orchestrator_reasoning: String,
  agent_responses: Object,
  final_decision: String,
  actions_taken: Array,
  timestamp: ISODate
}
```

#### `supplier_risk_analyses`
```javascript
{
  supplier_name: String,
  risk_level: String,  // low, medium, high, critical
  risk_score: Number,
  risk_factors: Array,
  news_summary: String,
  llm_reasoning: String,
  recommendation: String,
  timestamp: ISODate
}
```

#### `status_checks`
```javascript
{
  id: String,
  client_name: String,
  timestamp: ISODate
}
```

---

## Validation Layers

### Request Validation
- Supplier name length (2-100 characters)
- Task description minimum length (5 characters)
- Required fields presence

### State Transition Validation
- Valid agent state transitions enforced
- Invalid transitions logged and prevented

### LLM Output Validation
- Empty response detection
- JSON format verification
- Error indicator detection
- Harmful content warnings

### Example Usage
```python
from server import ValidationLayer, validation_layer

# Validate supplier risk request
result = validation_layer.validate_supplier_risk_request(request)
if not result.is_valid:
    raise HTTPException(status_code=400, detail={"errors": result.errors})

# Validate LLM output
llm_validation = validation_layer.validate_llm_output(response, "json")
if llm_validation.warnings:
    logger.warning(f"LLM output warnings: {llm_validation.warnings}")
```

---

## LLM Integration

### Ollama Configuration
SupplyMind integrates with Ollama for local LLM inference using llama3.2:latest.2:latest.2:latest.

### Service Architecture
```python
class OllamaService:
    def __init__(self, host: str, model: str):
        self.host = host      # Default: http://localhost:11434
        self.model = model    # Default: llama3.2:latest.2:latest.2:latest
    
    async def chat(self, messages, temperature=0.7) -> str:
        # Send chat request to Ollama
        # Falls back to mock response if Ollama unavailable
    
    async def health_check(self) -> bool:
        # Check if Ollama server is running
```

### Mock Fallback
When Ollama is unavailable, the system provides intelligent mock responses:
- Risk analysis with predefined factors
- Demand forecasts with confidence intervals
- Inventory optimization recommendations

### Connecting Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull llama3.2:latest.2:latest.2:latest

# Verify connection
curl http://localhost:11434/api/tags
```

---

## Academic Deliverables

### Master Report Sections
1. **Executive Summary**: Project overview and key innovations
2. **Problem Statement**: Supply chain challenges addressed
3. **Solution Architecture**: Multi-agent system design
4. **10 Workflows**: Comprehensive workflow documentation
5. **ROI Analysis**: Cost savings and business impact
6. **Patentability**: Novel innovations for IP protection

### Pitch Deck (10 Slides)
1. Title & Vision
2. The Problem
3. Our Solution
4. Multi-Agent Architecture
5. 10 Workflows
6. Technical Innovation
7. Business Impact
8. ROI Analysis
9. Roadmap
10. Team & Ask

---

## ROI Analysis

### Projected ROI for Mid-Size Enterprise ($500M Revenue)

| Category | Savings | Description |
|----------|---------|-------------|
| Inventory Reduction | $2.5M | 15% reduction in carrying costs |
| Logistics Optimization | $1.8M | 12% transportation savings |
| Procurement Efficiency | $800K | Reduced maverick spending |
| **Total Annual Benefit** | **$5.1M** | |

### Operational Improvements
- 40% faster response to disruptions
- 25% improvement in forecast accuracy
- 60% reduction in manual planning effort

### Financial Summary
- **Implementation Cost**: $1.2M
- **Payback Period**: 2.8 months
- **3-Year ROI**: 425%

---

## Roadmap

### Year 0-1: MVP
- [x] Multi-agent architecture
- [x] Supplier risk detection
- [x] Demand forecasting workflow
- [x] Basic dashboard
- [ ] Production Ollama deployment

### Year 1-2: Full Suite
- [ ] Inventory optimization automation
- [ ] Route optimization with real-time tracking
- [ ] ERP integrations (SAP, Oracle)
- [ ] Advanced analytics with ML predictions

### Year 2-3: Autonomous OS
- [ ] Autonomous procurement execution
- [ ] Predictive logistics management
- [ ] Full supply chain digital twin
- [ ] Self-healing supply chain capabilities

---

## Contributing

We welcome contributions! Please follow these steps:

### 1. Fork the Repository
```bash
git fork https://github.com/yourusername/supplymind.git
```

### 2. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes
- Follow existing code style
- Add tests for new features
- Update documentation

### 4. Commit Changes
```bash
git commit -m "feat: add your feature description"
```

### 5. Push and Create PR
```bash
git push origin feature/your-feature-name
# Create Pull Request on GitHub
```

### Code Style
- **Python**: Follow PEP 8, use type hints
- **JavaScript**: Use ES6+, functional components
- **CSS**: Use Tailwind utility classes

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Ollama** for local LLM inference
- **Shadcn/UI** for beautiful React components
- **FastAPI** for high-performance Python backend
- **MongoDB** for flexible document storage

---

<div align="center">

**Built with passion for supply chain innovation**

[⬆ Back to Top](#supplymind-agentic-ai-operating-system-for-supply-chain-intelligence)

</div>
