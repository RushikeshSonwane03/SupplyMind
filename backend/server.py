from fastapi import FastAPI, APIRouter, HTTPException, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uuid
from datetime import datetime, timezone
from enum import Enum
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

from datasets import (
    get_products, get_suppliers, get_warehouses, get_news_events,
    get_demand_by_product, get_inventory_by_warehouse, get_inventory_by_product,
    get_supplier_performance, get_shipments_by_supplier, get_supplier_risk_data,
    get_demand_summary, get_inventory_summary, get_cost_breakdown,
    DEMAND_DATA, INVENTORY_DATA, SHIPMENT_DATA, SUPPLIER_PERFORMANCE_DATA,
    PRODUCTS, SUPPLIERS, WAREHOUSES
)
from langchain_agents import agent_system, ALL_TOOLS

app = FastAPI(title="SupplyMind API")
api_router = APIRouter(prefix="/api")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== ENUMS & MODELS =====================

class AgentRole(str, Enum):
    ORCHESTRATOR   = "orchestrator"
    DEMAND         = "demand"
    INVENTORY      = "inventory"
    SUPPLIER       = "supplier"
    ACTION         = "action"
    LOGISTICS      = "logistics"
    PLANNING       = "planning"
    SUSTAINABILITY = "sustainability"

class AgentStatus(str, Enum):
    IDLE     = "idle"
    THINKING = "thinking"
    ACTING   = "acting"
    WAITING  = "waiting"

class OrchestratorRequest(BaseModel):
    task: str
    context: Optional[Dict[str, Any]] = None

class BusinessQueryRequest(BaseModel):
    query: str

class SupplierRiskRequest(BaseModel):
    supplier_id: str
    include_news: bool = True

class DemandForecastRequest(BaseModel):
    product_id: str
    periods: int = 3

class InventoryAnalysisRequest(BaseModel):
    product_id: Optional[str] = None
    warehouse_id: Optional[str] = None

class ToolInvocationRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]

class WorkflowRunRequest(BaseModel):
    params: Optional[Dict[str, Any]] = {}

# ===================== PRIORITY HELPERS =====================

PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

def sort_by_priority(items: list, key: str = "priority") -> list:
    return sorted(items, key=lambda x: PRIORITY_ORDER.get(x.get(key, "low"), 3))

def needs_attention(items: list, key: str = "priority") -> list:
    return [i for i in items if i.get(key) in ("critical", "high")]

# ===================== AGENT STATE =====================

agent_states = {role: {"status": AgentStatus.IDLE, "current_task": None, "last_action": None} for role in AgentRole}
agent_messages = []

def update_agent_state(agent: AgentRole, status: AgentStatus, task: str = None):
    agent_states[agent]["status"] = status
    agent_states[agent]["current_task"] = task
    if task:
        agent_states[agent]["last_action"] = datetime.now(timezone.utc).isoformat()

def add_agent_message(from_agent: str, to_agent: str, message_type: str, content: str):
    agent_messages.append({"id": str(uuid.uuid4()), "from_agent": from_agent, "to_agent": to_agent,
                            "message_type": message_type, "content": content[:300],
                            "timestamp": datetime.now(timezone.utc).isoformat()})
    if len(agent_messages) > 100:
        agent_messages.pop(0)

# ===================== WORKFLOWS =====================

WORKFLOWS = [
    {"id": "wf-1",  "name": "Demand Forecasting",          "description": "Forecast demand for ALL 10 products. Flags high-risk stockout products first.",                       "category": "customer",       "workflow_type": "efficiency", "agents_involved": ["orchestrator", "demand"],             "tools": ["forecast_demand", "analyze_stockout_risk"],            "metrics": {"products_analysed": 10, "forecast_accuracy": 0.915, "mape_avg": 8.5, "months_history": 24}},
    {"id": "wf-2",  "name": "Inventory Optimization",      "description": "EOQ and safety stock for ALL 10 products across all 4 warehouses, sorted by urgency.",                "category": "operations",     "workflow_type": "efficiency", "agents_involved": ["orchestrator", "inventory", "demand"],    "tools": ["calculate_reorder_point", "check_warehouse_levels"],   "metrics": {"products_analysed": 10, "warehouses": 4, "fill_rate": 0.968, "stockout_events": 3}},
    {"id": "wf-3",  "name": "Warehouse Automation",        "description": "Scan ALL 4 warehouses for capacity and critical SKUs. Generates priority-ranked alerts.",              "category": "infrastructure", "workflow_type": "execution",  "agents_involved": ["orchestrator", "inventory", "action"],    "tools": ["check_warehouse_levels"],                              "metrics": {"warehouses": 4, "avg_utilization": 74, "critical_alerts": 3, "health_score": 82}},
    {"id": "wf-4",  "name": "Route Optimization",          "description": "Optimise shipping modes for ALL 8 suppliers. Flags expensive lanes for re-routing.",                  "category": "operations",     "workflow_type": "efficiency", "agents_involved": ["orchestrator", "supplier", "action"],     "tools": ["get_shipping_options"],                                "metrics": {"suppliers_analysed": 8, "avg_transit_days": 8, "cost_saving_pct": 12, "on_time_delivery": 0.934}},
    {"id": "wf-5",  "name": "Shipment Delay Prediction",   "description": "Predict delay risk for ALL 8 suppliers using history and live news. Sorted by delay probability.",   "category": "operations",     "workflow_type": "exception",  "agents_involved": ["orchestrator", "supplier"],               "tools": ["predict_shipment_delay", "get_market_intelligence"],   "metrics": {"suppliers_analysed": 8, "prediction_accuracy": 0.84, "avg_delay_days": 3.2, "high_risk_lanes": 2}},
    {"id": "wf-6",  "name": "Supplier Risk Detection",     "description": "Score ALL 8 suppliers across reliability, delivery, quality, and financial health.",                  "category": "business",       "workflow_type": "exception",  "agents_involved": ["orchestrator", "supplier", "action"],     "tools": ["score_supplier_risk", "get_market_intelligence"],      "metrics": {"suppliers_monitored": 8, "high_risk_count": 2, "avg_risk_score": 32, "news_events_tracked": 8}},
    {"id": "wf-7",  "name": "Procurement Automation",      "description": "Auto-generate POs for ALL critical and low inventory items using the safest supplier.",              "category": "business",       "workflow_type": "execution",  "agents_involved": ["orchestrator", "supplier", "action"],     "tools": ["generate_purchase_order", "score_supplier_risk"],      "metrics": {"pos_generated": 12, "avg_po_value": 45000, "auto_approval_rate": 0.72, "lead_time_saved_days": 3}},
    {"id": "wf-8",  "name": "Logistics Cost Optimization", "description": "Identify dollar-value saving opportunities across all cost categories and shipping lanes.",           "category": "business",       "workflow_type": "efficiency", "agents_involved": ["orchestrator", "supplier", "inventory"],  "tools": ["optimize_logistics_cost", "get_shipping_options"],     "metrics": {"monthly_logistics_cost": 950000, "saving_potential_pct": 9.2, "opportunities_found": 4}},
    {"id": "wf-9",  "name": "Production Planning",         "description": "8-week production schedules for ALL 10 products. Sorted by most urgent capacity needs.",             "category": "operations",     "workflow_type": "execution",  "agents_involved": ["orchestrator", "demand", "inventory"],    "tools": ["generate_production_plan", "forecast_demand"],         "metrics": {"products_planned": 10, "planning_horizon_weeks": 8, "avg_utilization_pct": 71, "alerts_generated": 2}},
    {"id": "wf-10", "name": "Sustainability Tracking",     "description": "Carbon footprint, ESG news, and ISO 14001 compliance for all suppliers.",                           "category": "sustainability",  "workflow_type": "expansion",  "agents_involved": ["orchestrator", "supplier", "action"],     "tools": ["track_sustainability", "get_market_intelligence"],     "metrics": {"sustainability_score": 68, "co2_tonnes_estimated": 2840, "iso14001_suppliers": 3, "green_actions": 5}},
]

# ===================== BASIC ROUTES =====================

@api_router.get("/")
async def root():
    return {"message": "SupplyMind API", "version": "2.0.0", "llm_active": agent_system.llm is not None}

@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "ollama_connected": agent_system.llm is not None,
            "model": agent_system.model, "tools_available": len(ALL_TOOLS),
            "agents_available": 8, "timestamp": datetime.now(timezone.utc).isoformat()}

@api_router.get("/products")
async def list_products():
    return {"products": PRODUCTS, "count": len(PRODUCTS)}

@api_router.get("/suppliers")
async def list_suppliers():
    return {"suppliers": SUPPLIERS, "count": len(SUPPLIERS)}

@api_router.get("/suppliers/{supplier_id}")
async def get_supplier(supplier_id: str):
    s = next((s for s in SUPPLIERS if s["id"] == supplier_id), None)
    if not s: raise HTTPException(404, "Supplier not found")
    return {"supplier": s, "performance_history": get_supplier_performance(supplier_id)[-12:],
            "recent_shipments": get_shipments_by_supplier(supplier_id)[-10:]}

@api_router.get("/warehouses")
async def list_warehouses():
    return {"warehouses": WAREHOUSES, "count": len(WAREHOUSES)}

@api_router.get("/warehouses/{warehouse_id}")
async def get_warehouse(warehouse_id: str):
    w = next((w for w in WAREHOUSES if w["id"] == warehouse_id), None)
    if not w: raise HTTPException(404, "Warehouse not found")
    inv = get_inventory_by_warehouse(warehouse_id)
    return {"warehouse": w, "inventory_items": len(inv), "inventory": inv}

@api_router.get("/demand")
async def get_demand_data(product_id: Optional[str] = Query(None), months: int = Query(12)):
    data = get_demand_by_product(product_id)
    return {"demand_data": data, "count": len(data)}

@api_router.post("/demand/forecast")
async def forecast_demand_endpoint(request: DemandForecastRequest):
    update_agent_state(AgentRole.DEMAND, AgentStatus.THINKING, f"Forecasting {request.product_id}")
    try:
        result = await agent_system.invoke_tool("forecast_demand", product_id=request.product_id, periods=request.periods)
        update_agent_state(AgentRole.DEMAND, AgentStatus.IDLE)
        await db.agent_results.insert_one({"type": "forecast", "result": result, "timestamp": datetime.now(timezone.utc).isoformat()})
        return result
    except Exception as e:
        update_agent_state(AgentRole.DEMAND, AgentStatus.IDLE)
        raise HTTPException(500, str(e))

@api_router.get("/demand/summary")
async def get_demand_summary_data():
    return {"summary": get_demand_summary()}

@api_router.get("/inventory")
async def get_inventory(warehouse_id: Optional[str] = Query(None), product_id: Optional[str] = Query(None), status: Optional[str] = Query(None)):
    if warehouse_id: data = get_inventory_by_warehouse(warehouse_id)
    elif product_id: data = get_inventory_by_product(product_id)
    else: data = INVENTORY_DATA.to_dict(orient="records")
    if status: data = [d for d in data if d.get("status") == status]
    return {"inventory": data, "count": len(data)}

@api_router.post("/inventory/analyze")
async def analyze_inventory(request: InventoryAnalysisRequest):
    update_agent_state(AgentRole.INVENTORY, AgentStatus.THINKING, "Analyzing")
    results = {}
    if request.product_id:
        results["stockout_analysis"] = await agent_system.invoke_tool("analyze_stockout_risk", product_id=request.product_id, warehouse_id=request.warehouse_id)
        results["reorder_analysis"]  = await agent_system.invoke_tool("calculate_reorder_point", product_id=request.product_id)
    if request.warehouse_id:
        results["warehouse_analysis"] = await agent_system.invoke_tool("check_warehouse_levels", warehouse_id=request.warehouse_id)
    update_agent_state(AgentRole.INVENTORY, AgentStatus.IDLE)
    return results

@api_router.get("/inventory/summary")
async def get_inventory_summary_data():
    return {"summary": get_inventory_summary()}

@api_router.post("/supplier-risk/analyze")
async def analyze_supplier_risk(request: SupplierRiskRequest):
    update_agent_state(AgentRole.SUPPLIER, AgentStatus.THINKING, f"Analyzing {request.supplier_id}")
    try:
        risk_result     = await agent_system.invoke_tool("score_supplier_risk", supplier_id=request.supplier_id)
        news_result     = None
        if request.include_news:
            s = next((s for s in SUPPLIERS if s["id"] == request.supplier_id), None)
            if s: news_result = await agent_system.invoke_tool("get_market_intelligence", topic=s["region"])
        shipping_result = await agent_system.invoke_tool("get_shipping_options", supplier_id=request.supplier_id, urgency="normal")
        agent_response  = await agent_system.run_agent("supplier", f"Analyze risk for {request.supplier_id}", {"risk_score": risk_result})
        update_agent_state(AgentRole.SUPPLIER, AgentStatus.IDLE)
        result = {"supplier_id": request.supplier_id, "risk_assessment": risk_result,
                  "market_intelligence": news_result, "shipping_options": shipping_result,
                  "agent_insights": agent_response, "timestamp": datetime.now(timezone.utc).isoformat()}
        await db.supplier_risk_analyses.insert_one(result.copy())
        return result
    except Exception as e:
        update_agent_state(AgentRole.SUPPLIER, AgentStatus.IDLE)
        raise HTTPException(500, str(e))

@api_router.get("/supplier-risk/quick/{supplier_id}")
async def quick_supplier_risk(supplier_id: str):
    return await agent_system.invoke_tool("score_supplier_risk", supplier_id=supplier_id)

# ===================== BUSINESS QUERY — the main AI intelligence endpoint =====================

@api_router.post("/business/query")
async def business_query(request: BusinessQueryRequest):
    """
    Natural language business intelligence endpoint.
    Activates relevant agents, calls real tools, uses LLM to reason,
    returns priority-ranked actions + 7-day plan.
    """
    if not request.query or len(request.query.strip()) < 3:
        raise HTTPException(400, "Query must be at least 3 characters")

    # Set orchestrator + all likely agents to thinking immediately
    update_agent_state(AgentRole.ORCHESTRATOR, AgentStatus.THINKING, request.query[:60])
    # Pre-activate common agents based on query keywords so status bar lights up right away
    q_lower = request.query.lower()
    preactivate = []
    if any(kw in q_lower for kw in ["today","priority","focus","urgent","week","plan","morning","what should"]):
        preactivate = ["demand","inventory","supplier","logistics"]
    else:
        if any(kw in q_lower for kw in ["demand","forecast","stockout"]): preactivate.append("demand")
        if any(kw in q_lower for kw in ["inventory","stock","warehouse"]): preactivate.append("inventory")
        if any(kw in q_lower for kw in ["supplier","vendor","risk","delay"]): preactivate.append("supplier")
        if any(kw in q_lower for kw in ["cost","logistics","freight"]): preactivate.append("logistics")
        if any(kw in q_lower for kw in ["production","plan","schedule"]): preactivate.append("planning")
        if any(kw in q_lower for kw in ["sustainability","carbon","esg"]): preactivate.append("sustainability")
    for agent_name in preactivate:
        try:
            role = AgentRole(agent_name)
            update_agent_state(role, AgentStatus.THINKING, request.query[:60])
        except ValueError:
            pass

    try:
        # Run the full business query through the agent system
        result = await agent_system.business_query(request.query)

        # Update agent states and populate the message bus with every step
        for agent_name in result.get("agents_activated", []):
            # Orchestrator → agent delegation
            add_agent_message(
                "orchestrator", agent_name,
                "task_delegation",
                f"Query: {request.query[:80]}"
            )

        # Log each tool call as a message (tool → agent direction)
        tool_agent_map = {
            "forecast_demand":         "demand",
            "analyze_stockout_risk":   "demand",
            "calculate_reorder_point": "inventory",
            "check_warehouse_levels":  "inventory",
            "score_supplier_risk":     "supplier",
            "predict_shipment_delay":  "supplier",
            "get_shipping_options":    "supplier",
            "get_market_intelligence": "supplier",
            "generate_purchase_order": "action",
            "optimize_logistics_cost": "logistics",
            "generate_production_plan":"planning",
            "track_sustainability":    "sustainability",
        }
        for tool_name in result.get("tools_called", []):
            owning_agent = tool_agent_map.get(tool_name, "orchestrator")
            add_agent_message(
                owning_agent, owning_agent,
                f"tool_call:{tool_name}",
                f"{tool_name}() executed"
            )

        # Agent → orchestrator completion messages with insight summary
        for agent_name, insight in result.get("agent_insights", {}).items():
            analysis_snippet = insight.get("analysis", "")[:150]
            add_agent_message(
                agent_name, "orchestrator",
                "analysis_complete",
                analysis_snippet if analysis_snippet else "Analysis complete"
            )

        # Orchestrator final synthesis message
        action_count = len(result.get("priority_actions", []))
        add_agent_message(
            "orchestrator", "orchestrator",
            "synthesis_complete",
            f"Synthesised {action_count} priority actions from {len(result.get('agents_activated',[]))} agents"
        )

        # Update agent states precisely based on what the orchestrator actually activated
        for agent_name in result.get("agents_activated", []):
            try:
                role = AgentRole(agent_name)
                update_agent_state(role, AgentStatus.ACTING, f"Processing: {request.query[:50]}")
            except ValueError:
                pass

        # Reset all states to IDLE now that response is ready
        for role in AgentRole:
            update_agent_state(role, AgentStatus.IDLE)

        # Persist to MongoDB
        await db.business_queries.insert_one({
            "query": request.query,
            "domains": result.get("domains_analysed", []),
            "agents": result.get("agents_activated", []),
            "action_count": len(result.get("priority_actions", [])),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        return result

    except Exception as e:
        for role in AgentRole:
            update_agent_state(role, AgentStatus.IDLE)
        logger.error(f"Business query failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))

# ===================== ORCHESTRATOR =====================

@api_router.post("/orchestrator/process")
async def process_orchestrator_task(request: OrchestratorRequest):
    update_agent_state(AgentRole.ORCHESTRATOR, AgentStatus.THINKING, request.task)
    try:
        result = await agent_system.orchestrate(request.task, request.context)
        update_agent_state(AgentRole.ORCHESTRATOR, AgentStatus.IDLE)
        await db.orchestrator_tasks.insert_one(result.copy())
        return result
    except Exception as e:
        update_agent_state(AgentRole.ORCHESTRATOR, AgentStatus.IDLE)
        raise HTTPException(500, str(e))

# ===================== TOOLS =====================

@api_router.get("/tools")
async def list_tools():
    return {"tools": [{"name": t.name, "description": t.description,
                       "agent": next((a for a, tools in {
                           "demand": ["forecast_demand","analyze_stockout_risk"],
                           "inventory": ["calculate_reorder_point","check_warehouse_levels"],
                           "supplier": ["score_supplier_risk","get_shipping_options","get_market_intelligence","predict_shipment_delay"],
                           "action": ["generate_purchase_order"],
                           "logistics": ["optimize_logistics_cost"],
                           "planning": ["generate_production_plan"],
                           "sustainability": ["track_sustainability"]
                       }.items() if t.name in tools), "orchestrator")}
                      for t in ALL_TOOLS], "count": len(ALL_TOOLS)}

@api_router.post("/tools/invoke")
async def invoke_tool(request: ToolInvocationRequest):
    result = await agent_system.invoke_tool(request.tool_name, **request.parameters)
    return {"tool": request.tool_name, "result": result}

# ===================== AGENT STATE ROUTES =====================

@api_router.get("/agents/states")
async def get_agent_states():
    return {"agents": {
        role.value: {"status": state["status"].value if isinstance(state["status"], AgentStatus) else state["status"],
                     "current_task": state["current_task"], "last_action": state["last_action"]}
        for role, state in agent_states.items()}}

@api_router.get("/agents/messages")
async def get_agent_messages(limit: int = 50):
    return {"messages": agent_messages[-limit:], "count": len(agent_messages[-limit:])}

# ===================== ANALYTICS =====================

@api_router.get("/analytics/demand-trend")
async def get_demand_trend(product_id: Optional[str] = None):
    if product_id: data = DEMAND_DATA[DEMAND_DATA["product_id"] == product_id].tail(12)
    else: data = DEMAND_DATA.groupby("month").agg({"actual_demand":"sum","forecast_demand":"sum"}).reset_index().tail(12)
    return {"data": [{"month": row.get("month",str(row.name)), "actual": int(row["actual_demand"]), "forecast": int(row["forecast_demand"])} for _,row in data.iterrows()]}

@api_router.get("/analytics/inventory-levels")
async def get_inventory_levels():
    return {"data": get_inventory_summary()}

@api_router.get("/analytics/supplier-performance")
async def get_supplier_perf_analytics(supplier_id: Optional[str] = None):
    data = SUPPLIER_PERFORMANCE_DATA[SUPPLIER_PERFORMANCE_DATA["supplier_id"] == supplier_id] if supplier_id else SUPPLIER_PERFORMANCE_DATA.groupby("supplier_id").last().reset_index()
    result = []
    for _, row in data.iterrows():
        s = next((s for s in SUPPLIERS if s["id"] == row["supplier_id"]), None)
        result.append({"supplier": row.get("supplier_name", s["name"] if s else "Unknown"),
                       "quality": row["quality_score"], "delivery": row["delivery_score"],
                       "cost": row["cost_score"], "risk": 100 - row["overall_score"]})
    return {"data": result}

@api_router.get("/analytics/cost-breakdown")
async def get_cost_analytics():
    data = get_cost_breakdown()
    latest = max(d["month"] for d in data)
    return {"data": [d for d in data if d["month"] == latest], "month": latest}

@api_router.get("/analytics/shipments")
async def get_shipment_analytics(supplier_id: Optional[str] = None):
    data = SHIPMENT_DATA[SHIPMENT_DATA["supplier_id"] == supplier_id] if supplier_id else SHIPMENT_DATA
    total = len(data); on_time = len(data[data["delay_days"] == 0]); delayed = len(data[data["delay_days"] > 0])
    avg_delay = data[data["delay_days"] > 0]["delay_days"].mean() if delayed else 0
    return {"total_shipments": total, "on_time": on_time, "delayed": delayed,
            "on_time_rate": round(on_time/total*100,1) if total else 0,
            "avg_delay_days": round(avg_delay,1),
            "delay_reasons": data[data["delay_days"] > 0]["delay_reason"].value_counts().to_dict()}

# ===================== METRICS =====================

@api_router.get("/metrics/system")
async def get_system_metrics():
    records  = INVENTORY_DATA.to_dict(orient="records")
    critical = len([i for i in records if i["status"] == "Critical"])
    total    = len(records)
    return {"total_workflows": 10, "active_agents": 8,
            "decisions_today": await db.orchestrator_tasks.count_documents({}),
            "risk_analyses": await db.supplier_risk_analyses.count_documents({}),
            "workflow_runs": await db.workflow_runs.count_documents({}),
            "business_queries": await db.business_queries.count_documents({}),
            "inventory_health": round(((total-critical)/total*100),1) if total else 0,
            "products_tracked": len(PRODUCTS), "suppliers_monitored": len(SUPPLIERS),
            "warehouses": len(WAREHOUSES), "llm_active": agent_system.llm is not None}

@api_router.get("/metrics/kpis")
async def get_kpis():
    on_time_rate = SHIPMENT_DATA[SHIPMENT_DATA["delay_days"] == 0].shape[0] / len(SHIPMENT_DATA) * 100
    inv_value    = INVENTORY_DATA["inventory_value"].sum()
    return {"kpis": [
        {"name": "Cost Savings",      "value": "$234.5K",              "change": 12.5, "trend": "up"},
        {"name": "On-Time Delivery",  "value": f"{on_time_rate:.1f}%", "change": 2.3,  "trend": "up"},
        {"name": "Inventory Value",   "value": f"${inv_value/1e6:.1f}M","change": -5.2, "trend": "down"},
        {"name": "Supplier Risk Avg", "value": "32%",                  "change": -4.1, "trend": "up"},
        {"name": "Forecast Accuracy", "value": "91.5%",                "change": 3.1,  "trend": "up"},
        {"name": "Fill Rate",         "value": "96.8%",                "change": 1.2,  "trend": "up"},
    ]}

@api_router.get("/reports/master")
async def get_master_report():
    return {"title": "SupplyMind Master Report",
            "sections": [{"id": "executive-summary", "title": "Executive Summary",
                          "content": "SupplyMind autonomously orchestrates supply chains using 8 LangChain agents and 12 tools across 10 workflows."}],
            "generated_at": datetime.now(timezone.utc).isoformat()}

@api_router.get("/reports/slides")
async def get_slides_content():
    return {"title": "SupplyMind Pitch Deck",
            "slides": [{"number": 1, "title": "SupplyMind", "subtitle": "Agentic AI OS", "bullets": ["8 agents","12 tools","10 workflows"]}],
            "generated_at": datetime.now(timezone.utc).isoformat()}

@api_router.get("/news")
async def get_news():
    return {"news": get_news_events()}

# ===================== WORKFLOWS =====================

@api_router.get("/workflows")
async def get_workflows():
    return {"workflows": WORKFLOWS, "count": len(WORKFLOWS)}

@api_router.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    wf = next((w for w in WORKFLOWS if w["id"] == workflow_id), None)
    if not wf: raise HTTPException(404, "Workflow not found")
    return wf

@api_router.post("/workflows/{workflow_id}/run")
async def run_workflow(workflow_id: str, request: WorkflowRunRequest):
    wf = next((w for w in WORKFLOWS if w["id"] == workflow_id), None)
    if not wf: raise HTTPException(404, "Workflow not found")

    result = {"workflow_id": workflow_id, "workflow_name": wf["name"],
              "started_at": datetime.now(timezone.utc).isoformat(),
              "status": "running", "steps": [], "outputs": {}}

    def step(msg: str):
        result["steps"].append({"step": msg, "status": "complete"})

    try:
        # ── WF-1 ─────────────────────────────────────────────────────────────
        if workflow_id == "wf-1":
            update_agent_state(AgentRole.DEMAND, AgentStatus.THINKING, "WF-1")
            step("Forecasting demand for all 10 products")
            fc_map = {}
            for p in PRODUCTS:
                fc = await agent_system.invoke_tool("forecast_demand", product_id=p["id"], periods=3)
                if "error" not in fc: fc_map[p["id"]] = fc
            step("Analysing stockout risk for all 10 products")
            combined = []
            for p in PRODUCTS:
                sk = await agent_system.invoke_tool("analyze_stockout_risk", product_id=p["id"])
                if "error" in sk: continue
                highest  = sk.get("highest_risk", {})
                priority = highest.get("risk_level", "low")
                fc = fc_map.get(p["id"], {})
                combined.append({"product_id": p["id"], "product_name": p["name"], "priority": priority,
                                  "highest_risk_warehouse": highest.get("warehouse"), "days_of_supply": highest.get("days_of_supply"),
                                  "current_stock": highest.get("current_stock"), "recommendation": highest.get("recommendation"),
                                  "trend_pct": fc.get("trend_pct"), "historical_avg": fc.get("historical_avg"),
                                  "mape": fc.get("mape"), "seasonality_detected": fc.get("seasonality_detected"),
                                  "next_3_periods": fc.get("forecasts", []), "all_warehouse_risks": sk.get("stockout_risks", [])})
            combined = sort_by_priority(combined)
            step("Running Demand Agent LLM analysis")
            agent_resp = await agent_system.run_agent("demand", "Analyse demand across all products", {"forecasts": combined})
            step("Compiling priority report")
            result["outputs"] = {"all_products": combined,
                                  "needs_attention": needs_attention(combined),
                                  "agent_analysis": agent_resp.get("analysis", ""),
                                  "agent_recommendations": agent_resp.get("recommendations", []),
                                  "summary": {"critical": len([x for x in combined if x["priority"]=="critical"]),
                                              "high": len([x for x in combined if x["priority"]=="high"]),
                                              "medium": len([x for x in combined if x["priority"]=="medium"]),
                                              "low": len([x for x in combined if x["priority"]=="low"]),
                                              "total_products": len(combined)}}
            update_agent_state(AgentRole.DEMAND, AgentStatus.IDLE)

        # ── WF-2 ─────────────────────────────────────────────────────────────
        elif workflow_id == "wf-2":
            update_agent_state(AgentRole.INVENTORY, AgentStatus.THINKING, "WF-2")
            step("Calculating EOQ and reorder points for all 10 products")
            all_reorders = []
            for p in PRODUCTS:
                rr = await agent_system.invoke_tool("calculate_reorder_point", product_id=p["id"], service_level=0.95)
                if "error" in rr: continue
                inv = get_inventory_by_product(p["id"])
                total_stock = sum(i["current_stock"] for i in inv)
                rp = rr.get("reorder_point", 0); ss = rr.get("safety_stock", 0)
                priority = "critical" if total_stock < ss else "high" if total_stock < rp else "medium" if total_stock < rp*1.3 else "low"
                all_reorders.append({**rr, "priority": priority, "current_total_stock": total_stock})
            all_reorders = sort_by_priority(all_reorders)
            step("Checking all 4 warehouse levels")
            warehouse_results = []
            for wh in WAREHOUSES:
                wr = await agent_system.invoke_tool("check_warehouse_levels", warehouse_id=wh["id"])
                if "error" in wr: continue
                util = wr.get("capacity_utilization", 0); crit = wr.get("summary", {}).get("critical_items", 0)
                wh_prio = "critical" if crit > 3 or util > 90 else "high" if crit > 1 or util > 80 else "medium" if crit > 0 or util > 70 else "low"
                warehouse_results.append({**wr, "priority": wh_prio})
            warehouse_results = sort_by_priority(warehouse_results)
            step("Running Inventory Agent LLM analysis")
            agent_resp = await agent_system.run_agent("inventory", "Optimise inventory across all products", {"reorders": all_reorders, "critical_items_count": len([r for r in all_reorders if r["priority"]=="critical"])})
            result["outputs"] = {"product_reorder_analysis": all_reorders, "warehouse_levels": warehouse_results,
                                  "needs_attention": needs_attention(all_reorders) + needs_attention(warehouse_results),
                                  "agent_analysis": agent_resp.get("analysis",""),
                                  "agent_recommendations": agent_resp.get("recommendations",[]),
                                  "summary": {"products_critical": len([x for x in all_reorders if x["priority"]=="critical"]),
                                              "products_high": len([x for x in all_reorders if x["priority"]=="high"]),
                                              "products_medium": len([x for x in all_reorders if x["priority"]=="medium"]),
                                              "products_low": len([x for x in all_reorders if x["priority"]=="low"]),
                                              "warehouses_critical": len([w for w in warehouse_results if w["priority"]=="critical"])}}
            update_agent_state(AgentRole.INVENTORY, AgentStatus.IDLE)

        # ── WF-3 ─────────────────────────────────────────────────────────────
        elif workflow_id == "wf-3":
            update_agent_state(AgentRole.INVENTORY, AgentStatus.THINKING, "WF-3")
            step("Scanning all 4 warehouses")
            warehouse_scans = []; all_alerts = []
            for wh in WAREHOUSES:
                lvl = await agent_system.invoke_tool("check_warehouse_levels", warehouse_id=wh["id"])
                if "error" in lvl: continue
                s = lvl.get("summary",{}); util = lvl.get("capacity_utilization",0); crit = s.get("critical_items",0); over = s.get("overstock_items",0)
                alerts = []
                if crit > 0:  alerts.append({"level":"critical","message":f"{crit} SKU(s) at critical level — immediate reorder"})
                if util > 90: alerts.append({"level":"high","message":f"Warehouse at {util}% capacity — overflow routing needed"})
                if over > 0:  alerts.append({"level":"medium","message":f"{over} SKU(s) overstocked — review purchasing"})
                if util < 50: alerts.append({"level":"low","message":f"Low utilization ({util}%) — consolidation opportunity"})
                top = min((PRIORITY_ORDER.get(a["level"],3) for a in alerts), default=3)
                top_label = next((k for k,v in PRIORITY_ORDER.items() if v==top), "low")
                warehouse_scans.append({**lvl, "priority": top_label, "automation_alerts": alerts})
                for a in alerts: all_alerts.append({**a, "priority": a["level"], "warehouse": lvl.get("warehouse_name")})
            warehouse_scans = sort_by_priority(warehouse_scans); all_alerts = sort_by_priority(all_alerts)
            step("Generating priority alerts")
            result["outputs"] = {"warehouses": warehouse_scans, "all_alerts": all_alerts,
                                  "needs_attention": needs_attention(warehouse_scans),
                                  "summary": {"total_warehouses": len(warehouse_scans),
                                              "critical_warehouses": len([w for w in warehouse_scans if w["priority"]=="critical"]),
                                              "total_alerts": len(all_alerts),
                                              "critical_alerts": len([a for a in all_alerts if a["level"]=="critical"])}}
            update_agent_state(AgentRole.INVENTORY, AgentStatus.IDLE)

        # ── WF-4 ─────────────────────────────────────────────────────────────
        elif workflow_id == "wf-4":
            update_agent_state(AgentRole.SUPPLIER, AgentStatus.THINKING, "WF-4")
            step("Fetching shipping options for all 8 suppliers")
            all_routes = []
            for sup in SUPPLIERS:
                opts = await agent_system.invoke_tool("get_shipping_options", supplier_id=sup["id"], urgency="normal")
                if "error" in opts: continue
                rec = opts.get("recommended_option",{}); cf = rec.get("cost_factor",1.0); days = rec.get("transit_days",7)
                priority = "high" if cf >= 3.0 or days >= 20 else "medium" if cf >= 2.0 or days >= 10 else "low"
                all_routes.append({"supplier_id": sup["id"], "supplier_name": sup["name"], "region": sup["region"],
                                    "priority": priority, "recommended_mode": rec.get("mode"),
                                    "transit_days": days, "cost_factor": cf, "reliability_pct": rec.get("reliability"),
                                    "co2_kg_per_tonne": rec.get("co2_kg_per_tonne"), "all_options": opts.get("all_options",[])})
            all_routes = sort_by_priority(all_routes)
            step("Running Supplier Agent analysis on routes")
            agent_resp = await agent_system.run_agent("supplier", "Optimise shipping routes across all suppliers", {"routes": all_routes})
            result["outputs"] = {"all_routes": all_routes, "needs_attention": needs_attention(all_routes),
                                  "agent_analysis": agent_resp.get("analysis",""),
                                  "agent_recommendations": agent_resp.get("recommendations",[]),
                                  "summary": {"high_cost_lanes": len([r for r in all_routes if r["priority"]=="high"]),
                                              "medium_cost_lanes": len([r for r in all_routes if r["priority"]=="medium"]),
                                              "optimal_lanes": len([r for r in all_routes if r["priority"]=="low"]),
                                              "avg_transit_days": round(sum(r["transit_days"] for r in all_routes)/len(all_routes),1) if all_routes else 0}}
            update_agent_state(AgentRole.SUPPLIER, AgentStatus.IDLE)

        # ── WF-5 ─────────────────────────────────────────────────────────────
        elif workflow_id == "wf-5":
            update_agent_state(AgentRole.SUPPLIER, AgentStatus.THINKING, "WF-5")
            step("Running delay prediction for all 8 suppliers")
            all_preds = []
            for sup in SUPPLIERS:
                dp = await agent_system.invoke_tool("predict_shipment_delay", supplier_id=sup["id"], destination_warehouse=WAREHOUSES[0]["id"])
                if "error" not in dp: all_preds.append({**dp, "priority": dp.get("risk_level","low")})
            all_preds = sort_by_priority(all_preds)
            step("Fetching market intelligence")
            news = await agent_system.invoke_tool("get_market_intelligence", topic="shipping")
            step("Running Supplier Agent LLM analysis")
            agent_resp = await agent_system.run_agent("supplier", "Analyse shipment delay risks across all suppliers",
                                                       {"predictions": all_preds, "news": news, "high_risk_count": len([x for x in all_preds if x["priority"] in ("critical","high")])})
            result["outputs"] = {"all_suppliers": all_preds, "needs_attention": needs_attention(all_preds),
                                  "market_context": news,
                                  "agent_analysis": agent_resp.get("analysis",""),
                                  "agent_recommendations": agent_resp.get("recommendations",[]),
                                  "summary": {"critical_suppliers": len([x for x in all_preds if x["priority"]=="critical"]),
                                              "high_risk_suppliers": len([x for x in all_preds if x["priority"]=="high"]),
                                              "medium_risk_suppliers": len([x for x in all_preds if x["priority"]=="medium"]),
                                              "low_risk_suppliers": len([x for x in all_preds if x["priority"]=="low"]),
                                              "avg_delay_probability_pct": round(sum(x.get("prediction",{}).get("delay_probability_pct",0) for x in all_preds)/len(all_preds),1) if all_preds else 0}}
            update_agent_state(AgentRole.SUPPLIER, AgentStatus.IDLE)

        # ── WF-6 ─────────────────────────────────────────────────────────────
        elif workflow_id == "wf-6":
            update_agent_state(AgentRole.SUPPLIER, AgentStatus.THINKING, "WF-6")
            step("Scoring all 8 suppliers")
            all_risks = []
            for sup in SUPPLIERS:
                rk = await agent_system.invoke_tool("score_supplier_risk", supplier_id=sup["id"])
                if "error" not in rk: all_risks.append({**rk, "priority": rk.get("risk_level","low")})
            all_risks = sort_by_priority(all_risks)
            step("Fetching market intelligence")
            news = await agent_system.invoke_tool("get_market_intelligence", topic="geopolitical")
            step("Running Supplier Agent LLM analysis")
            agent_resp = await agent_system.run_agent("supplier", "Assess risk across all suppliers",
                                                       {"risks": all_risks, "news": news,
                                                        "high_risk_count": len([r for r in all_risks if r["priority"] in ("critical","high")])})
            result["outputs"] = {"supplier_risks": all_risks, "needs_attention": needs_attention(all_risks),
                                  "market_intelligence": news,
                                  "agent_analysis": agent_resp.get("analysis",""),
                                  "agent_recommendations": agent_resp.get("recommendations",[]),
                                  "summary": {"critical": len([x for x in all_risks if x["priority"]=="critical"]),
                                              "high": len([x for x in all_risks if x["priority"]=="high"]),
                                              "medium": len([x for x in all_risks if x["priority"]=="medium"]),
                                              "low": len([x for x in all_risks if x["priority"]=="low"]),
                                              "avg_risk_score": round(sum(x.get("risk_score",0) for x in all_risks)/len(all_risks),1) if all_risks else 0}}
            update_agent_state(AgentRole.SUPPLIER, AgentStatus.IDLE)

        # ── WF-7 ─────────────────────────────────────────────────────────────
        elif workflow_id == "wf-7":
            update_agent_state(AgentRole.ACTION, AgentStatus.ACTING, "WF-7")
            step("Finding all critical and low inventory items")
            critical_items = INVENTORY_DATA[INVENTORY_DATA["status"] == "Critical"].to_dict(orient="records")
            low_items      = INVENTORY_DATA[INVENTORY_DATA["status"] == "Low"].to_dict(orient="records")
            step("Scoring all suppliers to select lowest-risk")
            risk_scores = {}
            for sup in SUPPLIERS:
                rk = await agent_system.invoke_tool("score_supplier_risk", supplier_id=sup["id"])
                risk_scores[sup["id"]] = {"score": rk.get("risk_score",100), "name": rk.get("supplier_name",sup["id"])}
            best_sup = min(risk_scores, key=lambda k: risk_scores[k]["score"])
            step("Generating purchase orders for all critical + low items")
            pos = []
            for item in critical_items:
                po = await agent_system.invoke_tool("generate_purchase_order", product_id=item["product_id"], quantity=item["reorder_point"]*2, supplier_id=best_sup)
                if "error" not in po: pos.append({**po, "priority": "critical", "trigger": "Critical stock level"})
            for item in low_items[:5]:
                po = await agent_system.invoke_tool("generate_purchase_order", product_id=item["product_id"], quantity=item["reorder_point"], supplier_id=best_sup)
                if "error" not in po: pos.append({**po, "priority": "high", "trigger": "Low stock level"})
            pos = sort_by_priority(pos)
            step("Running Action Agent LLM analysis")
            agent_resp = await agent_system.run_agent("action", "Review and recommend purchase orders",
                                                       {"purchase_orders": pos[:5], "critical_items_count": len(critical_items)})
            result["outputs"] = {"purchase_orders": pos, "needs_attention": needs_attention(pos),
                                  "selected_supplier": {"id": best_sup, **risk_scores[best_sup]},
                                  "supplier_risk_scores": risk_scores,
                                  "agent_analysis": agent_resp.get("analysis",""),
                                  "agent_recommendations": agent_resp.get("recommendations",[]),
                                  "summary": {"critical_pos": len([p for p in pos if p["priority"]=="critical"]),
                                              "high_pos": len([p for p in pos if p["priority"]=="high"]),
                                              "total_pos": len(pos),
                                              "total_value": round(sum(p.get("total_cost",0) for p in pos),2),
                                              "critical_items_found": len(critical_items), "low_items_found": len(low_items)}}
            update_agent_state(AgentRole.ACTION, AgentStatus.IDLE)

        # ── WF-8 ─────────────────────────────────────────────────────────────
        elif workflow_id == "wf-8":
            update_agent_state(AgentRole.LOGISTICS, AgentStatus.THINKING, "WF-8")
            step("Analysing logistics cost breakdown")
            cost = await agent_system.invoke_tool("optimize_logistics_cost", time_horizon_months=3, focus_category="all")
            opps = cost.get("optimization_opportunities", [])
            for opp in opps: opp["priority"] = "high" if opp.get("potential_saving_pct",0) >= 10 else "medium" if opp.get("potential_saving_pct",0) >= 6 else "low"
            cost["optimization_opportunities"] = sort_by_priority(opps)
            step("Comparing shipping modes for all 8 suppliers")
            shipping_comparison = []
            for sup in SUPPLIERS:
                n = await agent_system.invoke_tool("get_shipping_options", supplier_id=sup["id"], urgency="normal")
                e = await agent_system.invoke_tool("get_shipping_options", supplier_id=sup["id"], urgency="economy")
                nr = n.get("recommended_option",{}); er = e.get("recommended_option",{})
                saving = round(nr.get("cost_factor",1) - er.get("cost_factor",1), 2)
                shipping_comparison.append({"supplier": sup["name"], "region": sup["region"],
                                             "priority": "high" if saving >= 1.5 else "medium" if saving >= 0.5 else "low",
                                             "normal_mode": nr.get("mode"), "normal_days": nr.get("transit_days"), "normal_cost_factor": nr.get("cost_factor"),
                                             "economy_mode": er.get("mode"), "economy_days": er.get("transit_days"), "economy_cost_factor": er.get("cost_factor"),
                                             "cost_factor_saving": saving})
            shipping_comparison = sort_by_priority(shipping_comparison)
            step("Running Logistics Agent LLM analysis")
            agent_resp = await agent_system.run_agent("logistics", "Identify all logistics cost savings",
                                                       {"cost_analysis": cost, "shipping_comparison": shipping_comparison[:3]})
            result["outputs"] = {"cost_analysis": cost, "shipping_comparison": shipping_comparison,
                                  "needs_attention": needs_attention(opps) + needs_attention(shipping_comparison),
                                  "agent_analysis": agent_resp.get("analysis",""),
                                  "agent_recommendations": agent_resp.get("recommendations",[]),
                                  "summary": {"total_cost": cost.get("total_logistics_cost"),
                                              "total_saving_potential": cost.get("total_potential_savings_usd"),
                                              "saving_pct": cost.get("potential_saving_pct"),
                                              "high_priority_opportunities": len([o for o in opps if o["priority"]=="high"]),
                                              "high_saving_lanes": len([s for s in shipping_comparison if s["priority"]=="high"])}}
            update_agent_state(AgentRole.LOGISTICS, AgentStatus.IDLE)

        # ── WF-9 ─────────────────────────────────────────────────────────────
        elif workflow_id == "wf-9":
            update_agent_state(AgentRole.PLANNING, AgentStatus.THINKING, "WF-9")
            step("Generating 8-week production plans for all 10 products")
            all_plans = []
            for p in PRODUCTS:
                plan = await agent_system.invoke_tool("generate_production_plan", product_id=p["id"], planning_horizon_weeks=8)
                if "error" in plan: continue
                weeks = plan.get("weekly_schedule",[]); cw = len([w for w in weeks if w.get("status")=="critical"]); lw = len([w for w in weeks if w.get("status")=="low"])
                priority = "critical" if cw >= 2 else "high" if cw == 1 else "medium" if lw >= 2 else "low"
                all_plans.append({**plan, "priority": priority, "critical_weeks": cw, "low_weeks": lw})
            all_plans = sort_by_priority(all_plans)
            step("Calculating reorder points for all products")
            reorder_map = {}
            for p in PRODUCTS:
                rr = await agent_system.invoke_tool("calculate_reorder_point", product_id=p["id"])
                reorder_map[p["id"]] = rr
            step("Running Planning Agent LLM analysis")
            agent_resp = await agent_system.run_agent("planning", "Identify production gaps and schedule optimisation",
                                                       {"plans_summary": [{"product": p["product_name"], "priority": p["priority"], "alerts": p.get("alerts",[])} for p in all_plans[:5]]})
            result["outputs"] = {"all_products": all_plans, "needs_attention": needs_attention(all_plans),
                                  "reorder_analysis": reorder_map,
                                  "agent_analysis": agent_resp.get("analysis",""),
                                  "agent_recommendations": agent_resp.get("recommendations",[]),
                                  "summary": {"critical_products": len([x for x in all_plans if x["priority"]=="critical"]),
                                              "high_products": len([x for x in all_plans if x["priority"]=="high"]),
                                              "medium_products": len([x for x in all_plans if x["priority"]=="medium"]),
                                              "low_products": len([x for x in all_plans if x["priority"]=="low"]),
                                              "total_planned_units": sum(p.get("summary",{}).get("total_planned_production",0) for p in all_plans),
                                              "total_alerts": sum(len(p.get("alerts",[])) for p in all_plans)}}
            update_agent_state(AgentRole.PLANNING, AgentStatus.IDLE)

        # ── WF-10 ────────────────────────────────────────────────────────────
        elif workflow_id == "wf-10":
            update_agent_state(AgentRole.SUSTAINABILITY, AgentStatus.THINKING, "WF-10")
            step("Computing global carbon footprint")
            sus = await agent_system.invoke_tool("track_sustainability", region="Global")
            step("Scanning ESG news")
            esg = await agent_system.invoke_tool("get_market_intelligence", topic="sustainability")
            step("Checking supplier ESG compliance")
            supplier_esg = []
            for sup in SUPPLIERS:
                has_iso = "ISO 14001" in sup.get("certifications",[])
                priority = "high" if not has_iso and sup["region"]=="Asia Pacific" else "medium" if not has_iso else "low"
                supplier_esg.append({"supplier_id": sup["id"], "supplier_name": sup["name"], "region": sup["region"],
                                      "priority": priority, "iso_14001": has_iso, "certifications": sup.get("certifications",[]),
                                      "action_needed": "Require ISO 14001 certification" if not has_iso else "Compliant"})
            supplier_esg = sort_by_priority(supplier_esg)
            actions = sus.get("green_action_plan",[])
            for a in actions: a["priority"] = "high" if a.get("co2_reduction_pct",0) >= 10 else "medium" if a.get("co2_reduction_pct",0) >= 5 else "low"
            sus["green_action_plan"] = sort_by_priority(actions)
            score = sus.get("sustainability_score",0)
            step("Running Sustainability Agent LLM analysis")
            agent_resp = await agent_system.run_agent("sustainability", "Assess sustainability and recommend improvements",
                                                       {"score": score, "non_compliant_suppliers": len([s for s in supplier_esg if s["priority"] in ("high","medium")]),
                                                        "air_share_pct": sus.get("carbon_footprint",{}).get("air_freight_share_pct")})
            result["outputs"] = {"sustainability_overview": {**sus, "priority": "critical" if score < 50 else "high" if score < 65 else "medium" if score < 80 else "low"},
                                  "supplier_esg_compliance": supplier_esg, "esg_news": esg,
                                  "needs_attention": needs_attention(supplier_esg) + [a for a in actions if a["priority"]=="high"],
                                  "agent_analysis": agent_resp.get("analysis",""),
                                  "agent_recommendations": agent_resp.get("recommendations",[]),
                                  "summary": {"sustainability_score": score,
                                              "non_compliant_suppliers": len([s for s in supplier_esg if s["priority"] in ("high","medium")]),
                                              "compliant_suppliers": len([s for s in supplier_esg if s["priority"]=="low"]),
                                              "high_impact_actions": len([a for a in actions if a["priority"]=="high"]),
                                              "estimated_co2_tonnes": sus.get("carbon_footprint",{}).get("total_estimated_co2_tonnes")}}
            update_agent_state(AgentRole.SUSTAINABILITY, AgentStatus.IDLE)

        else:
            raise HTTPException(404, f"No handler for {workflow_id}")

        result["status"] = "completed"
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        await db.workflow_runs.insert_one({k:v for k,v in result.items() if k != "_id"})
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workflow {workflow_id} failed: {e}", exc_info=True)
        result["status"] = "failed"; result["error"] = str(e)
        return result


app.include_router(api_router)
app.add_middleware(CORSMiddleware, allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS','*').split(','),
    allow_methods=["*"], allow_headers=["*"])

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
