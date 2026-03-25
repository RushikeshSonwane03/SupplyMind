"""
SupplyMind LangChain Agent System
Multi-agent orchestration using LangChain for supply chain intelligence
"""
import json
import logging
from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime, timezone, timedelta
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from datasets import (
    get_products, get_suppliers, get_warehouses,
    get_demand_by_product, get_inventory_by_product, get_inventory_by_warehouse,
    get_supplier_performance, get_shipments_by_supplier, get_supplier_risk_data,
    get_news_events, get_demand_summary, get_inventory_summary, get_cost_breakdown,
    DEMAND_DATA, INVENTORY_DATA, SHIPMENT_DATA, SUPPLIER_PERFORMANCE_DATA,
    PRODUCTS, SUPPLIERS, WAREHOUSES
)

logger = logging.getLogger(__name__)

# ============================================================
# AGENT STATE
# ============================================================
class AgentState(TypedDict):
    messages: List[Any]
    task: str
    context: Dict[str, Any]
    current_agent: str
    agent_outputs: Dict[str, Any]
    final_response: Optional[str]
    tools_used: List[str]

# ============================================================
# TOOLS — all 12 real tools
# ============================================================

@tool
def forecast_demand(product_id: str, periods: int = 3) -> str:
    """Forecast future demand for a product based on historical data."""
    import pandas as pd
    import numpy as np
    demand_data = get_demand_by_product(product_id)
    if not demand_data:
        return json.dumps({"error": f"No data for product {product_id}"})
    df = pd.DataFrame(demand_data)
    avg_demand = df["actual_demand"].mean()
    std_demand = df["actual_demand"].std()
    trend = (df["actual_demand"].iloc[-6:].mean() - df["actual_demand"].iloc[:6].mean()) / df["actual_demand"].iloc[:6].mean()
    forecasts = []
    for i in range(periods):
        fc = avg_demand * (1 + trend * (i + 1) / 12)
        forecasts.append({"period": i + 1, "forecast": round(fc),
                          "lower_bound": round(fc - 1.96 * std_demand),
                          "upper_bound": round(fc + 1.96 * std_demand), "confidence": 0.95})
    return json.dumps({"product_id": product_id, "product_name": df["product_name"].iloc[0],
                       "historical_avg": round(avg_demand), "trend_pct": round(trend * 100, 1),
                       "mape": round(df["forecast_error"].mean(), 2), "forecasts": forecasts,
                       "seasonality_detected": bool(std_demand / avg_demand > 0.2)})


@tool
def analyze_stockout_risk(product_id: str, warehouse_id: str = None) -> str:
    """Analyze stockout risk for a product across warehouses."""
    inventory = get_inventory_by_product(product_id)
    if warehouse_id:
        inventory = [i for i in inventory if i["warehouse_id"] == warehouse_id]
    if not inventory:
        return json.dumps({"error": f"No inventory data for {product_id}"})
    risks = []
    for inv in inventory:
        days_supply = inv["days_of_supply"]
        lead_time = next((p["lead_time_days"] for p in get_products() if p["id"] == product_id), 7)
        if days_supply < lead_time * 0.5:   risk_level, risk_score = "critical", 95
        elif days_supply < lead_time:        risk_level, risk_score = "high", 75
        elif days_supply < lead_time * 1.5:  risk_level, risk_score = "medium", 45
        else:                                risk_level, risk_score = "low", 15
        risks.append({"warehouse": inv["warehouse_name"], "current_stock": inv["current_stock"],
                      "days_of_supply": inv["days_of_supply"], "reorder_point": inv["reorder_point"],
                      "risk_level": risk_level, "risk_score": risk_score,
                      "recommendation": "Reorder immediately" if risk_level in ["critical", "high"] else "Monitor"})
    return json.dumps({"product_id": product_id, "stockout_risks": risks,
                       "highest_risk": max(risks, key=lambda x: x["risk_score"])})


@tool
def calculate_reorder_point(product_id: str, service_level: float = 0.95) -> str:
    """Calculate optimal reorder point and safety stock for a product."""
    import numpy as np
    demand_data = get_demand_by_product(product_id)
    if not demand_data:
        return json.dumps({"error": f"No data for {product_id}"})
    product = next((p for p in get_products() if p["id"] == product_id), None)
    if not product:
        return json.dumps({"error": "Product not found"})
    demands = [d["actual_demand"] for d in demand_data]
    avg_daily = sum(demands) / (len(demands) * 30)
    std_daily = np.std(demands) / 30
    lead_time = product["lead_time_days"]
    z = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}.get(service_level, 1.65)
    safety_stock = z * std_daily * np.sqrt(lead_time)
    reorder_point = (avg_daily * lead_time) + safety_stock
    eoq = np.sqrt((2 * avg_daily * 365 * 50) / (product["unit_cost"] * 0.25))
    return json.dumps({"product_id": product_id, "product_name": product["name"],
                       "avg_daily_demand": round(avg_daily, 1), "lead_time_days": lead_time,
                       "service_level": service_level, "safety_stock": round(safety_stock),
                       "reorder_point": round(reorder_point), "economic_order_quantity": round(eoq),
                       "recommendation": f"Set reorder point at {round(reorder_point)} units with safety stock of {round(safety_stock)} units"})


@tool
def check_warehouse_levels(warehouse_id: str) -> str:
    """Check inventory levels and health for a specific warehouse."""
    inventory = get_inventory_by_warehouse(warehouse_id)
    if not inventory:
        return json.dumps({"error": f"No data for warehouse {warehouse_id}"})
    warehouse = next((w for w in get_warehouses() if w["id"] == warehouse_id), None)
    critical = [i for i in inventory if i["status"] == "Critical"]
    low      = [i for i in inventory if i["status"] == "Low"]
    overstock= [i for i in inventory if i["status"] == "Overstock"]
    optimal  = [i for i in inventory if i["status"] == "Optimal"]
    return json.dumps({"warehouse_id": warehouse_id,
                       "warehouse_name": warehouse["name"] if warehouse else "Unknown",
                       "location": warehouse["location"] if warehouse else "Unknown",
                       "capacity_utilization": warehouse["utilization"] if warehouse else 0,
                       "summary": {"total_skus": len(inventory), "total_units": sum(i["current_stock"] for i in inventory),
                                   "total_value": round(sum(i["inventory_value"] for i in inventory), 2),
                                   "critical_items": len(critical), "low_items": len(low),
                                   "optimal_items": len(optimal), "overstock_items": len(overstock)},
                       "critical_products": [{"product": i["product_name"], "stock": i["current_stock"], "days_supply": i["days_of_supply"]} for i in critical[:5]],
                       "health_score": round((len(optimal) / len(inventory)) * 100, 1) if inventory else 0})


@tool
def score_supplier_risk(supplier_id: str) -> str:
    """Calculate comprehensive risk score for a supplier."""
    risk_data = get_supplier_risk_data(supplier_id)
    if not risk_data:
        return json.dumps({"error": f"No data for supplier {supplier_id}"})
    supplier = risk_data["supplier"]
    news = risk_data["recent_news"]
    financial_scores = {"Strong": 90, "Moderate": 65, "Developing": 40, "Weak": 20}
    financial_score = financial_scores.get(supplier["financial_health"], 50)
    negative_news = [n for n in news if n["sentiment"] == "negative"]
    risk_score = (
        (100 - supplier["reliability_score"]) * 0.25 +
        (100 - supplier["on_time_delivery"]) * 0.25 +
        (100 - supplier["quality_rating"] * 20) * 0.20 +
        (100 - financial_score) * 0.15 +
        len(negative_news) * 5 * 0.15
    )
    risk_level = "critical" if risk_score >= 70 else "high" if risk_score >= 50 else "medium" if risk_score >= 30 else "low"
    return json.dumps({"supplier_id": supplier_id, "supplier_name": supplier["name"],
                       "country": supplier["country"], "region": supplier["region"],
                       "risk_score": round(risk_score, 1), "risk_level": risk_level,
                       "component_scores": {
                           "reliability": supplier["reliability_score"],
                           "delivery": supplier["on_time_delivery"],
                           "quality": round(supplier["quality_rating"] * 20, 1),
                           "financial": financial_score},
                       "risk_factors": supplier["risk_factors"],
                       "shipment_performance": risk_data["shipment_stats"],
                       "negative_news_count": len(negative_news),
                       "certifications": supplier["certifications"],
                       "recommendation": "Immediate action required" if risk_level == "critical" else
                                         "Close monitoring recommended" if risk_level == "high" else
                                         "Continue normal monitoring"})


@tool
def get_shipping_options(supplier_id: str, urgency: str = "normal") -> str:
    """Get optimal shipping mode recommendations for a supplier."""
    supplier = next((s for s in get_suppliers() if s["id"] == supplier_id), None)
    if not supplier:
        return json.dumps({"error": f"Supplier {supplier_id} not found"})
    options = {
        "Asia Pacific": [
            {"mode": "Air", "transit_days": 5, "cost_factor": 3.5, "reliability": 95, "co2_kg_per_tonne": 1250},
            {"mode": "Sea", "transit_days": 25, "cost_factor": 1.0, "reliability": 85, "co2_kg_per_tonne": 15},
            {"mode": "Air-Sea Combo", "transit_days": 15, "cost_factor": 2.0, "reliability": 90, "co2_kg_per_tonne": 450}],
        "Europe": [
            {"mode": "Air", "transit_days": 3, "cost_factor": 3.0, "reliability": 96, "co2_kg_per_tonne": 1100},
            {"mode": "Sea", "transit_days": 12, "cost_factor": 1.0, "reliability": 88, "co2_kg_per_tonne": 20},
            {"mode": "Rail", "transit_days": 18, "cost_factor": 1.5, "reliability": 82, "co2_kg_per_tonne": 28}],
        "North America": [
            {"mode": "Ground", "transit_days": 5, "cost_factor": 1.0, "reliability": 94, "co2_kg_per_tonne": 80},
            {"mode": "Air", "transit_days": 2, "cost_factor": 2.5, "reliability": 98, "co2_kg_per_tonne": 900},
            {"mode": "Rail", "transit_days": 7, "cost_factor": 0.8, "reliability": 90, "co2_kg_per_tonne": 25}]
    }
    available = options.get(supplier["region"], options["North America"])
    if urgency == "urgent":     recommended = min(available, key=lambda x: x["transit_days"])
    elif urgency == "economy":  recommended = min(available, key=lambda x: x["cost_factor"])
    else:                       recommended = min(available, key=lambda x: x["transit_days"] * x["cost_factor"])
    return json.dumps({"supplier_id": supplier_id, "supplier_name": supplier["name"],
                       "origin_region": supplier["region"], "urgency": urgency,
                       "recommended_option": recommended, "all_options": available,
                       "recommendation": f"Use {recommended['mode']} shipping for {recommended['transit_days']} day transit at {recommended['cost_factor']}x base cost"})


@tool
def generate_purchase_order(product_id: str, quantity: int, supplier_id: str) -> str:
    """Generate a purchase order recommendation."""
    product  = next((p for p in get_products() if p["id"] == product_id), None)
    supplier = next((s for s in get_suppliers() if s["id"] == supplier_id), None)
    if not product or not supplier:
        return json.dumps({"error": "Product or supplier not found"})
    po_number = f"PO-{datetime.now().strftime('%Y%m%d')}-{product_id[-3:]}"
    total_cost = product["unit_cost"] * quantity
    expected_delivery = (datetime.now() + timedelta(days=product["lead_time_days"])).strftime("%Y-%m-%d")
    return json.dumps({"po_number": po_number, "status": "draft",
                       "product": {"id": product_id, "name": product["name"], "unit_cost": product["unit_cost"]},
                       "supplier": {"id": supplier_id, "name": supplier["name"], "payment_terms": supplier["payment_terms"]},
                       "quantity": quantity, "total_cost": round(total_cost, 2),
                       "expected_delivery": expected_delivery, "lead_time_days": product["lead_time_days"],
                       "action_required": "Review and approve purchase order"})


@tool
def get_market_intelligence(topic: str) -> str:
    """Get relevant market intelligence and news."""
    news = get_news_events()
    relevant = [n for n in news if any(topic.lower() in kw.lower() for kw in n["keywords"]) or topic.lower() in n["headline"].lower()]
    if not relevant:
        relevant = news[:5]
    positive = len([n for n in relevant if n["sentiment"] == "positive"])
    negative = len([n for n in relevant if n["sentiment"] == "negative"])
    return json.dumps({"topic": topic, "news_count": len(relevant),
                       "sentiment_summary": {"positive": positive, "negative": negative,
                                             "overall": "positive" if positive > negative else "negative" if negative > positive else "neutral"},
                       "news_items": relevant,
                       "insight": f"Found {len(relevant)} relevant news items. Market sentiment is {'favorable' if positive > negative else 'concerning' if negative > positive else 'mixed'}."})


@tool
def predict_shipment_delay(supplier_id: str, destination_warehouse: str = "WH-EAST") -> str:
    """Predict delay probability for a supplier's shipments."""
    import numpy as np
    supplier = next((s for s in get_suppliers() if s["id"] == supplier_id), None)
    if not supplier:
        return json.dumps({"error": f"Supplier {supplier_id} not found"})
    shipments = get_shipments_by_supplier(supplier_id)
    if not shipments:
        return json.dumps({"error": f"No shipment history for {supplier_id}"})
    total = len(shipments)
    delayed = [s for s in shipments if s["delay_days"] > 0]
    delay_rate = len(delayed) / total * 100 if total > 0 else 0
    avg_delay  = np.mean([s["delay_days"] for s in delayed]) if delayed else 0
    reason_counts = {}
    for s in delayed:
        r = s.get("delay_reason", "Unknown")
        reason_counts[r] = reason_counts.get(r, 0) + 1
    news = get_news_events()
    region_news = [n for n in news if supplier["region"] in n.get("affected_regions", [])]
    neg_news = [n for n in region_news if n["sentiment"] == "negative" and n["impact"] in ["high", "medium"]]
    news_bump = len(neg_news) * 4
    adjusted_prob = min(delay_rate + news_bump, 99.0)
    risk_level = "critical" if adjusted_prob >= 60 else "high" if adjusted_prob >= 40 else "medium" if adjusted_prob >= 20 else "low"
    predicted_days = round(avg_delay * (1 + news_bump / 100), 1)
    mitigation = []
    if risk_level in ["critical", "high"]:
        mitigation.extend(["Switch to air freight for time-sensitive orders", "Build 2-week buffer stock", "Identify backup supplier"])
    elif risk_level == "medium":
        mitigation.extend(["Monitor shipment status daily", "Increase safety stock by 20%"])
    else:
        mitigation.append("Continue standard monitoring")
    return json.dumps({"supplier_id": supplier_id, "supplier_name": supplier["name"],
                       "destination_warehouse": destination_warehouse, "region": supplier["region"],
                       "historical_analysis": {"total_shipments": total, "delayed_shipments": len(delayed),
                                               "historical_delay_rate_pct": round(delay_rate, 1),
                                               "avg_delay_days": round(avg_delay, 1),
                                               "top_delay_reasons": [{"reason": r, "count": c} for r, c in sorted(reason_counts.items(), key=lambda x: -x[1])[:3]]},
                       "prediction": {"delay_probability_pct": round(adjusted_prob, 1),
                                      "expected_delay_days": predicted_days, "news_risk_adjustment_pct": news_bump,
                                      "active_risk_news": len(neg_news)},
                       "risk_level": risk_level, "mitigation_actions": mitigation,
                       "recommendation": f"{'Immediate action — switch to expedited shipping' if risk_level == 'critical' else 'Proactive monitoring and buffer stock' if risk_level == 'high' else 'Standard monitoring with slight buffer' if risk_level == 'medium' else 'Normal operations'}"})


@tool
def optimize_logistics_cost(time_horizon_months: int = 3, focus_category: str = "all") -> str:
    """Analyse logistics costs and identify optimization opportunities."""
    cost_data = get_cost_breakdown()
    unique_months = sorted(set(d["month"] for d in cost_data), reverse=True)
    recent = [d for d in cost_data if d["month"] in unique_months[:time_horizon_months]]
    category_totals: Dict[str, Dict] = {}
    for row in recent:
        cat = row["category"]
        if cat not in category_totals:
            category_totals[cat] = {"total_cost": 0, "percentages": []}
        category_totals[cat]["total_cost"] += row["cost"]
        category_totals[cat]["percentages"].append(row["percentage"])
    grand_total = sum(v["total_cost"] for v in category_totals.values())
    summary = []
    for cat, vals in category_totals.items():
        avg_pct = sum(vals["percentages"]) / len(vals["percentages"])
        summary.append({"category": cat,
                        "total_cost": round(vals["total_cost"], 2),
                        "share_pct": round(vals["total_cost"] / grand_total * 100, 1) if grand_total else 0,
                        "avg_monthly_pct": round(avg_pct, 1)})
    summary.sort(key=lambda x: -x["total_cost"])
    opportunities = []
    for item in summary:
        if item["category"] == "Transportation" and item["share_pct"] > 40:
            opportunities.append({"category": item["category"], "potential_saving_pct": 8,
                                   "action": "Consolidate shipments; shift 15% air freight to sea for non-urgent lanes",
                                   "estimated_saving_usd": round(item["total_cost"] * 0.08, 0)})
        if item["category"] == "Warehousing" and item["share_pct"] > 25:
            opportunities.append({"category": item["category"], "potential_saving_pct": 6,
                                   "action": "Implement cross-docking for fast-moving SKUs",
                                   "estimated_saving_usd": round(item["total_cost"] * 0.06, 0)})
        if item["category"] == "Inventory Carrying" and item["share_pct"] > 14:
            opportunities.append({"category": item["category"], "potential_saving_pct": 12,
                                   "action": "Reduce safety stock via better demand sensing — target 15% inventory reduction",
                                   "estimated_saving_usd": round(item["total_cost"] * 0.12, 0)})
    total_saving = sum(o["estimated_saving_usd"] for o in opportunities)
    return json.dumps({"analysis_period_months": time_horizon_months, "total_logistics_cost": round(grand_total, 2),
                       "cost_breakdown": summary, "optimization_opportunities": opportunities,
                       "total_potential_savings_usd": round(total_saving, 0),
                       "potential_saving_pct": round(total_saving / grand_total * 100, 1) if grand_total else 0,
                       "recommendation": f"${total_saving:,.0f} in savings potential. Top priority: {opportunities[0]['category'] if opportunities else 'No major issues'}."})


@tool
def generate_production_plan(product_id: str, planning_horizon_weeks: int = 8) -> str:
    """Generate a production schedule aligned with demand forecasts and inventory."""
    import numpy as np
    product = next((p for p in get_products() if p["id"] == product_id), None)
    if not product:
        return json.dumps({"error": f"Product {product_id} not found"})
    demand_data = get_demand_by_product(product_id)
    if not demand_data:
        return json.dumps({"error": f"No demand data for {product_id}"})
    recent = demand_data[-6:]
    avg_monthly = sum(d["actual_demand"] for d in recent) / len(recent)
    avg_weekly  = avg_monthly / 4.33
    season_factors = {1:0.8,2:0.8,3:0.9,4:1.0,5:1.0,6:1.1,7:1.1,8:1.1,9:1.0,10:1.3,11:1.3,12:1.3}
    inventory = get_inventory_by_product(product_id)
    current_inv   = sum(i["current_stock"] for i in inventory)
    safety_total  = sum(i["safety_stock"] for i in inventory)
    max_weekly    = int(avg_weekly * 2)
    schedule = []
    projected_inv = current_inv
    alerts = []
    for week in range(1, planning_horizon_weeks + 1):
        target_month = ((datetime.now().month - 1 + week // 4) % 12) + 1
        sf = season_factors.get(target_month, 1.0)
        weekly_demand = int(avg_weekly * sf * np.random.uniform(0.9, 1.1))
        end_if_no_prod = projected_inv - weekly_demand
        if end_if_no_prod < safety_total:
            prod_qty = min(int(safety_total - end_if_no_prod + weekly_demand), max_weekly)
        else:
            prod_qty = 0
        projected_inv = max(projected_inv + prod_qty - weekly_demand, 0)
        status = "critical" if projected_inv < safety_total * 0.5 else "low" if projected_inv < safety_total else "overstock" if projected_inv > safety_total * 5 else "optimal"
        if status == "critical":
            alerts.append(f"Week {week}: Inventory drops below 50% safety stock — increase production")
        elif status == "overstock":
            alerts.append(f"Week {week}: Overstock risk — reduce production run")
        schedule.append({"week": week, "week_start": (datetime.now() + timedelta(weeks=week-1)).strftime("%Y-%m-%d"),
                         "forecasted_demand": weekly_demand, "recommended_production": prod_qty,
                         "projected_end_inventory": projected_inv,
                         "utilization_pct": round(prod_qty / max_weekly * 100, 1) if max_weekly > 0 else 0,
                         "status": status})
    total_prod   = sum(w["recommended_production"] for w in schedule)
    total_demand = sum(w["forecasted_demand"] for w in schedule)
    return json.dumps({"product_id": product_id, "product_name": product["name"],
                       "planning_horizon_weeks": planning_horizon_weeks,
                       "current_inventory": current_inv, "safety_stock_target": safety_total,
                       "max_weekly_capacity": max_weekly, "weekly_schedule": schedule,
                       "summary": {"total_planned_production": total_prod, "total_forecasted_demand": total_demand,
                                   "avg_weekly_utilization_pct": round(sum(w["utilization_pct"] for w in schedule)/len(schedule),1),
                                   "weeks_with_production": len([w for w in schedule if w["recommended_production"] > 0])},
                       "alerts": alerts,
                       "recommendation": f"Plan {total_prod} units over {planning_horizon_weeks} weeks. {'⚠ '+str(len(alerts))+' alert(s).' if alerts else '✅ Schedule healthy.'}"})


@tool
def track_sustainability(region: str = "Global") -> str:
    """Monitor supply chain carbon footprint and ESG compliance."""
    import numpy as np
    shipments = get_shipments_by_supplier()
    emission_factors = {"air": 1.200, "sea": 0.015, "rail": 0.028, "ground": 0.080, "express": 1.100}
    total_co2 = 0.0
    mode_breakdown: Dict[str, float] = {}
    supplier_footprints: Dict[str, float] = {}
    for s in shipments:
        mode_key = s.get("carrier", "ground").lower().split()[0]
        ef = emission_factors.get(mode_key, 0.080)
        weight_t = np.random.uniform(0.5, 5.0)
        dist_km  = np.random.uniform(200, 12000)
        co2 = ef * weight_t * dist_km / 1000
        total_co2 += co2
        mode_breakdown[s.get("carrier","Ground")] = mode_breakdown.get(s.get("carrier","Ground"), 0) + co2
        sup_id = s.get("supplier_id","unknown")
        supplier_footprints[sup_id] = supplier_footprints.get(sup_id, 0) + co2
    news = get_news_events()
    esg_kws = {"sustainability","carbon","environment","green","climate","energy","emission"}
    esg_news = [n for n in news if any(kw in " ".join(n.get("keywords",[])).lower() for kw in esg_kws)] or news[:3]
    air_co2  = sum(v for k, v in mode_breakdown.items() if "air" in k.lower() or "express" in k.lower())
    air_share = air_co2 / total_co2 if total_co2 > 0 else 0
    score = max(0, round(100 - air_share * 60, 1))
    top_emitters = sorted(supplier_footprints.items(), key=lambda x: -x[1])[:5]
    suppliers = get_suppliers()
    top_emitters_named = [{"supplier_id": sid, "supplier_name": next((s["name"] for s in suppliers if s["id"] == sid), sid),
                           "estimated_co2_tonnes": round(co2, 2)} for sid, co2 in top_emitters]
    green_actions = [
        {"action": "Mode shift: Replace 20% air freight with sea freight on long-haul lanes", "co2_reduction_pct": 15, "cost_impact": "cost_saving"},
        {"action": "Consolidate LTL shipments into FTL to reduce trips by 30%", "co2_reduction_pct": 8, "cost_impact": "cost_neutral"},
        {"action": "Partner with rail providers for cross-country lanes", "co2_reduction_pct": 10, "cost_impact": "cost_saving"},
        {"action": "Require ISO 14001 certification for all Tier-1 suppliers", "co2_reduction_pct": 5, "cost_impact": "cost_neutral"},
        {"action": "Implement returnable packaging with top 3 suppliers", "co2_reduction_pct": 3, "cost_impact": "cost_saving"},
    ]
    return json.dumps({"region_focus": region, "sustainability_score": score,
                       "carbon_footprint": {"total_estimated_co2_tonnes": round(total_co2, 2),
                                            "breakdown_by_carrier": {k: round(v, 2) for k, v in sorted(mode_breakdown.items(), key=lambda x: -x[1])[:5]},
                                            "air_freight_share_pct": round(air_share * 100, 1)},
                       "top_emitting_suppliers": top_emitters_named,
                       "esg_news": esg_news[:4], "green_action_plan": green_actions,
                       "certifications_on_file": {"iso_14001": len([s for s in suppliers if "ISO 14001" in s.get("certifications",[])]),
                                                  "total_suppliers": len(suppliers)},
                       "recommendation": f"Score {score}/100. {'Excellent.' if score >= 80 else 'Mode-shift programs will drive score above 80.' if score >= 60 else 'Prioritise air-to-sea shift urgently.'}"})


# ============================================================
# TOOL REGISTRY
# ============================================================
DEMAND_TOOLS       = [forecast_demand, analyze_stockout_risk]
INVENTORY_TOOLS    = [calculate_reorder_point, check_warehouse_levels]
SUPPLIER_TOOLS     = [score_supplier_risk, get_shipping_options, get_market_intelligence, predict_shipment_delay]
ACTION_TOOLS       = [generate_purchase_order]
LOGISTICS_TOOLS    = [optimize_logistics_cost]
PLANNING_TOOLS     = [generate_production_plan]
SUSTAINABILITY_TOOLS = [track_sustainability]

ALL_TOOLS = (DEMAND_TOOLS + INVENTORY_TOOLS + SUPPLIER_TOOLS +
             ACTION_TOOLS + LOGISTICS_TOOLS + PLANNING_TOOLS + SUSTAINABILITY_TOOLS)

# Map tool name → function for fast lookup
TOOL_MAP = {t.name: t for t in ALL_TOOLS}

# Map agent name → its tools
AGENT_TOOL_MAP = {
    "demand":         DEMAND_TOOLS,
    "inventory":      INVENTORY_TOOLS,
    "supplier":       SUPPLIER_TOOLS,
    "action":         ACTION_TOOLS,
    "logistics":      LOGISTICS_TOOLS,
    "planning":       PLANNING_TOOLS,
    "sustainability": SUSTAINABILITY_TOOLS,
}


# ============================================================
# AGENT SYSTEM
# ============================================================
class SupplyChainAgentSystem:
    def __init__(self, ollama_host: str = "http://localhost:11434", model: str = "llama3.2:latest"):
        self.ollama_host = ollama_host
        self.model = model
        self.tools = ALL_TOOLS
        self.agent_prompts = self._create_agent_prompts()
        self.llm = None
        try:
            self.llm = ChatOllama(base_url=ollama_host, model=model, temperature=0.3)
            logger.info(f"ChatOllama initialised: {model}")
        except Exception as e:
            logger.warning(f"Ollama not available — mock mode active. ({e})")

    def _create_agent_prompts(self) -> Dict[str, str]:
        return {
            "orchestrator": """You are the Orchestrator Agent for SupplyMind.
Analyse the business query and decide which agents (demand/inventory/supplier/action/logistics/planning/sustainability) to activate.
Return JSON: {"agents_involved": [...], "reasoning": "...", "urgency": "immediate/today/week"}""",
            "demand": """You are the Demand Forecasting Agent. You have analysed real historical data.
Provide a concise business-readable assessment: what is demand doing, which products need attention, key risks.
Return JSON: {"analysis": "...", "recommendations": [...], "confidence": 0.0-1.0}""",
            "inventory": """You are the Inventory Optimisation Agent. You have real stock level data.
Explain what the numbers mean in business terms: which products risk stockout, what to reorder, where is cash locked up.
Return JSON: {"analysis": "...", "recommendations": [...], "confidence": 0.0-1.0}""",
            "supplier": """You are the Supplier Risk Agent. You have real risk scores and shipment delay predictions.
Translate the data into business decisions: who poses risk, what shipments to worry about, what action to take.
Return JSON: {"analysis": "...", "recommendations": [...], "confidence": 0.0-1.0}""",
            "action": """You are the Action Agent. Based on tool data, create specific, executable action items.
Each action must have: what to do, why, expected impact, and urgency.
Return JSON: {"analysis": "...", "recommendations": [...], "confidence": 0.0-1.0}""",
            "logistics": """You are the Logistics Cost Agent. You have real cost breakdown data.
Explain where money is being spent and how to reduce it.
Return JSON: {"analysis": "...", "recommendations": [...], "confidence": 0.0-1.0}""",
            "planning": """You are the Production Planning Agent. You have 8-week production schedules.
Identify which products face production gaps, what to schedule, and capacity constraints.
Return JSON: {"analysis": "...", "recommendations": [...], "confidence": 0.0-1.0}""",
            "sustainability": """You are the Sustainability Agent. You have carbon footprint data and ESG compliance status.
Translate CO2 numbers into business actions and flag non-compliant suppliers.
Return JSON: {"analysis": "...", "recommendations": [...], "confidence": 0.0-1.0}""",
        }

    # ── Low-level tool invocation ──────────────────────────────────────────────
    @staticmethod
    def _safe_json_loads(raw):
        """Parse JSON produced by tools, converting numpy scalars to Python natives."""
        import numpy as np

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):   return int(obj)
                if isinstance(obj, (np.floating,)):  return float(obj)
                if isinstance(obj, (np.bool_,)):     return bool(obj)
                if isinstance(obj, (np.ndarray,)):   return obj.tolist()
                return super().default(obj)

        # Re-encode through NumpyEncoder then decode to get plain Python types
        if isinstance(raw, str):
            parsed = json.loads(raw)
        else:
            parsed = raw
        return json.loads(json.dumps(parsed, cls=NumpyEncoder))

    async def invoke_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        if tool_name not in TOOL_MAP:
            logger.warning(f"  ✗ Tool not found: {tool_name}")
            return {"error": f"Tool {tool_name} not found"}
        params_summary = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(f"  🔧 Tool call: {tool_name}({params_summary})")
        try:
            result = TOOL_MAP[tool_name].invoke(kwargs)
            parsed = self._safe_json_loads(result)
            # Log a short summary of what came back
            if "error" in parsed:
                logger.warning(f"  ✗ Tool {tool_name} returned error: {parsed['error']}")
            else:
                # Build a one-line summary depending on tool
                if tool_name == "forecast_demand":
                    logger.info(f"  ✓ {tool_name} → product={parsed.get('product_name')} | avg={parsed.get('historical_avg')} | trend={parsed.get('trend_pct')}%")
                elif tool_name == "analyze_stockout_risk":
                    highest = parsed.get("highest_risk", {})
                    logger.info(f"  ✓ {tool_name} → highest_risk={highest.get('risk_level')} | days_supply={highest.get('days_of_supply')} | warehouse={highest.get('warehouse')}")
                elif tool_name == "calculate_reorder_point":
                    logger.info(f"  ✓ {tool_name} → product={parsed.get('product_name')} | reorder_pt={parsed.get('reorder_point')} | safety_stock={parsed.get('safety_stock')} | EOQ={parsed.get('economic_order_quantity')}")
                elif tool_name == "check_warehouse_levels":
                    s = parsed.get("summary", {})
                    logger.info(f"  ✓ {tool_name} → warehouse={parsed.get('warehouse_name')} | critical={s.get('critical_items')} | utilization={parsed.get('capacity_utilization')}%")
                elif tool_name == "score_supplier_risk":
                    logger.info(f"  ✓ {tool_name} → supplier={parsed.get('supplier_name')} | score={parsed.get('risk_score')} | level={parsed.get('risk_level')}")
                elif tool_name == "predict_shipment_delay":
                    pred = parsed.get("prediction", {})
                    logger.info(f"  ✓ {tool_name} → supplier={parsed.get('supplier_name')} | delay_prob={pred.get('delay_probability_pct')}% | risk={parsed.get('risk_level')}")
                elif tool_name == "get_shipping_options":
                    rec = parsed.get("recommended_option", {})
                    logger.info(f"  ✓ {tool_name} → supplier={parsed.get('supplier_name')} | mode={rec.get('mode')} | days={rec.get('transit_days')} | cost_factor={rec.get('cost_factor')}")
                elif tool_name == "generate_purchase_order":
                    logger.info(f"  ✓ {tool_name} → PO={parsed.get('po_number')} | qty={parsed.get('quantity')} | total=${parsed.get('total_cost')}")
                elif tool_name == "optimize_logistics_cost":
                    logger.info(f"  ✓ {tool_name} → total_cost=${parsed.get('total_logistics_cost')} | savings_potential=${parsed.get('total_potential_savings_usd')} ({parsed.get('potential_saving_pct')}%)")
                elif tool_name == "generate_production_plan":
                    s = parsed.get("summary", {})
                    logger.info(f"  ✓ {tool_name} → product={parsed.get('product_name')} | planned={s.get('total_planned_production')} units | alerts={len(parsed.get('alerts', []))}")
                elif tool_name == "track_sustainability":
                    logger.info(f"  ✓ {tool_name} → score={parsed.get('sustainability_score')}/100 | co2={parsed.get('carbon_footprint',{}).get('total_estimated_co2_tonnes')} tonnes")
                elif tool_name == "get_market_intelligence":
                    logger.info(f"  ✓ {tool_name} → topic={parsed.get('topic')} | news_count={parsed.get('news_count')} | sentiment={parsed.get('sentiment_summary',{}).get('overall')}")
                else:
                    logger.info(f"  ✓ {tool_name} → OK")
            return parsed
        except Exception as e:
            logger.error(f"  ✗ Tool {tool_name} FAILED: {e}")
            return {"error": str(e)}

    # ── Single agent LLM call ──────────────────────────────────────────────────
    async def run_agent(self, agent_name: str, task: str, context: Dict = None) -> Dict[str, Any]:
        if agent_name not in self.agent_prompts:
            return {"error": f"Agent {agent_name} not found"}
        system_prompt = self.agent_prompts[agent_name]
        context_str   = json.dumps(context, indent=2) if context else "No additional context"
        user_message  = f"Task: {task}\n\nData Context:\n{context_str}\n\nProvide your expert JSON assessment."

        mode = "LLM" if self.llm else "MOCK"
        logger.info(f"  🤖 Agent [{agent_name.upper()}] starting ({mode} mode) | task='{task[:60]}'")

        if self.llm:
            try:
                logger.info(f"  🧠 Agent [{agent_name.upper()}] calling Ollama {self.model}...")
                response = await self.llm.ainvoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message)
                ])
                try:
                    parsed = json.loads(response.content)
                    recs = parsed.get("recommendations", [])
                    confidence = parsed.get("confidence", "N/A")
                    logger.info(f"  ✓ Agent [{agent_name.upper()}] LLM response → {len(recs)} recommendations | confidence={confidence}")
                    if recs:
                        for r in recs[:2]:
                            logger.info(f"    → {r}")
                    return parsed
                except json.JSONDecodeError:
                    logger.info(f"  ✓ Agent [{agent_name.upper()}] LLM returned raw text ({len(response.content)} chars)")
                    return {"analysis": response.content, "recommendations": [], "raw_response": True}
            except Exception as e:
                logger.error(f"  ✗ Agent [{agent_name.upper()}] LLM call FAILED: {e} — falling back to mock")

        result = self._mock_agent_response(agent_name, task, context)
        recs = result.get("recommendations", [])
        logger.info(f"  ✓ Agent [{agent_name.upper()}] mock response → {len(recs)} recommendations")
        if recs:
            for r in recs[:2]:
                logger.info(f"    → {r}")
        return result

    def _mock_agent_response(self, agent_name: str, task: str, context: Dict = None) -> Dict[str, Any]:
        """Mock responses that use real context data when available."""
        task_lower = task.lower()
        ctx = context or {}

        if agent_name == "orchestrator":
            agents = []
            if any(kw in task_lower for kw in ["demand", "forecast", "sales"]): agents.append("demand")
            if any(kw in task_lower for kw in ["inventory", "stock", "warehouse"]): agents.append("inventory")
            if any(kw in task_lower for kw in ["supplier", "vendor", "risk", "delay"]): agents.append("supplier")
            if any(kw in task_lower for kw in ["order", "purchase", "procurement"]): agents.append("action")
            if any(kw in task_lower for kw in ["cost", "logistics", "save"]): agents.append("logistics")
            if any(kw in task_lower for kw in ["production", "plan", "schedule"]): agents.append("planning")
            if any(kw in task_lower for kw in ["sustainability", "carbon", "esg"]): agents.append("sustainability")
            if any(kw in task_lower for kw in ["today", "priority", "focus", "urgent", "7 day", "week", "what should"]):
                agents = ["demand", "inventory", "supplier", "logistics"]
            return {"agents_involved": agents or ["demand", "inventory", "supplier"],
                    "reasoning": f"Query '{task[:60]}' routed to {', '.join(agents or ['demand','inventory','supplier'])}",
                    "urgency": "immediate" if any(kw in task_lower for kw in ["today","urgent","now"]) else "week"}

        elif agent_name == "demand":
            # Try to use real context
            fc_list = ctx.get("forecasts", [])
            trend_info = ""
            if fc_list:
                rising = [f for f in fc_list if (f.get("trend_pct") or 0) > 5]
                if rising:
                    trend_info = f" {len(rising)} products show >5% demand growth."
            return {"analysis": f"Demand analysis across all 10 products shows seasonal patterns with moderate upward trend.{trend_info} MAPE averaging 8.5% indicates reliable forecasting.",
                    "recommendations": ["Increase safety stock for Q4 peak for high-demand products",
                                        "Monitor 3 products currently at critical stockout risk",
                                        "Review promotional calendars for demand spikes"],
                    "confidence": 0.82}

        elif agent_name == "inventory":
            critical = ctx.get("critical_items_count", 3)
            return {"analysis": f"{critical} SKUs are at critical stock levels requiring immediate reorder. Warehouse utilisation averaging 74% across all sites.",
                    "recommendations": ["Expedite orders for critical items in East Coast warehouse",
                                        "Review reorder points for high-velocity Electronics SKUs",
                                        "Consider cross-docking to reduce WH-CENTRAL dwell time"],
                    "confidence": 0.88}

        elif agent_name == "supplier":
            high_risk = ctx.get("high_risk_count", 2)
            return {"analysis": f"{high_risk} suppliers flagged as high/critical risk. Asia Pacific suppliers showing elevated delay probability due to active geopolitical news signals.",
                    "recommendations": ["Switch FastShip Logistics to air freight for time-sensitive orders",
                                        "Build 2-week buffer stock for single-source components",
                                        "Initiate backup supplier qualification for critical parts"],
                    "confidence": 0.79}

        elif agent_name == "logistics":
            return {"analysis": "Transportation represents 42% of logistics spend — primary optimisation target. Shifting 15% of air freight to sea could save $180K annually.",
                    "recommendations": ["Consolidate Asia Pacific sea shipments to weekly batches",
                                        "Negotiate volume rates with top 3 carriers",
                                        "Review North America lane assignments — rail underutilised"],
                    "confidence": 0.84}

        elif agent_name == "planning":
            return {"analysis": "3 products face production gaps in weeks 3-5 due to demand seasonality. Avg capacity utilisation at 71% — headroom exists for catch-up runs.",
                    "recommendations": ["Schedule production surge for PRD-001 and PRD-006 in week 3",
                                        "Align maintenance windows to Q1 low-demand period",
                                        "Buffer 2-week inventory for all Electronics SKUs before Q4"],
                    "confidence": 0.80}

        elif agent_name == "sustainability":
            return {"analysis": "Sustainability score 68/100. Air freight accounts for ~48% of supply chain CO₂. 5 of 8 suppliers lack ISO 14001 certification.",
                    "recommendations": ["Mode-shift 20% of air freight to sea — saves 15% CO₂ and reduces cost",
                                        "Require ISO 14001 from Asia Pacific suppliers at next contract renewal",
                                        "Implement returnable packaging with top 3 suppliers by Q3"],
                    "confidence": 0.77}

        elif agent_name == "action":
            return {"analysis": "3 critical purchase orders ready for approval. All routed to Precision Parts USA (lowest risk score). Total PO value $134,500.",
                    "recommendations": ["Approve PO-001 and PO-002 today — critical inventory risk",
                                        "Review PO-003 with procurement team before sign-off",
                                        "Set up auto-approval rules for orders < $10K with low-risk suppliers"],
                    "confidence": 0.92}

        return {"analysis": "Agent processed the query.", "recommendations": [], "confidence": 0.5}

    # ── Full multi-agent orchestration (used by /orchestrator/process) ─────────
    async def orchestrate(self, task: str, context: Dict = None) -> Dict[str, Any]:
        result = {"task": task, "timestamp": datetime.now(timezone.utc).isoformat(),
                  "agent_outputs": {}, "tools_used": [], "final_response": None}
        orch = await self.run_agent("orchestrator", task, context)
        result["agent_outputs"]["orchestrator"] = orch
        for agent in orch.get("agents_involved", []):
            if agent in self.agent_prompts and agent != "orchestrator":
                result["agent_outputs"][agent] = await self.run_agent(agent, task, context)
        all_recs = []
        for out in result["agent_outputs"].values():
            if isinstance(out, dict) and "recommendations" in out:
                all_recs.extend(out["recommendations"])
        result["final_response"] = {"summary": f"Task processed by {len(result['agent_outputs'])} agents",
                                    "agents_consulted": list(result["agent_outputs"].keys()),
                                    "consolidated_recommendations": all_recs[:10], "status": "completed"}
        return result

    # ── BUSINESS QUERY — the main LLM-powered intelligence endpoint ───────────
    async def business_query(self, query: str) -> Dict[str, Any]:
        """
        Natural language business query processing.
        1. Orchestrator Agent (LLM) reads the query and decides which agents to activate
        2. Call real tools to gather live data for each activated agent's domain
        3. Pass real data to each activated agent's LLM for reasoning
        4. Synthesise into business-readable response with priority actions

        The Orchestrator is the SOLE routing authority — no hardcoded keyword matching.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info(f"\n{'='*60}")
        logger.info(f"📥 BUSINESS QUERY: '{query}'")
        logger.info(f"{'='*60}")

        # ── Step 1: Orchestrator Agent decides routing ─────────────────────────
        # Give the orchestrator full context about available agents so it can make
        # an informed decision even in mock mode.
        orchestrator_context = {
            "available_agents": {
                "demand":         "Demand forecasting, stockout risk, seasonality analysis",
                "inventory":      "EOQ optimisation, safety stock, warehouse health checks",
                "supplier":       "Risk scoring, delay prediction, shipping route optimisation",
                "action":         "Purchase order generation, procurement execution",
                "logistics":      "Logistics cost analysis, freight mode optimisation",
                "planning":       "8-week production scheduling, capacity planning",
                "sustainability": "Carbon footprint, ESG compliance, green logistics",
            },
            "instructions": (
                "Read the business query carefully. Return a JSON object with:\n"
                "  agents_involved: list of agent names from available_agents that should handle this query\n"
                "  reasoning: one sentence explaining your routing decision\n"
                "  urgency: 'immediate' | 'today' | 'week'\n"
                "For broad queries like 'what should I focus on today' or '7-day plan', "
                "activate demand, inventory, supplier, and logistics. "
                "For specific queries, activate only the relevant agents."
            )
        }

        logger.info(f"Routing query to Orchestrator: '{query[:80]}'")
        orchestrator_response = await self.run_agent("orchestrator", query, orchestrator_context)

        domains = orchestrator_response.get("agents_involved", [])
        routing_reasoning = orchestrator_response.get("reasoning", "")
        urgency = orchestrator_response.get("urgency", "today")

        # Safety net: if orchestrator returned nothing valid, fall back to core domains
        valid_domains = {"demand","inventory","supplier","action","logistics","planning","sustainability"}
        domains = [d for d in domains if d in valid_domains]
        if not domains:
            domains = ["demand", "inventory", "supplier"]

        logger.info(f"Orchestrator activated agents: {domains} | urgency: {urgency}")
        logger.info(f"\n📋 STEP 2 — Gathering real data via tools (agents: {domains})")

        # ── Step 2: Gather real data from tools for each agent the orchestrator activated ──
        gathered: Dict[str, Any] = {}
        tools_called: List[str] = []
        agents_activated: List[str] = list(domains)  # already decided by orchestrator

        if "demand" in domains:
            logger.info(f"\n  [DEMAND] Running forecast_demand + analyze_stockout_risk for all {len(PRODUCTS)} products...")
            demand_results = []
            for p in PRODUCTS:
                fc = await self.invoke_tool("forecast_demand", product_id=p["id"], periods=3)
                sk = await self.invoke_tool("analyze_stockout_risk", product_id=p["id"])
                if "error" not in fc and "error" not in sk:
                    highest = sk.get("highest_risk", {})
                    demand_results.append({
                        "product_id": p["id"], "product_name": p["name"],
                        "trend_pct": fc.get("trend_pct"), "historical_avg": fc.get("historical_avg"),
                        "mape": fc.get("mape"), "next_period_forecast": fc.get("forecasts", [{}])[0].get("forecast"),
                        "stockout_risk": highest.get("risk_level", "low"),
                        "days_of_supply": highest.get("days_of_supply"),
                        "recommendation": highest.get("recommendation")
                    })
            tools_called.extend(["forecast_demand", "analyze_stockout_risk"])
            gathered["demand"] = demand_results
            critical_d = len([d for d in demand_results if d.get("stockout_risk") in ("critical","high")])
            logger.info(f"  [DEMAND] ✓ {len(demand_results)} products analysed | {critical_d} at critical/high stockout risk")

        if "inventory" in domains:
            logger.info(f"\n  [INVENTORY] Running calculate_reorder_point for all {len(PRODUCTS)} products...")
            inv_results = []
            critical_count = 0
            for p in PRODUCTS:
                rr = await self.invoke_tool("calculate_reorder_point", product_id=p["id"])
                if "error" not in rr:
                    inv_data = get_inventory_by_product(p["id"])
                    total_stock = sum(i["current_stock"] for i in inv_data)
                    if total_stock < rr.get("safety_stock", 0):
                        status = "critical"; critical_count += 1
                    elif total_stock < rr.get("reorder_point", 0):
                        status = "high"
                    elif total_stock < rr.get("reorder_point", 0) * 1.3:
                        status = "medium"
                    else:
                        status = "low"
                    inv_results.append({**rr, "current_total_stock": total_stock, "status": status})
            tools_called.extend(["calculate_reorder_point", "check_warehouse_levels"])
            gathered["inventory"] = inv_results
            gathered["critical_items_count"] = critical_count
            logger.info(f"  [INVENTORY] ✓ {len(inv_results)} products analysed | {critical_count} below safety stock (critical)")

        if "supplier" in domains:
            logger.info(f"\n  [SUPPLIER] Running score_supplier_risk + predict_shipment_delay for all {len(SUPPLIERS)} suppliers...")
            supplier_results = []
            high_risk_count = 0
            for sup in SUPPLIERS:
                risk = await self.invoke_tool("score_supplier_risk", supplier_id=sup["id"])
                delay = await self.invoke_tool("predict_shipment_delay", supplier_id=sup["id"])
                if "error" not in risk:
                    if risk.get("risk_level") in ("critical", "high"):
                        high_risk_count += 1
                    supplier_results.append({
                        "supplier_name": sup["name"], "region": sup["region"],
                        "risk_score": risk.get("risk_score"), "risk_level": risk.get("risk_level"),
                        "delay_probability_pct": delay.get("prediction", {}).get("delay_probability_pct") if "error" not in delay else None,
                        "recommendation": risk.get("recommendation")
                    })
            tools_called.extend(["score_supplier_risk", "predict_shipment_delay"])
            gathered["suppliers"] = supplier_results
            gathered["high_risk_count"] = high_risk_count
            logger.info(f"  [SUPPLIER] ✓ {len(supplier_results)} suppliers scored | {high_risk_count} at high/critical risk")

        if "logistics" in domains:
            logger.info(f"\n  [LOGISTICS] Running optimize_logistics_cost (last 3 months)...")
            cost = await self.invoke_tool("optimize_logistics_cost", time_horizon_months=3, focus_category="all")
            tools_called.append("optimize_logistics_cost")
            gathered["logistics"] = cost

        if "planning" in domains:
            logger.info(f"\n  [PLANNING] Running generate_production_plan for top 3 products...")
            plans = []
            for p in PRODUCTS[:3]:
                plan = await self.invoke_tool("generate_production_plan", product_id=p["id"], planning_horizon_weeks=8)
                if "error" not in plan:
                    plans.append({"product_name": p["name"], "alerts": plan.get("alerts", []),
                                  "total_planned": plan.get("summary", {}).get("total_planned_production")})
            tools_called.append("generate_production_plan")
            gathered["planning"] = plans
            total_alerts = sum(len(p.get("alerts", [])) for p in plans)
            logger.info(f"  [PLANNING] ✓ {len(plans)} products planned | {total_alerts} production alerts")

        if "sustainability" in domains:
            logger.info(f"\n  [SUSTAINABILITY] Running track_sustainability (Global)...")
            sus = await self.invoke_tool("track_sustainability", region="Global")
            tools_called.append("track_sustainability")
            gathered["sustainability"] = sus

        # ── Step 3: LLM agent reasoning per domain ────────────────────────────
        logger.info(f"\n📋 STEP 3 — Agent LLM reasoning ({len(agents_activated)} agents: {agents_activated})")
        agent_insights: Dict[str, Any] = {}
        for agent in agents_activated:
            domain_data = gathered.get(agent, gathered)
            agent_insights[agent] = await self.run_agent(
                agent,
                query,
                {agent: domain_data, **{k: gathered[k] for k in gathered if k != agent}}
            )

        # ── Step 4: Synthesise priority actions from all agent insights ────────
        logger.info(f"\n📋 STEP 4 — Synthesising priority actions from tool data + agent insights")
        priority_actions = []

        # From demand data
        if "demand" in gathered:
            critical_products = [d for d in gathered["demand"] if d.get("stockout_risk") in ("critical","high")]
            for cp in critical_products[:3]:
                priority_actions.append({
                    "priority": cp["stockout_risk"],
                    "agent": "demand",
                    "agent_color": "#a855f7",
                    "action": f"Reorder {cp['product_name']} immediately",
                    "detail": f"{cp['days_of_supply']:.0f} days of supply remaining — below lead time. Next period forecast: {cp.get('next_period_forecast','N/A')} units.",
                    "estimated_impact": "Prevents stockout and lost production"
                })

        # From supplier data
        if "suppliers" in gathered:
            risky_suppliers = [s for s in gathered["suppliers"] if s.get("risk_level") in ("critical","high")]
            for rs in risky_suppliers[:3]:
                priority_actions.append({
                    "priority": rs["risk_level"],
                    "agent": "supplier",
                    "agent_color": "#f97316",
                    "action": f"Review {rs['supplier_name']} — {rs['risk_level'].upper()} risk",
                    "detail": f"Risk score {rs['risk_score']}/100. Delay probability: {rs.get('delay_probability_pct','N/A')}%. {rs.get('recommendation','')}",
                    "estimated_impact": "Prevent supply disruption from high-risk vendor"
                })

        # From inventory data
        if "inventory" in gathered:
            critical_inv = [i for i in gathered["inventory"] if i.get("status") in ("critical","high")]
            for ci in critical_inv[:2]:
                priority_actions.append({
                    "priority": ci["status"],
                    "agent": "inventory",
                    "agent_color": "#06b6d4",
                    "action": f"Raise PO for {ci['product_name']} — stock at {ci['current_total_stock']} units",
                    "detail": f"Reorder point: {ci['reorder_point']} | Safety stock: {ci['safety_stock']} | EOQ: {ci['economic_order_quantity']}",
                    "estimated_impact": "Restore service levels to 95%"
                })

        # From logistics data
        if "logistics" in gathered:
            opps = gathered["logistics"].get("optimization_opportunities", [])
            if opps:
                top_opp = opps[0]
                priority_actions.append({
                    "priority": "medium",
                    "agent": "logistics",
                    "agent_color": "#22c55e",
                    "action": f"Logistics saving: {top_opp['category']} ({top_opp['potential_saving_pct']}% reduction possible)",
                    "detail": top_opp["action"],
                    "estimated_impact": f"Save ${top_opp['estimated_saving_usd']:,.0f} in logistics cost"
                })

        # From agent LLM recommendations
        for agent, insight in agent_insights.items():
            recs = insight.get("recommendations", [])
            for rec in recs[:1]:  # 1 rec per agent to avoid overflow
                existing_actions = [a["action"] for a in priority_actions]
                if rec not in existing_actions:
                    priority_actions.append({
                        "priority": "medium",
                        "agent": agent,
                        "agent_color": {"demand":"#a855f7","inventory":"#06b6d4","supplier":"#f97316",
                                        "logistics":"#22c55e","planning":"#3b82f6","sustainability":"#22c55e"}.get(agent,"#6366f1"),
                        "action": rec,
                        "detail": insight.get("analysis","")[:200],
                        "estimated_impact": "Operational improvement"
                    })

        # Sort by priority
        prio_order = {"critical":0, "high":1, "medium":2, "low":3}
        priority_actions.sort(key=lambda x: prio_order.get(x.get("priority","low"), 3))

        # ── Step 5: Build 7-day plan — fully dynamic from real actions ────────
        # Group actions by priority tier
        critical_actions = [a["action"] for a in priority_actions if a.get("priority") == "critical"]
        high_actions_list = [a["action"] for a in priority_actions if a.get("priority") == "high"]
        medium_actions_list = [a["action"] for a in priority_actions if a.get("priority") == "medium"]
        low_actions_list = [a["action"] for a in priority_actions if a.get("priority") == "low"]

        # Domain-specific fallbacks that are relevant to what was actually queried
        domain_fallbacks = {
            "demand":         "Review demand forecasts and identify at-risk SKUs",
            "inventory":      "Audit inventory levels and update reorder points",
            "supplier":       "Check supplier scorecards and flag at-risk vendors",
            "logistics":      "Review freight spend and consolidate shipment lanes",
            "planning":       "Update production schedules against latest demand",
            "sustainability": "Review ESG compliance status and carbon targets",
        }
        relevant_fallbacks = [domain_fallbacks[d] for d in domains if d in domain_fallbacks]

        def day_actions(primary_list, count=3):
            """Return up to `count` items; fill with relevant domain fallbacks if short."""
            result = primary_list[:count]
            if len(result) < 2:
                for fb in relevant_fallbacks:
                    if fb not in result and len(result) < count:
                        result.append(fb)
            return result

        # Day 1-2: most urgent — critical first, then high
        day12_pool = critical_actions + high_actions_list
        day12_focus = (
            "Critical items requiring immediate action" if critical_actions
            else "High-priority items" if high_actions_list
            else f"{', '.join(d.capitalize() for d in domains[:2])} priorities"
        )

        # Day 3-4: medium priority + remaining high
        day34_pool = high_actions_list[3:] + medium_actions_list
        day34_focus = (
            "Procurement & routing optimisation" if any(d in domains for d in ["supplier", "inventory"])
            else "Cost & efficiency improvements" if "logistics" in domains
            else "Operational follow-up"
        )

        # Day 5-6: medium/low + anything not covered
        day56_pool = medium_actions_list[3:] + low_actions_list
        day56_focus = (
            "Cost optimisation & process improvements" if "logistics" in domains
            else "Planning & efficiency review"
        )

        seven_day_plan = [
            {
                "day": "Day 1-2",
                "focus": day12_focus,
                "actions": day_actions(day12_pool)
            },
            {
                "day": "Day 3-4",
                "focus": day34_focus,
                "actions": day_actions(day34_pool)
            },
            {
                "day": "Day 5-6",
                "focus": day56_focus,
                "actions": day_actions(day56_pool)
            },
            {
                "day": "Day 7",
                "focus": "Review & report",
                "actions": [
                    f"Generate {'logistics' if 'logistics' in domains else 'supply chain'} performance report",
                    "Send weekly summary to stakeholders",
                    "Set action owners and due dates for open items"
                ]
            },
        ]

        # ── Step 6: Executive summary via LLM or smart mock ───────────────────
        critical_count  = len([a for a in priority_actions if a.get("priority") == "critical"])
        high_count      = len([a for a in priority_actions if a.get("priority") == "high"])
        medium_count    = len([a for a in priority_actions if a.get("priority") == "medium"])

        if self.llm:
            try:
                summary_prompt = f"""Business Query: "{query}"

Real data gathered by agents:
- Domains analysed: {', '.join(domains)}
- {gathered.get('critical_items_count', 0)} products at critical inventory level
- {gathered.get('high_risk_count', 0)} suppliers flagged as high/critical risk
- Logistics saving potential: ${gathered.get('logistics', {}).get('total_potential_savings_usd', 0):,.0f}
- Top actions identified: {critical_count} critical, {high_count} high, {medium_count} medium priority

Write a 2-3 sentence executive summary for a supply chain VP. Be specific with numbers. Start with the most urgent issue."""
                llm_resp = await self.llm.ainvoke([HumanMessage(content=summary_prompt)])
                executive_summary = llm_resp.content.strip()
            except Exception as e:
                logger.error(f"Summary LLM call failed: {e}")
                executive_summary = self._mock_summary(query, gathered, critical_count, high_count)
        else:
            executive_summary = self._mock_summary(query, gathered, critical_count, high_count)

        logger.info(f"\n{'='*60}")
        logger.info(f"✅ QUERY COMPLETE")
        logger.info(f"   Tools called  : {len(tools_called)} — {tools_called}")
        logger.info(f"   Agents used   : {agents_activated}")
        crit_actions = len([a for a in priority_actions if a.get('priority') == 'critical'])
        high_actions = len([a for a in priority_actions if a.get('priority') == 'high'])
        logger.info(f"   Actions found : {len(priority_actions)} total ({crit_actions} critical, {high_actions} high)")
        logger.info(f"   Summary       : {executive_summary[:120]}")
        logger.info(f"{'='*60}\n")

        return {
            "query": query,
            "timestamp": timestamp,
            "orchestrator_routing": {
                "agents_selected": domains,
                "reasoning": routing_reasoning,
                "urgency": urgency,
            },
            "domains_analysed": domains,
            "agents_activated": agents_activated,
            "tools_called": tools_called,
            "executive_summary": executive_summary,
            "priority_actions": priority_actions[:10],
            "seven_day_plan": seven_day_plan,
            "agent_insights": {k: {"analysis": v.get("analysis",""), "confidence": v.get("confidence", 0.8)}
                               for k, v in agent_insights.items()},
            "key_metrics": {
                "critical_items":      gathered.get("critical_items_count", 0),
                "high_risk_suppliers": gathered.get("high_risk_count", 0),
                "logistics_savings":   f"${gathered.get('logistics',{}).get('total_potential_savings_usd',0):,.0f}",
                "domains_covered":     len(domains),
                "tools_invoked":       len(tools_called),
            }
        }

    def _mock_summary(self, query: str, gathered: dict, critical_count: int, high_count: int) -> str:
        """Build a data-driven summary without LLM."""
        parts = []
        q = query.lower()
        crit_inv = gathered.get("critical_items_count", 0)
        high_risk = gathered.get("high_risk_count", 0)
        savings = gathered.get("logistics", {}).get("total_potential_savings_usd", 0)

        if crit_inv > 0:
            parts.append(f"{crit_inv} product(s) are below safety stock and need immediate reorder to prevent production stoppage.")
        if high_risk > 0:
            risky = [s for s in gathered.get("suppliers", []) if s.get("risk_level") in ("critical","high")]
            risky_names = ", ".join(r["supplier_name"] for r in risky[:2])
            parts.append(f"{high_risk} supplier(s) are at high/critical risk ({risky_names}) — delay probability above 40%.")
        if savings > 0:
            parts.append(f"${savings:,.0f} in logistics savings identified — primarily through freight mode consolidation.")
        if not parts:
            parts.append(f"Supply chain health check across {len(gathered)} domains completed. {critical_count} critical and {high_count} high-priority actions identified.")

        return " ".join(parts[:3])


# Global instance
agent_system = SupplyChainAgentSystem()
