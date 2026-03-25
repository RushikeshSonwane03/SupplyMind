"""
SupplyMind Datasets Module
Contains realistic supply chain datasets for demonstration
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# ============================================================
# PRODUCTS DATASET
# ============================================================
PRODUCTS = [
    {"id": "PRD-001", "name": "Industrial Motor A500", "category": "Motors", "unit_cost": 450.00, "lead_time_days": 14},
    {"id": "PRD-002", "name": "Hydraulic Pump HP-200", "category": "Pumps", "unit_cost": 890.00, "lead_time_days": 21},
    {"id": "PRD-003", "name": "Control Valve CV-50", "category": "Valves", "unit_cost": 125.00, "lead_time_days": 7},
    {"id": "PRD-004", "name": "Bearing Assembly BA-100", "category": "Bearings", "unit_cost": 78.50, "lead_time_days": 5},
    {"id": "PRD-005", "name": "Sensor Module SM-X1", "category": "Electronics", "unit_cost": 234.00, "lead_time_days": 10},
    {"id": "PRD-006", "name": "Gear Box GB-300", "category": "Transmission", "unit_cost": 1250.00, "lead_time_days": 28},
    {"id": "PRD-007", "name": "Filter Element FE-20", "category": "Filters", "unit_cost": 45.00, "lead_time_days": 3},
    {"id": "PRD-008", "name": "Circuit Board CB-Pro", "category": "Electronics", "unit_cost": 320.00, "lead_time_days": 12},
    {"id": "PRD-009", "name": "Steel Shaft SS-150", "category": "Components", "unit_cost": 180.00, "lead_time_days": 8},
    {"id": "PRD-010", "name": "Pneumatic Cylinder PC-80", "category": "Pneumatics", "unit_cost": 560.00, "lead_time_days": 15},
]

# ============================================================
# SUPPLIERS DATASET
# ============================================================
SUPPLIERS = [
    {
        "id": "SUP-001", "name": "Acme Industrial Corp", "country": "USA", "region": "North America",
        "reliability_score": 92, "quality_rating": 4.5, "on_time_delivery": 94,
        "financial_health": "Strong", "certifications": ["ISO 9001", "ISO 14001"],
        "risk_factors": ["Labor disputes in region", "Single source for motors"],
        "annual_volume": 2500000, "payment_terms": "Net 30"
    },
    {
        "id": "SUP-002", "name": "Global Parts Ltd", "country": "Germany", "region": "Europe",
        "reliability_score": 88, "quality_rating": 4.7, "on_time_delivery": 91,
        "financial_health": "Strong", "certifications": ["ISO 9001", "IATF 16949"],
        "risk_factors": ["Currency fluctuation", "Long lead times"],
        "annual_volume": 1800000, "payment_terms": "Net 45"
    },
    {
        "id": "SUP-003", "name": "FastShip Logistics", "country": "China", "region": "Asia Pacific",
        "reliability_score": 78, "quality_rating": 3.9, "on_time_delivery": 82,
        "financial_health": "Moderate", "certifications": ["ISO 9001"],
        "risk_factors": ["Geopolitical tensions", "Quality inconsistency", "Port congestion"],
        "annual_volume": 3200000, "payment_terms": "Net 60"
    },
    {
        "id": "SUP-004", "name": "MegaSupply Co", "country": "Mexico", "region": "North America",
        "reliability_score": 85, "quality_rating": 4.2, "on_time_delivery": 88,
        "financial_health": "Moderate", "certifications": ["ISO 9001", "ISO 45001"],
        "risk_factors": ["Infrastructure limitations", "Customs delays"],
        "annual_volume": 1500000, "payment_terms": "Net 30"
    },
    {
        "id": "SUP-005", "name": "TechComponents Asia", "country": "Taiwan", "region": "Asia Pacific",
        "reliability_score": 90, "quality_rating": 4.6, "on_time_delivery": 93,
        "financial_health": "Strong", "certifications": ["ISO 9001", "ISO 14001", "IATF 16949"],
        "risk_factors": ["Semiconductor shortage risk", "Natural disaster exposure"],
        "annual_volume": 2100000, "payment_terms": "Net 45"
    },
    {
        "id": "SUP-006", "name": "EuroSteel Industries", "country": "Poland", "region": "Europe",
        "reliability_score": 82, "quality_rating": 4.1, "on_time_delivery": 86,
        "financial_health": "Moderate", "certifications": ["ISO 9001"],
        "risk_factors": ["Energy cost volatility", "Raw material dependency"],
        "annual_volume": 980000, "payment_terms": "Net 30"
    },
    {
        "id": "SUP-007", "name": "Pacific Rim Trading", "country": "Vietnam", "region": "Asia Pacific",
        "reliability_score": 75, "quality_rating": 3.7, "on_time_delivery": 79,
        "financial_health": "Developing", "certifications": ["ISO 9001"],
        "risk_factors": ["Quality control issues", "Limited capacity", "Communication barriers"],
        "annual_volume": 750000, "payment_terms": "Net 60"
    },
    {
        "id": "SUP-008", "name": "Precision Parts USA", "country": "USA", "region": "North America",
        "reliability_score": 94, "quality_rating": 4.8, "on_time_delivery": 96,
        "financial_health": "Strong", "certifications": ["ISO 9001", "AS9100", "IATF 16949"],
        "risk_factors": ["Higher costs", "Capacity constraints during peak"],
        "annual_volume": 1200000, "payment_terms": "Net 30"
    },
]

# ============================================================
# WAREHOUSES DATASET
# ============================================================
WAREHOUSES = [
    {"id": "WH-EAST", "name": "East Coast Distribution", "location": "Newark, NJ", "capacity": 50000, "utilization": 78},
    {"id": "WH-WEST", "name": "West Coast Hub", "location": "Los Angeles, CA", "capacity": 65000, "utilization": 82},
    {"id": "WH-CENTRAL", "name": "Central Logistics", "location": "Chicago, IL", "capacity": 45000, "utilization": 71},
    {"id": "WH-SOUTH", "name": "Southern Distribution", "location": "Houston, TX", "capacity": 40000, "utilization": 65},
]

# ============================================================
# GENERATE HISTORICAL DEMAND DATA (24 months)
# ============================================================
def generate_demand_data():
    """Generate 24 months of historical demand data for all products"""
    data = []
    base_date = datetime.now() - timedelta(days=730)  # 2 years ago
    
    for product in PRODUCTS:
        # Base demand varies by product
        base_demand = random.randint(500, 2000)
        
        for month_offset in range(24):
            date = base_date + timedelta(days=month_offset * 30)
            
            # Add seasonality (higher in Q4, lower in Q1)
            month = date.month
            if month in [10, 11, 12]:
                seasonality = 1.3
            elif month in [1, 2, 3]:
                seasonality = 0.8
            elif month in [6, 7, 8]:
                seasonality = 1.1
            else:
                seasonality = 1.0
            
            # Add trend (slight growth over time)
            trend = 1 + (month_offset * 0.01)
            
            # Add random noise
            noise = random.uniform(0.85, 1.15)
            
            actual_demand = int(base_demand * seasonality * trend * noise)
            
            # Generate forecast (with some error)
            forecast_error = random.uniform(0.9, 1.1)
            forecast_demand = int(actual_demand * forecast_error)
            
            data.append({
                "product_id": product["id"],
                "product_name": product["name"],
                "category": product["category"],
                "date": date.strftime("%Y-%m-%d"),
                "month": date.strftime("%Y-%m"),
                "actual_demand": actual_demand,
                "forecast_demand": forecast_demand,
                "forecast_error": abs(actual_demand - forecast_demand) / actual_demand * 100
            })
    
    return pd.DataFrame(data)

# ============================================================
# GENERATE INVENTORY DATA
# ============================================================
def generate_inventory_data():
    """Generate current inventory levels across warehouses"""
    data = []
    
    for product in PRODUCTS:
        for warehouse in WAREHOUSES:
            # Random inventory levels
            current_stock = random.randint(100, 2000)
            reorder_point = random.randint(200, 500)
            safety_stock = int(reorder_point * 0.3)
            max_stock = reorder_point * 3
            
            # Calculate days of supply
            avg_daily_demand = random.randint(20, 80)
            days_of_supply = current_stock / avg_daily_demand if avg_daily_demand > 0 else 0
            
            # Determine status
            if current_stock < safety_stock:
                status = "Critical"
            elif current_stock < reorder_point:
                status = "Low"
            elif current_stock > max_stock:
                status = "Overstock"
            else:
                status = "Optimal"
            
            data.append({
                "product_id": product["id"],
                "product_name": product["name"],
                "warehouse_id": warehouse["id"],
                "warehouse_name": warehouse["name"],
                "warehouse_location": warehouse["location"],
                "current_stock": current_stock,
                "reorder_point": reorder_point,
                "safety_stock": safety_stock,
                "max_stock": max_stock,
                "avg_daily_demand": avg_daily_demand,
                "days_of_supply": round(days_of_supply, 1),
                "status": status,
                "unit_cost": product["unit_cost"],
                "inventory_value": current_stock * product["unit_cost"]
            })
    
    return pd.DataFrame(data)

# ============================================================
# GENERATE SHIPMENT DATA
# ============================================================
def generate_shipment_data():
    """Generate shipment history with delays"""
    data = []
    base_date = datetime.now() - timedelta(days=180)
    
    shipping_modes = ["Air", "Sea", "Ground", "Rail"]
    carriers = ["FedEx", "UPS", "DHL", "Maersk", "BNSF", "XPO Logistics"]
    
    for i in range(500):
        ship_date = base_date + timedelta(days=random.randint(0, 180))
        product = random.choice(PRODUCTS)
        supplier = random.choice(SUPPLIERS)
        warehouse = random.choice(WAREHOUSES)
        
        # Expected transit time based on region
        if supplier["region"] == "Asia Pacific":
            base_transit = random.randint(14, 28)
        elif supplier["region"] == "Europe":
            base_transit = random.randint(7, 14)
        else:
            base_transit = random.randint(3, 7)
        
        # Add delay probability
        delay_probability = 1 - (supplier["on_time_delivery"] / 100)
        has_delay = random.random() < delay_probability
        
        if has_delay:
            delay_days = random.randint(1, 10)
            delay_reason = random.choice([
                "Port congestion", "Weather delays", "Customs clearance",
                "Carrier capacity", "Documentation issues", "Equipment failure"
            ])
        else:
            delay_days = 0
            delay_reason = None
        
        actual_transit = base_transit + delay_days
        expected_arrival = ship_date + timedelta(days=base_transit)
        actual_arrival = ship_date + timedelta(days=actual_transit)
        
        data.append({
            "shipment_id": f"SHP-{i+1:05d}",
            "product_id": product["id"],
            "product_name": product["name"],
            "supplier_id": supplier["id"],
            "supplier_name": supplier["name"],
            "origin_country": supplier["country"],
            "destination_warehouse": warehouse["name"],
            "shipping_mode": random.choice(shipping_modes),
            "carrier": random.choice(carriers),
            "ship_date": ship_date.strftime("%Y-%m-%d"),
            "expected_arrival": expected_arrival.strftime("%Y-%m-%d"),
            "actual_arrival": actual_arrival.strftime("%Y-%m-%d"),
            "expected_transit_days": base_transit,
            "actual_transit_days": actual_transit,
            "delay_days": delay_days,
            "delay_reason": delay_reason,
            "quantity": random.randint(50, 500),
            "status": "Delivered" if actual_arrival < datetime.now() else "In Transit"
        })
    
    return pd.DataFrame(data)

# ============================================================
# GENERATE SUPPLIER PERFORMANCE DATA
# ============================================================
def generate_supplier_performance():
    """Generate monthly supplier performance metrics"""
    data = []
    base_date = datetime.now() - timedelta(days=365)
    
    for supplier in SUPPLIERS:
        for month_offset in range(12):
            date = base_date + timedelta(days=month_offset * 30)
            
            # Add variation to base metrics
            quality = min(100, max(0, supplier["quality_rating"] * 20 + random.uniform(-5, 5)))
            delivery = min(100, max(0, supplier["on_time_delivery"] + random.uniform(-8, 5)))
            reliability = min(100, max(0, supplier["reliability_score"] + random.uniform(-5, 5)))
            
            # Calculate cost competitiveness (inverse of quality in some sense)
            cost_score = random.uniform(70, 95)
            
            # Calculate overall score
            overall = (quality * 0.3 + delivery * 0.3 + reliability * 0.25 + cost_score * 0.15)
            
            data.append({
                "supplier_id": supplier["id"],
                "supplier_name": supplier["name"],
                "country": supplier["country"],
                "region": supplier["region"],
                "month": date.strftime("%Y-%m"),
                "quality_score": round(quality, 1),
                "delivery_score": round(delivery, 1),
                "reliability_score": round(reliability, 1),
                "cost_score": round(cost_score, 1),
                "overall_score": round(overall, 1),
                "orders_placed": random.randint(10, 50),
                "orders_fulfilled": random.randint(8, 50),
                "defect_rate": round(random.uniform(0.1, 3.0), 2),
                "response_time_hours": random.randint(2, 48)
            })
    
    return pd.DataFrame(data)

# ============================================================
# NEWS & RISK EVENTS DATA
# ============================================================
NEWS_EVENTS = [
    {
        "id": "NEWS-001",
        "date": "2026-03-15",
        "headline": "Port workers strike threatens West Coast supply chains",
        "source": "Reuters",
        "sentiment": "negative",
        "impact": "high",
        "affected_regions": ["North America", "Asia Pacific"],
        "keywords": ["port strike", "logistics", "delays"]
    },
    {
        "id": "NEWS-002",
        "date": "2026-03-14",
        "headline": "Semiconductor shortage easing as new fabs come online",
        "source": "Bloomberg",
        "sentiment": "positive",
        "impact": "medium",
        "affected_regions": ["Asia Pacific"],
        "keywords": ["semiconductor", "supply", "manufacturing"]
    },
    {
        "id": "NEWS-003",
        "date": "2026-03-13",
        "headline": "Typhoon warning issued for Taiwan manufacturing region",
        "source": "AP News",
        "sentiment": "negative",
        "impact": "high",
        "affected_regions": ["Asia Pacific"],
        "keywords": ["typhoon", "natural disaster", "taiwan"]
    },
    {
        "id": "NEWS-004",
        "date": "2026-03-12",
        "headline": "European energy prices stabilize after winter peak",
        "source": "Financial Times",
        "sentiment": "positive",
        "impact": "medium",
        "affected_regions": ["Europe"],
        "keywords": ["energy", "prices", "manufacturing costs"]
    },
    {
        "id": "NEWS-005",
        "date": "2026-03-11",
        "headline": "Trade tensions escalate between major economies",
        "source": "WSJ",
        "sentiment": "negative",
        "impact": "high",
        "affected_regions": ["Asia Pacific", "North America"],
        "keywords": ["trade war", "tariffs", "geopolitical"]
    },
    {
        "id": "NEWS-006",
        "date": "2026-03-10",
        "headline": "Major automotive supplier announces capacity expansion",
        "source": "Industry Week",
        "sentiment": "positive",
        "impact": "low",
        "affected_regions": ["North America"],
        "keywords": ["capacity", "expansion", "automotive"]
    },
    {
        "id": "NEWS-007",
        "date": "2026-03-09",
        "headline": "Shipping container rates drop 15% from peak",
        "source": "Shipping Gazette",
        "sentiment": "positive",
        "impact": "medium",
        "affected_regions": ["Global"],
        "keywords": ["shipping", "container", "freight rates"]
    },
    {
        "id": "NEWS-008",
        "date": "2026-03-08",
        "headline": "Quality issues reported at major Chinese supplier",
        "source": "Supply Chain Dive",
        "sentiment": "negative",
        "impact": "medium",
        "affected_regions": ["Asia Pacific"],
        "keywords": ["quality", "defects", "recall"]
    },
]

# ============================================================
# COST DATA
# ============================================================
def generate_cost_data():
    """Generate logistics cost breakdown data"""
    data = []
    base_date = datetime.now() - timedelta(days=365)
    
    cost_categories = [
        {"category": "Transportation", "base_pct": 42, "volatility": 5},
        {"category": "Warehousing", "base_pct": 28, "volatility": 2},
        {"category": "Inventory Carrying", "base_pct": 15, "volatility": 3},
        {"category": "Packaging", "base_pct": 8, "volatility": 1},
        {"category": "Administration", "base_pct": 7, "volatility": 1},
    ]
    
    for month_offset in range(12):
        date = base_date + timedelta(days=month_offset * 30)
        total_cost = random.randint(800000, 1200000)
        
        for cat in cost_categories:
            pct = cat["base_pct"] + random.uniform(-cat["volatility"], cat["volatility"])
            cost = total_cost * (pct / 100)
            
            data.append({
                "month": date.strftime("%Y-%m"),
                "category": cat["category"],
                "percentage": round(pct, 1),
                "cost": round(cost, 2),
                "total_logistics_cost": total_cost
            })
    
    return pd.DataFrame(data)

# ============================================================
# INITIALIZE ALL DATASETS
# ============================================================
DEMAND_DATA = generate_demand_data()
INVENTORY_DATA = generate_inventory_data()
SHIPMENT_DATA = generate_shipment_data()
SUPPLIER_PERFORMANCE_DATA = generate_supplier_performance()
COST_DATA = generate_cost_data()

# ============================================================
# DATA ACCESS FUNCTIONS
# ============================================================
def get_products():
    return PRODUCTS

def get_suppliers():
    return SUPPLIERS

def get_warehouses():
    return WAREHOUSES

def get_demand_by_product(product_id: str = None):
    if product_id:
        return DEMAND_DATA[DEMAND_DATA["product_id"] == product_id].to_dict(orient="records")
    return DEMAND_DATA.to_dict(orient="records")

def get_inventory_by_warehouse(warehouse_id: str = None):
    if warehouse_id:
        return INVENTORY_DATA[INVENTORY_DATA["warehouse_id"] == warehouse_id].to_dict(orient="records")
    return INVENTORY_DATA.to_dict(orient="records")

def get_inventory_by_product(product_id: str = None):
    if product_id:
        return INVENTORY_DATA[INVENTORY_DATA["product_id"] == product_id].to_dict(orient="records")
    return INVENTORY_DATA.to_dict(orient="records")

def get_shipments_by_supplier(supplier_id: str = None):
    if supplier_id:
        return SHIPMENT_DATA[SHIPMENT_DATA["supplier_id"] == supplier_id].to_dict(orient="records")
    return SHIPMENT_DATA.to_dict(orient="records")

def get_supplier_performance(supplier_id: str = None):
    if supplier_id:
        return SUPPLIER_PERFORMANCE_DATA[SUPPLIER_PERFORMANCE_DATA["supplier_id"] == supplier_id].to_dict(orient="records")
    return SUPPLIER_PERFORMANCE_DATA.to_dict(orient="records")

def get_news_events():
    return NEWS_EVENTS

def get_cost_breakdown():
    return COST_DATA.to_dict(orient="records")

def get_demand_summary():
    """Get aggregated demand summary by product"""
    summary = DEMAND_DATA.groupby(["product_id", "product_name", "category"]).agg({
        "actual_demand": ["sum", "mean", "std"],
        "forecast_demand": ["sum", "mean"],
        "forecast_error": "mean"
    }).reset_index()
    summary.columns = ["product_id", "product_name", "category", 
                       "total_demand", "avg_demand", "demand_std",
                       "total_forecast", "avg_forecast", "avg_forecast_error"]
    return summary.to_dict(orient="records")

def get_inventory_summary():
    """Get aggregated inventory summary"""
    summary = INVENTORY_DATA.groupby(["warehouse_id", "warehouse_name"]).agg({
        "current_stock": "sum",
        "inventory_value": "sum",
        "days_of_supply": "mean"
    }).reset_index()
    summary.columns = ["warehouse_id", "warehouse_name", "total_stock", "total_value", "avg_days_supply"]
    return summary.to_dict(orient="records")

def get_supplier_risk_data(supplier_id: str):
    """Get comprehensive risk data for a supplier"""
    supplier = next((s for s in SUPPLIERS if s["id"] == supplier_id), None)
    if not supplier:
        return None
    
    # Get performance history
    perf = SUPPLIER_PERFORMANCE_DATA[SUPPLIER_PERFORMANCE_DATA["supplier_id"] == supplier_id]
    
    # Get shipment history
    ships = SHIPMENT_DATA[SHIPMENT_DATA["supplier_id"] == supplier_id]
    
    # Get relevant news
    news = [n for n in NEWS_EVENTS if supplier["region"] in n["affected_regions"]]
    
    # Calculate risk metrics
    delay_rate = (ships["delay_days"] > 0).sum() / len(ships) * 100 if len(ships) > 0 else 0
    avg_delay = ships[ships["delay_days"] > 0]["delay_days"].mean() if len(ships[ships["delay_days"] > 0]) > 0 else 0
    
    return {
        "supplier": supplier,
        "performance_trend": perf.tail(6).to_dict(orient="records"),
        "shipment_stats": {
            "total_shipments": len(ships),
            "on_time_shipments": len(ships[ships["delay_days"] == 0]),
            "delayed_shipments": len(ships[ships["delay_days"] > 0]),
            "delay_rate_pct": round(delay_rate, 1),
            "avg_delay_days": round(avg_delay, 1)
        },
        "recent_news": news[:5],
        "risk_factors": supplier["risk_factors"]
    }