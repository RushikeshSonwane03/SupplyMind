import { useState, useEffect } from "react";
import axios from "axios";
import { 
  TrendingUp, 
  TrendingDown, 
  Package, 
  Truck, 
  AlertTriangle,
  DollarSign,
  BarChart3,
  Activity,
  ArrowUpRight,
  ArrowDownRight,
  Zap,
  Bot,
  RefreshCw,
  Filter
} from "lucide-react";
import { 
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  PieChart, Pie, Cell
} from "recharts";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// Custom tooltip for charts
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="custom-tooltip">
        <p className="label">{label}</p>
        {payload.map((entry, index) => (
          <p key={index} className="value" style={{ color: entry.color }}>
            {entry.name}: {entry.value.toLocaleString()}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

// KPI Card Component
const KPICard = ({ label, value, change, trend, icon: Icon, color }) => (
  <div className="kpi-card" style={{ '--accent-color': color }} data-testid={`kpi-${label.toLowerCase().replace(/\s+/g, '-')}`}>
    <div className="flex justify-between items-start">
      <div>
        <p className="kpi-label">{label}</p>
        <p className="kpi-value">{value}</p>
        <div className={`kpi-change ${trend === 'up' ? 'positive' : 'negative'}`}>
          {trend === 'up' ? (
            <ArrowUpRight className="w-4 h-4" />
          ) : (
            <ArrowDownRight className="w-4 h-4" />
          )}
          <span>{Math.abs(change)}% vs last period</span>
        </div>
      </div>
      <div 
        className="w-10 h-10 rounded-lg flex items-center justify-center"
        style={{ backgroundColor: `${color}20` }}
      >
        <Icon className="w-5 h-5" style={{ color }} />
      </div>
    </div>
  </div>
);

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [kpis, setKpis] = useState([]);
  const [demandTrend, setDemandTrend] = useState([]);
  const [inventoryLevels, setInventoryLevels] = useState([]);
  const [products, setProducts] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState("all");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchDashboardData = async (productFilter = null) => {
    try {
      const [metricsRes, kpisRes, demandRes, inventoryRes, productsRes] = await Promise.all([
        axios.get(`${API}/metrics/system`),
        axios.get(`${API}/metrics/kpis`),
        axios.get(`${API}/analytics/demand-trend${productFilter ? `?product_id=${productFilter}` : ''}`),
        axios.get(`${API}/analytics/inventory-levels`),
        axios.get(`${API}/products`),
      ]);
      
      setMetrics(metricsRes.data);
      setKpis(kpisRes.data.kpis);
      setDemandTrend(demandRes.data.data);
      setInventoryLevels(inventoryRes.data.data);
      setProducts(productsRes.data.products);
    } catch (error) {
      console.error("Failed to fetch dashboard data:", error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const handleProductChange = async (value) => {
    setSelectedProduct(value);
    setRefreshing(true);
    await fetchDashboardData(value === "all" ? null : value);
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await fetchDashboardData(selectedProduct === "all" ? null : selectedProduct);
  };

  const kpiIcons = {
    "Cost Savings": DollarSign,
    "On-Time Delivery": Truck,
    "Inventory Value": Package,
    "Supplier Risk Avg": AlertTriangle,
    "Forecast Accuracy": BarChart3,
    "Fill Rate": Activity,
  };

  const kpiColors = {
    "Cost Savings": "#10b981",
    "On-Time Delivery": "#3b82f6",
    "Inventory Value": "#f59e0b",
    "Supplier Risk Avg": "#f97316",
    "Forecast Accuracy": "#6366f1",
    "Fill Rate": "#22c55e",
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96" data-testid="dashboard-loading">
        <div className="text-zinc-500">Loading dashboard...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="dashboard-page">
      {/* Header with Controls */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Dashboard</h1>
          <p className="text-zinc-500 mt-1">Supply chain intelligence powered by LangChain agents</p>
        </div>
        <div className="flex items-center gap-4">
          {/* Product Filter */}
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-zinc-500" />
            <Select value={selectedProduct} onValueChange={handleProductChange}>
              <SelectTrigger className="w-48 bg-zinc-900 border-zinc-800" data-testid="product-filter">
                <SelectValue placeholder="Filter by product" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Products</SelectItem>
                {products.map((product) => (
                  <SelectItem key={product.id} value={product.id}>
                    {product.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          
          <Button 
            variant="outline" 
            size="sm" 
            onClick={handleRefresh}
            disabled={refreshing}
            className="bg-zinc-900 border-zinc-800"
            data-testid="refresh-btn"
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          
          <div className="flex items-center gap-3 px-4 py-2 bg-zinc-900 rounded-lg border border-zinc-800">
            <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
            <span className="text-sm text-zinc-400">System Active</span>
          </div>
        </div>
      </div>

      {/* System Stats Banner */}
      <div className="p-4 bg-gradient-to-r from-indigo-500/10 to-purple-500/10 rounded-xl border border-indigo-500/20">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-6 flex-wrap">
            <div className="flex items-center gap-2">
              <Bot className="w-5 h-5 text-indigo-400" />
              <span className="text-sm text-zinc-300">{metrics?.active_agents || 8} Agents</span>
            </div>
            <div className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-purple-400" />
              <span className="text-sm text-zinc-300">{metrics?.total_workflows || 10} Workflows</span>
            </div>
            <div className="flex items-center gap-2">
              <Package className="w-5 h-5 text-cyan-400" />
              <span className="text-sm text-zinc-300">{metrics?.products_tracked || 10} Products</span>
            </div>
            <div className="flex items-center gap-2">
              <Truck className="w-5 h-5 text-amber-400" />
              <span className="text-sm text-zinc-300">{metrics?.suppliers_monitored || 8} Suppliers</span>
            </div>
            <div className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-emerald-400" />
              <span className="text-sm text-zinc-300">{metrics?.business_queries || 0} Queries Run</span>
            </div>
            <div className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-blue-400" />
              <span className="text-sm text-zinc-300">{metrics?.workflow_runs || 0} Workflow Runs</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${metrics?.llm_active ? "bg-emerald-400 animate-pulse" : "bg-amber-400"}`} />
            <span className="text-xs text-zinc-400">
              {metrics?.llm_active ? "LLM Online (Ollama llama3.2:latest.2:latest.2:latest.2:latest )" : "Mock Mode — LLM Offline"}
            </span>
          </div>
        </div>
      </div>

      {/* KPI Grid */}
      <div className="kpi-grid">
        {kpis.map((kpi) => (
          <KPICard
            key={kpi.name}
            label={kpi.name}
            value={kpi.value}
            change={kpi.change}
            trend={kpi.trend}
            icon={kpiIcons[kpi.name] || Activity}
            color={kpiColors[kpi.name] || "#6366f1"}
          />
        ))}
      </div>

      {/* Charts Grid */}
      <div className="bento-grid">
        {/* Demand Trend Chart */}
        <div className="bento-item span-8" data-testid="demand-chart">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h3 className="text-lg font-semibold text-white">Demand Trend</h3>
              <p className="text-xs text-zinc-500">
                {selectedProduct === "all" ? "Aggregated across all products" : `Filtered by ${products.find(p => p.id === selectedProduct)?.name}`}
              </p>
            </div>
            <span className="text-xs text-zinc-500">Actual vs Forecast</span>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={demandTrend}>
                <defs>
                  <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis dataKey="month" stroke="#71717a" fontSize={12} />
                <YAxis stroke="#71717a" fontSize={12} />
                <Tooltip content={<CustomTooltip />} />
                <Area 
                  type="monotone" 
                  dataKey="actual" 
                  stroke="#6366f1" 
                  fillOpacity={1} 
                  fill="url(#colorActual)"
                  strokeWidth={2}
                  name="Actual"
                />
                <Area 
                  type="monotone" 
                  dataKey="forecast" 
                  stroke="#06b6d4" 
                  fillOpacity={1} 
                  fill="url(#colorForecast)"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  name="Forecast"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* System Status */}
        <div className="bento-item span-4" data-testid="system-status">
          <h3 className="text-lg font-semibold text-white mb-4">System Status</h3>
          <div className="space-y-3">
            {[
              { label: "LangChain Agents", value: metrics?.active_agents || 8,   icon: Bot,          color: "#6366f1" },
              { label: "Tools Available",  value: 12,                              icon: Zap,          color: "#a855f7" },
              { label: "Workflow Runs",    value: metrics?.workflow_runs || 0,     icon: Activity,     color: "#22c55e" },
              { label: "Business Queries", value: metrics?.business_queries || 0,  icon: BarChart3,    color: "#3b82f6" },
              { label: "Inventory Health", value: `${metrics?.inventory_health || 0}%`, icon: AlertTriangle, color: "#f59e0b" },
            ].map((item, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 bg-zinc-900/50 rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center"
                    style={{ backgroundColor: `${item.color}20` }}>
                    <item.icon className="w-4 h-4" style={{ color: item.color }} />
                  </div>
                  <span className="text-sm text-zinc-300">{item.label}</span>
                </div>
                <span className="text-lg font-mono font-bold text-white">{item.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Inventory Levels */}
        <div className="bento-item span-6" data-testid="inventory-chart">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-white">Inventory by Warehouse</h3>
          </div>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={inventoryLevels} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis type="number" stroke="#71717a" fontSize={12} />
                <YAxis dataKey="warehouse_name" type="category" stroke="#71717a" fontSize={10} width={100} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="total_stock" fill="#6366f1" radius={[0, 4, 4, 0]} name="Stock" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="bento-item span-6" data-testid="quick-actions">
          <h3 className="text-lg font-semibold text-white mb-4">Quick Actions</h3>
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: "Ask AI",             icon: Bot,          color: "#6366f1", path: "/agents" },
              { label: "Run Workflows",      icon: Zap,          color: "#a855f7", path: "/workflows" },
              { label: "Analyze Suppliers",  icon: AlertTriangle,color: "#f97316", path: "/supplier-risk" },
              { label: "View Reports",       icon: BarChart3,    color: "#22c55e", path: "/reports" },
            ].map((action) => (
              <button
                key={action.label}
                className="flex items-center gap-3 p-4 bg-zinc-900/50 rounded-lg border border-zinc-800 hover:border-zinc-700 transition-colors text-left"
                data-testid={`action-${action.label.toLowerCase().replace(/\s+/g, "-")}`}
                onClick={() => window.location.href = action.path}
              >
                <div className="w-10 h-10 rounded-lg flex items-center justify-center"
                  style={{ backgroundColor: `${action.color}20` }}>
                  <action.icon className="w-5 h-5" style={{ color: action.color }} />
                </div>
                <span className="text-sm font-medium text-zinc-300">{action.label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
