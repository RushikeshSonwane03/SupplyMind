import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import {
  TrendingUp, TrendingDown, DollarSign, Truck, Package,
  BarChart3, PieChart as PieIcon, Activity, Filter, RefreshCw
} from "lucide-react";
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from "recharts";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="custom-tooltip">
        <p className="label">{label}</p>
        {payload.map((entry, index) => (
          <p key={index} className="value" style={{ color: entry.color }}>
            {entry.name}: {typeof entry.value === "number" ? entry.value.toLocaleString() : entry.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

const COLORS = ["#6366f1", "#06b6d4", "#f97316", "#22c55e", "#a855f7"];

export default function Analytics() {
  const [products, setProducts]                 = useState([]);
  const [suppliers, setSuppliers]               = useState([]);
  const [selectedProduct, setSelectedProduct]   = useState("all");
  const [selectedSupplier, setSelectedSupplier] = useState("all");
  const [demandTrend, setDemandTrend]           = useState([]);
  const [inventoryLevels, setInventoryLevels]   = useState([]);
  const [supplierPerformance, setSupplierPerformance] = useState([]);
  const [costBreakdown, setCostBreakdown]       = useState([]);
  const [shipmentStats, setShipmentStats]       = useState(null);
  const [loading, setLoading]                   = useState(true);
  const [refreshing, setRefreshing]             = useState(false);

  // useCallback so the function reference is stable — fixes ESLint exhaustive-deps warnings
  const fetchData = useCallback(async () => {
    try {
      const [
        productsRes, suppliersRes, demandRes,
        inventoryRes, supplierRes, costRes, shipmentRes
      ] = await Promise.all([
        axios.get(`${API}/products`),
        axios.get(`${API}/suppliers`),
        axios.get(`${API}/analytics/demand-trend${selectedProduct !== "all" ? `?product_id=${selectedProduct}` : ""}`),
        axios.get(`${API}/analytics/inventory-levels`),
        axios.get(`${API}/analytics/supplier-performance${selectedSupplier !== "all" ? `?supplier_id=${selectedSupplier}` : ""}`),
        axios.get(`${API}/analytics/cost-breakdown`),
        axios.get(`${API}/analytics/shipments${selectedSupplier !== "all" ? `?supplier_id=${selectedSupplier}` : ""}`),
      ]);

      setProducts(productsRes.data.products);
      setSuppliers(suppliersRes.data.suppliers);
      setDemandTrend(demandRes.data.data);
      setInventoryLevels(inventoryRes.data.data);
      setSupplierPerformance(supplierRes.data.data);
      setCostBreakdown(costRes.data.data);
      setShipmentStats(shipmentRes.data);
    } catch (error) {
      console.error("Failed to fetch analytics data:", error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [selectedProduct, selectedSupplier]); // re-create only when filters change

  // Initial load
  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Re-fetch when filters change (fetchData itself changes due to useCallback deps)
  useEffect(() => {
    if (!loading) {
      setRefreshing(true);
      fetchData();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedProduct, selectedSupplier]);

  const radarData = supplierPerformance.length > 0 ? [
    { metric: "Quality",    value: supplierPerformance.reduce((a, b) => a + b.quality, 0) / supplierPerformance.length },
    { metric: "Delivery",   value: supplierPerformance.reduce((a, b) => a + b.delivery, 0) / supplierPerformance.length },
    { metric: "Cost",       value: supplierPerformance.reduce((a, b) => a + b.cost, 0) / supplierPerformance.length },
    { metric: "Risk (inv)", value: 100 - supplierPerformance.reduce((a, b) => a + b.risk, 0) / supplierPerformance.length },
  ] : [];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96" data-testid="analytics-loading">
        <div className="text-zinc-500">Loading analytics...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="analytics-page">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Analytics</h1>
          <p className="text-zinc-500 mt-1">Deep insights from real supply chain data</p>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-zinc-500" />
            <Select value={selectedProduct} onValueChange={setSelectedProduct}>
              <SelectTrigger className="w-44 bg-zinc-900 border-zinc-800" data-testid="product-filter">
                <SelectValue placeholder="Product" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Products</SelectItem>
                {products.map((p) => (
                  <SelectItem key={p.id} value={p.id}>{p.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <Select value={selectedSupplier} onValueChange={setSelectedSupplier}>
            <SelectTrigger className="w-44 bg-zinc-900 border-zinc-800" data-testid="supplier-filter">
              <SelectValue placeholder="Supplier" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Suppliers</SelectItem>
              {suppliers.map((s) => (
                <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button
            variant="outline" size="sm"
            onClick={() => { setRefreshing(true); fetchData(); }}
            disabled={refreshing}
            className="bg-zinc-900 border-zinc-800"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? "animate-spin" : ""}`} />
          </Button>
        </div>
      </div>

      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="bg-zinc-900 border border-zinc-800">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="suppliers">Suppliers</TabsTrigger>
          <TabsTrigger value="shipments">Shipments</TabsTrigger>
          <TabsTrigger value="costs">Costs</TabsTrigger>
          <TabsTrigger value="roi">ROI Analysis</TabsTrigger>
        </TabsList>

        {/* ── Overview ─────────────────────────────────────────────────────── */}
        <TabsContent value="overview" className="space-y-6 mt-6">
          <div className="bento-grid">
            <div className="bento-item span-8" data-testid="demand-trend-chart">
              <h3 className="text-lg font-semibold text-white mb-4">
                Demand Trend Analysis
                {selectedProduct !== "all" && (
                  <span className="text-sm font-normal text-zinc-500 ml-2">
                    ({products.find((p) => p.id === selectedProduct)?.name})
                  </span>
                )}
              </h3>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={demandTrend}>
                    <defs>
                      <linearGradient id="colorActual2" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%"  stopColor="#6366f1" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                    <XAxis dataKey="month" stroke="#71717a" fontSize={12} />
                    <YAxis stroke="#71717a" fontSize={12} />
                    <Tooltip content={<CustomTooltip />} />
                    <Area type="monotone" dataKey="actual" stroke="#6366f1"
                      fillOpacity={1} fill="url(#colorActual2)" strokeWidth={2} name="Actual Demand" />
                    <Line type="monotone" dataKey="forecast" stroke="#06b6d4"
                      strokeWidth={2} strokeDasharray="5 5" dot={false} name="Forecast" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bento-item span-4" data-testid="inventory-health-chart">
              <h3 className="text-lg font-semibold text-white mb-4">Inventory by Warehouse</h3>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={inventoryLevels}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                    <XAxis dataKey="warehouse_name" stroke="#71717a" fontSize={10} angle={-45} textAnchor="end" height={60} />
                    <YAxis stroke="#71717a" fontSize={12} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="total_stock" fill="#6366f1" radius={[4,4,0,0]} name="Stock" />
                    <Bar dataKey="total_value" fill="#22c55e" radius={[4,4,0,0]} name="Value ($)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </TabsContent>

        {/* ── Suppliers ────────────────────────────────────────────────────── */}
        <TabsContent value="suppliers" className="space-y-6 mt-6">
          <div className="grid grid-cols-12 gap-6">
            <div className="col-span-8 bg-zinc-900/50 rounded-xl border border-zinc-800/50 p-6" data-testid="supplier-table">
              <h3 className="text-lg font-semibold text-white mb-4">
                Supplier Performance Matrix
                {selectedSupplier !== "all" && (
                  <span className="text-sm font-normal text-zinc-500 ml-2">
                    ({suppliers.find((s) => s.id === selectedSupplier)?.name})
                  </span>
                )}
              </h3>
              <table className="data-table">
                <thead>
                  <tr><th>Supplier</th><th>Quality</th><th>Delivery</th><th>Cost</th><th>Risk Score</th></tr>
                </thead>
                <tbody>
                  {supplierPerformance.map((supplier, idx) => (
                    <tr key={idx}>
                      <td className="font-medium text-white">{supplier.supplier}</td>
                      {["quality","delivery","cost"].map((key) => (
                        <td key={key}>
                          <div className="flex items-center gap-2">
                            <div className="w-16 h-2 bg-zinc-800 rounded-full overflow-hidden">
                              <div className={`h-full rounded-full ${key==="quality"?"bg-emerald-500":key==="delivery"?"bg-blue-500":"bg-amber-500"}`}
                                style={{ width: `${supplier[key]}%` }} />
                            </div>
                            <span className="text-xs">{supplier[key]?.toFixed(1)}%</span>
                          </div>
                        </td>
                      ))}
                      <td>
                        <span className={`risk-badge ${supplier.risk < 30 ? "low" : supplier.risk < 50 ? "medium" : "high"}`}>
                          {supplier.risk?.toFixed(1)}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="col-span-4 bg-zinc-900/50 rounded-xl border border-zinc-800/50 p-6" data-testid="supplier-radar">
              <h3 className="text-lg font-semibold text-white mb-4">Average Performance</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="#3f3f46" />
                    <PolarAngleAxis dataKey="metric" stroke="#71717a" fontSize={12} />
                    <PolarRadiusAxis stroke="#3f3f46" fontSize={10} />
                    <Radar name="Average" dataKey="value" stroke="#6366f1" fill="#6366f1" fillOpacity={0.3} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </TabsContent>

        {/* ── Shipments ────────────────────────────────────────────────────── */}
        <TabsContent value="shipments" className="space-y-6 mt-6">
          <div className="grid grid-cols-12 gap-6">
            <div className="col-span-8 bg-zinc-900/50 rounded-xl border border-zinc-800/50 p-6" data-testid="shipment-stats">
              <h3 className="text-lg font-semibold text-white mb-6">Shipment Performance</h3>
              <div className="grid grid-cols-4 gap-4 mb-6">
                {[
                  { label: "Total Shipments", value: shipmentStats?.total_shipments || 0, color: "text-white" },
                  { label: "On Time",          value: shipmentStats?.on_time || 0,          color: "text-emerald-400" },
                  { label: "Delayed",          value: shipmentStats?.delayed || 0,          color: "text-amber-400" },
                  { label: "On-Time Rate",     value: `${shipmentStats?.on_time_rate || 0}%`, color: "text-blue-400" },
                ].map((s, i) => (
                  <div key={i} className="p-4 bg-zinc-950 rounded-lg text-center">
                    <p className={`text-3xl font-bold font-mono ${s.color}`}>{s.value}</p>
                    <p className="text-xs text-zinc-500 mt-1">{s.label}</p>
                  </div>
                ))}
              </div>
              {shipmentStats?.delay_reasons && (
                <div>
                  <h4 className="text-sm font-semibold text-zinc-400 mb-3">Delay Reasons</h4>
                  <div className="space-y-2">
                    {Object.entries(shipmentStats.delay_reasons).map(([reason, count], idx) => (
                      <div key={idx} className="flex items-center justify-between p-2 bg-zinc-950 rounded">
                        <span className="text-sm text-zinc-300">{reason}</span>
                        <span className="text-sm font-mono text-zinc-500">{count} shipments</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="col-span-4 bg-zinc-900/50 rounded-xl border border-zinc-800/50 p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Delay Metrics</h3>
              <div className="space-y-6">
                <div className="text-center p-6 bg-zinc-950 rounded-lg">
                  <p className="text-5xl font-bold text-amber-400 font-mono">
                    {shipmentStats?.avg_delay_days?.toFixed(1) || 0}
                  </p>
                  <p className="text-sm text-zinc-500 mt-2">Avg Delay (days)</p>
                </div>
                <div className="p-4 bg-zinc-950 rounded-lg space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-zinc-400">On-Time Target</span>
                    <span className="text-sm font-mono text-zinc-300">95%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-zinc-400">Current Rate</span>
                    <span className={`text-sm font-mono ${(shipmentStats?.on_time_rate||0) >= 95 ? "text-emerald-400" : "text-amber-400"}`}>
                      {shipmentStats?.on_time_rate || 0}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </TabsContent>

        {/* ── Costs ────────────────────────────────────────────────────────── */}
        <TabsContent value="costs" className="space-y-6 mt-6">
          <div className="grid grid-cols-12 gap-6">
            <div className="col-span-6 bg-zinc-900/50 rounded-xl border border-zinc-800/50 p-6" data-testid="cost-pie-chart">
              <h3 className="text-lg font-semibold text-white mb-4">Logistics Cost Breakdown</h3>
              <div className="h-72 flex items-center justify-center">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={costBreakdown} cx="50%" cy="50%"
                      innerRadius={60} outerRadius={100} paddingAngle={2}
                      dataKey="percentage" nameKey="category">
                      {costBreakdown.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="flex flex-wrap justify-center gap-4 mt-4">
                {costBreakdown.map((entry, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[idx % COLORS.length] }} />
                    <span className="text-xs text-zinc-400">{entry.category}: {entry.percentage?.toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="col-span-6 bg-zinc-900/50 rounded-xl border border-zinc-800/50 p-6" data-testid="cost-details">
              <h3 className="text-lg font-semibold text-white mb-4">Cost Details</h3>
              <div className="space-y-4">
                {costBreakdown.map((item, idx) => (
                  <div key={idx} className="flex items-center justify-between p-3 bg-zinc-800/30 rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[idx % COLORS.length] }} />
                      <span className="text-sm text-zinc-300">{item.category}</span>
                    </div>
                    <div className="text-right">
                      <span className="text-sm font-mono text-white">${(item.cost || 0).toLocaleString()}</span>
                      <span className="text-xs text-zinc-500 ml-2">({item.percentage?.toFixed(1)}%)</span>
                    </div>
                  </div>
                ))}
                <div className="pt-4 border-t border-zinc-800 flex justify-between items-center">
                  <span className="text-sm font-semibold text-zinc-300">Total Logistics Cost</span>
                  <span className="text-lg font-bold font-mono text-white">
                    ${(costBreakdown[0]?.total_logistics_cost || 0).toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </TabsContent>

        {/* ── ROI ──────────────────────────────────────────────────────────── */}
        <TabsContent value="roi" className="space-y-6 mt-6">
          <div className="grid grid-cols-12 gap-6">
            <div className="col-span-8 bg-zinc-900/50 rounded-xl border border-zinc-800/50 p-6" data-testid="roi-analysis">
              <h3 className="text-lg font-semibold text-white mb-6">ROI Analysis — Mid-Size Enterprise ($500M Revenue)</h3>
              <div className="space-y-6">
                <div>
                  <h4 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3">Direct Cost Savings</h4>
                  <div className="grid grid-cols-3 gap-4">
                    {[
                      { label: "Inventory Reduction",    value: "$2.5M", desc: "15% reduction in carrying costs" },
                      { label: "Logistics Optimization", value: "$1.8M", desc: "12% transportation savings" },
                      { label: "Procurement Efficiency", value: "$800K", desc: "Reduced maverick spending" },
                    ].map((item, idx) => (
                      <div key={idx} className="p-4 bg-zinc-800/30 rounded-lg">
                        <p className="text-xs text-zinc-500 mb-1">{item.label}</p>
                        <p className="text-2xl font-bold text-emerald-400 font-mono">{item.value}</p>
                        <p className="text-xs text-zinc-500 mt-1">{item.desc}</p>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3">AI-Driven Improvements</h4>
                  <div className="grid grid-cols-3 gap-4">
                    {[
                      { label: "Disruption Response", value: "40%",  desc: "Faster with LangChain agents" },
                      { label: "Forecast Accuracy",   value: "25%",  desc: "Improvement with ML" },
                      { label: "Manual Effort",        value: "60%",  desc: "Reduction in planning" },
                    ].map((item, idx) => (
                      <div key={idx} className="p-4 bg-zinc-800/30 rounded-lg">
                        <p className="text-xs text-zinc-500 mb-1">{item.label}</p>
                        <p className="text-2xl font-bold text-blue-400 font-mono">{item.value}</p>
                        <p className="text-xs text-zinc-500 mt-1">{item.desc}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <div className="col-span-4 space-y-4">
              {[
                { label: "Total Annual Benefit", value: "$5.1M",      color: "#10b981" },
                { label: "Implementation Cost",  value: "$1.2M",      color: "#f97316" },
                { label: "Payback Period",        value: "2.8 months", color: "#6366f1" },
                { label: "3-Year ROI",            value: "425%",       color: "#a855f7" },
              ].map((item, idx) => (
                <div key={idx} className="p-5 bg-zinc-900/50 rounded-xl border border-zinc-800/50">
                  <p className="text-xs text-zinc-500 uppercase tracking-wider">{item.label}</p>
                  <p className="text-3xl font-bold font-mono mt-2" style={{ color: item.color }}>{item.value}</p>
                </div>
              ))}
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
