import { useState, useEffect } from "react";
import axios from "axios";
import {
  TrendingUp, Package, Warehouse, Route, Clock, AlertTriangle,
  ShoppingCart, DollarSign, Factory, Leaf, Bot, Play, Loader2,
  CheckCircle2, XCircle, ChevronDown, ChevronUp, Zap, AlertCircle,
  Sparkles, Brain,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// ── Constants ─────────────────────────────────────────────────────────────────
const WORKFLOW_ICONS = {
  "Demand Forecasting": TrendingUp, "Inventory Optimization": Package,
  "Warehouse Automation": Warehouse, "Route Optimization": Route,
  "Shipment Delay Prediction": Clock, "Supplier Risk Detection": AlertTriangle,
  "Procurement Automation": ShoppingCart, "Logistics Cost Optimization": DollarSign,
  "Production Planning": Factory, "Sustainability Tracking": Leaf,
};
const CATEGORY_COLORS = {
  customer: "#a855f7", operations: "#3b82f6", infrastructure: "#06b6d4",
  business: "#f97316", sustainability: "#22c55e",
};
const TYPE_LABELS = {
  execution: "Execution", efficiency: "Efficiency",
  exception: "Exception", expansion: "Expansion",
};
const PRIORITY_CFG = {
  critical: { color: "#ef4444", bg: "rgba(239,68,68,0.1)",  border: "rgba(239,68,68,0.25)",  Icon: XCircle,       label: "Critical" },
  high:     { color: "#f97316", bg: "rgba(249,115,22,0.1)", border: "rgba(249,115,22,0.25)", Icon: AlertTriangle, label: "High" },
  medium:   { color: "#f59e0b", bg: "rgba(245,158,11,0.1)", border: "rgba(245,158,11,0.25)", Icon: AlertCircle,   label: "Medium" },
  low:      { color: "#22c55e", bg: "rgba(34,197,94,0.1)",  border: "rgba(34,197,94,0.25)",  Icon: CheckCircle2,  label: "Low" },
};

// ── Reusable small components ─────────────────────────────────────────────────
const PriorityBadge = ({ level }) => {
  const cfg = PRIORITY_CFG[level] || PRIORITY_CFG.low;
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold"
      style={{ backgroundColor: cfg.bg, color: cfg.color, border: `1px solid ${cfg.border}` }}>
      <cfg.Icon className="w-3 h-3" />{cfg.label}
    </span>
  );
};

const SummaryPill = ({ label, value, color }) => (
  <div className="flex flex-col items-center px-3 py-2 rounded-lg bg-zinc-900/60 border border-zinc-800">
    <span className="text-base font-bold font-mono" style={{ color }}>{value}</span>
    <span className="text-xs text-zinc-500 mt-0.5">{label}</span>
  </div>
);

// ── Agent Analysis panel (shown after every workflow run) ─────────────────────
const AgentAnalysisPanel = ({ analysis, recommendations, agentName }) => {
  const [open, setOpen] = useState(true);
  if (!analysis && (!recommendations || recommendations.length === 0)) return null;
  return (
    <div className="rounded-xl border border-indigo-500/25 bg-indigo-500/5 overflow-hidden">
      <button onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 px-4 py-3 hover:bg-indigo-500/10 transition-colors">
        <Brain className="w-4 h-4 text-indigo-400 shrink-0" />
        <span className="text-sm font-semibold text-indigo-300 flex-1 text-left">
          Agent Analysis{agentName ? ` — ${agentName}` : ""}
        </span>
        {open ? <ChevronUp className="w-4 h-4 text-indigo-400" /> : <ChevronDown className="w-4 h-4 text-indigo-400" />}
      </button>
      {open && (
        <div className="px-4 pb-4 space-y-3">
          {analysis && (
            <p className="text-sm text-zinc-300 leading-relaxed">{analysis}</p>
          )}
          {recommendations && recommendations.length > 0 && (
            <div className="space-y-1.5">
              <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Recommendations</p>
              {recommendations.map((rec, i) => (
                <div key={i} className="flex gap-2 text-sm text-zinc-300">
                  <span className="text-indigo-400 shrink-0 font-mono">{i + 1}.</span>
                  <span>{rec}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ── Priority summary counts bar ───────────────────────────────────────────────
const PrioritySummaryBar = ({ summary }) => {
  if (!summary) return null;
  const counts = [
    { label: "Critical", key: "critical", color: "#ef4444" },
    { label: "High",     key: "high",     color: "#f97316" },
    { label: "Medium",   key: "medium",   color: "#f59e0b" },
    { label: "Low",      key: "low",      color: "#22c55e" },
  ];
  return (
    <div className="flex flex-wrap gap-2">
      {counts.map(c => {
        const val = summary[c.key] ?? summary[`${c.key}_suppliers`] ?? summary[`products_${c.key}`] ?? summary[`${c.key}_products`];
        if (val === undefined) return null;
        return <SummaryPill key={c.key} label={c.label} value={val} color={c.color} />;
      })}
      {summary.total_products  !== undefined && <SummaryPill label="Products"   value={summary.total_products}  color="#94a3b8" />}
      {summary.total_pos       !== undefined && <SummaryPill label="POs"        value={summary.total_pos}       color="#a855f7" />}
      {summary.total_value     !== undefined && <SummaryPill label="PO Value"   value={`$${Number(summary.total_value).toLocaleString()}`} color="#a855f7" />}
      {summary.total_cost      !== undefined && <SummaryPill label="Total Cost" value={`$${Number(summary.total_cost).toLocaleString()}`}  color="#3b82f6" />}
      {summary.total_saving_potential !== undefined && <SummaryPill label="Savings" value={`$${Number(summary.total_saving_potential).toLocaleString()}`} color="#22c55e" />}
      {summary.sustainability_score   !== undefined && <SummaryPill label="ESG Score" value={`${summary.sustainability_score}/100`} color="#22c55e" />}
    </div>
  );
};

// ── Needs Attention banner ────────────────────────────────────────────────────
const AttentionBanner = ({ items, label = "Needs Attention" }) => {
  const [open, setOpen] = useState(true);
  if (!items || items.length === 0) return null;
  return (
    <div className="rounded-xl border border-red-500/30 bg-red-500/5 overflow-hidden">
      <button onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 px-4 py-3 hover:bg-red-500/10 transition-colors">
        <AlertTriangle className="w-4 h-4 text-red-400" />
        <span className="text-sm font-bold text-red-300 flex-1 text-left">
          {label} — {items.length} item(s) require immediate action
        </span>
        {open ? <ChevronUp className="w-4 h-4 text-red-400" /> : <ChevronDown className="w-4 h-4 text-red-400" />}
      </button>
      {open && (
        <div className="px-4 pb-4 space-y-2 max-h-64 overflow-y-auto">
          {items.map((item, i) => {
            const name = item.product_name || item.supplier_name || item.warehouse_name || item.supplier || item.name || `Item ${i + 1}`;
            const priority = item.priority || "high";
            const cfg = PRIORITY_CFG[priority] || PRIORITY_CFG.high;
            const detail = item.recommendation || item.action_needed || item.trigger || item.message || "";
            return (
              <div key={i} className="flex items-start gap-3 p-3 rounded-lg"
                style={{ backgroundColor: cfg.bg, border: `1px solid ${cfg.border}` }}>
                <cfg.Icon className="w-4 h-4 mt-0.5 shrink-0" style={{ color: cfg.color }} />
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-white truncate">{name}</p>
                  {detail && <p className="text-xs text-zinc-400 mt-0.5 line-clamp-2">{detail}</p>}
                </div>
                <PriorityBadge level={priority} />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

// ── Generic collapsible result block ─────────────────────────────────────────
const ResultBlock = ({ label, children }) => {
  const [open, setOpen] = useState(true);
  return (
    <div className="border border-zinc-800 rounded-xl overflow-hidden">
      <button onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 bg-zinc-900/60 hover:bg-zinc-900 transition-colors">
        <span className="text-sm font-semibold text-zinc-200 capitalize">{label.replace(/_/g, " ")}</span>
        {open ? <ChevronUp className="w-4 h-4 text-zinc-500" /> : <ChevronDown className="w-4 h-4 text-zinc-500" />}
      </button>
      {open && <div className="px-4 py-4 max-h-96 overflow-y-auto space-y-3">{children}</div>}
    </div>
  );
};

// ── Priority list renderer ────────────────────────────────────────────────────
const PriorityList = ({ items, titleKey, subtitleKey, extraKeys = [] }) => {
  if (!items || items.length === 0) return <p className="text-xs text-zinc-500">No items</p>;
  return (
    <div className="space-y-2">
      {items.map((item, i) => {
        const priority = item.priority || "low";
        const cfg = PRIORITY_CFG[priority] || PRIORITY_CFG.low;
        return (
          <div key={i} className="p-3 rounded-lg border" style={{ backgroundColor: cfg.bg, borderColor: cfg.border }}>
            <div className="flex items-center gap-2 mb-1">
              <PriorityBadge level={priority} />
              <span className="text-sm font-semibold text-white flex-1 truncate">{item[titleKey] || `Item ${i + 1}`}</span>
            </div>
            {subtitleKey && item[subtitleKey] && <p className="text-xs text-zinc-400 mb-1">{item[subtitleKey]}</p>}
            {extraKeys.map(k => item[k] !== undefined && (
              <div key={k} className="flex gap-1 text-xs text-zinc-400">
                <span className="text-zinc-500">{k.replace(/_/g," ")}:</span>
                <span className="text-zinc-200">{typeof item[k] === "object" ? JSON.stringify(item[k]) : String(item[k])}</span>
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
};

// ── Per-workflow result renderer ──────────────────────────────────────────────
const WorkflowResults = ({ workflowId, outputs }) => {
  if (!outputs) return null;
  const { needs_attention = [], summary, agent_analysis, agent_recommendations } = outputs;

  // Helper to render agent analysis at the bottom of every workflow
  const AgentBlock = () => (
    <AgentAnalysisPanel analysis={agent_analysis} recommendations={agent_recommendations} />
  );

  if (workflowId === "wf-1") return (
    <div className="space-y-4">
      <PrioritySummaryBar summary={summary} />
      <AttentionBanner items={needs_attention} label="Stockout Risk Alert" />
      <ResultBlock label="All Products — Demand & Stockout Risk">
        <PriorityList items={outputs.all_products} titleKey="product_name" subtitleKey="recommendation"
          extraKeys={["days_of_supply","historical_avg","trend_pct","mape","highest_risk_warehouse"]} />
      </ResultBlock>
      <AgentBlock />
    </div>
  );

  if (workflowId === "wf-2") return (
    <div className="space-y-4">
      <PrioritySummaryBar summary={summary} />
      <AttentionBanner items={needs_attention} label="Reorder Urgency Alert" />
      <ResultBlock label="Product Reorder Analysis">
        <PriorityList items={outputs.product_reorder_analysis} titleKey="product_name" subtitleKey="recommendation"
          extraKeys={["reorder_point","safety_stock","economic_order_quantity","current_total_stock"]} />
      </ResultBlock>
      <ResultBlock label="Warehouse Health">
        <PriorityList items={outputs.warehouse_levels} titleKey="warehouse_name" subtitleKey="location"
          extraKeys={["capacity_utilization","health_score"]} />
      </ResultBlock>
      <AgentBlock />
    </div>
  );

  if (workflowId === "wf-3") return (
    <div className="space-y-4">
      <PrioritySummaryBar summary={summary} />
      <AttentionBanner items={outputs.all_alerts} label="Warehouse Alerts" />
      <ResultBlock label="All Warehouses">
        {(outputs.warehouses || []).map((wh, i) => {
          const cfg = PRIORITY_CFG[wh.priority] || PRIORITY_CFG.low;
          return (
            <div key={i} className="p-3 rounded-lg border space-y-2" style={{ backgroundColor: cfg.bg, borderColor: cfg.border }}>
              <div className="flex items-center gap-2">
                <PriorityBadge level={wh.priority} />
                <span className="text-sm font-bold text-white">{wh.warehouse_name}</span>
                <span className="text-xs text-zinc-400 ml-auto">{wh.location} · {wh.capacity_utilization}% capacity</span>
              </div>
              {(wh.automation_alerts || []).map((a, j) => (
                <div key={j} className="text-xs px-2 py-1 rounded" style={{ backgroundColor: PRIORITY_CFG[a.level]?.bg, color: PRIORITY_CFG[a.level]?.color }}>
                  {a.message}
                </div>
              ))}
            </div>
          );
        })}
      </ResultBlock>
    </div>
  );

  if (workflowId === "wf-4") return (
    <div className="space-y-4">
      <PrioritySummaryBar summary={summary} />
      <AttentionBanner items={needs_attention} label="High-Cost Lanes" />
      <ResultBlock label="All Supplier Routes — Sorted by Cost">
        <PriorityList items={outputs.all_routes} titleKey="supplier_name" subtitleKey="region"
          extraKeys={["recommended_mode","transit_days","cost_factor","reliability_pct"]} />
      </ResultBlock>
      <AgentBlock />
    </div>
  );

  if (workflowId === "wf-5") return (
    <div className="space-y-4">
      <PrioritySummaryBar summary={summary} />
      <AttentionBanner items={needs_attention} label="High Delay Risk Suppliers" />
      <ResultBlock label="All Suppliers — Delay Risk">
        {(outputs.all_suppliers || []).map((sup, i) => {
          const cfg = PRIORITY_CFG[sup.priority] || PRIORITY_CFG.low;
          const pred = sup.prediction || {};
          return (
            <div key={i} className="p-3 rounded-lg border space-y-2" style={{ backgroundColor: cfg.bg, borderColor: cfg.border }}>
              <div className="flex items-center gap-2 flex-wrap">
                <PriorityBadge level={sup.priority} />
                <span className="text-sm font-bold text-white">{sup.supplier_name}</span>
                <span className="text-xs text-zinc-400">{sup.region}</span>
                <span className="ml-auto text-xs font-mono text-white">{pred.delay_probability_pct}% delay probability</span>
              </div>
              <div className="flex gap-4 text-xs text-zinc-400">
                <span>Expected delay: <b className="text-white">{pred.expected_delay_days}d</b></span>
                <span>Historical: <b className="text-white">{sup.historical_analysis?.historical_delay_rate_pct}%</b></span>
              </div>
              {(sup.mitigation_actions || []).slice(0,2).map((m, j) => (
                <p key={j} className="text-xs text-zinc-400">• {m}</p>
              ))}
            </div>
          );
        })}
      </ResultBlock>
      <AgentBlock />
    </div>
  );

  if (workflowId === "wf-6") return (
    <div className="space-y-4">
      <PrioritySummaryBar summary={summary} />
      <AttentionBanner items={needs_attention} label="High-Risk Suppliers" />
      <ResultBlock label="All Suppliers — Risk Score">
        {(outputs.supplier_risks || []).map((sup, i) => {
          const cfg = PRIORITY_CFG[sup.priority] || PRIORITY_CFG.low;
          const comp = sup.component_scores || {};
          return (
            <div key={i} className="p-3 rounded-lg border space-y-2" style={{ backgroundColor: cfg.bg, borderColor: cfg.border }}>
              <div className="flex items-center gap-2 flex-wrap">
                <PriorityBadge level={sup.priority} />
                <span className="text-sm font-bold text-white">{sup.supplier_name}</span>
                <span className="text-xs text-zinc-400">{sup.country} · {sup.region}</span>
                <span className="ml-auto text-sm font-mono font-bold text-white">Score: {sup.risk_score}</span>
              </div>
              <div className="grid grid-cols-4 gap-2 text-xs text-center">
                {Object.entries(comp).map(([k, v]) => (
                  <div key={k} className="bg-zinc-900/40 rounded px-2 py-1">
                    <div className="text-zinc-500">{k}</div>
                    <div className="font-mono font-bold text-white">{v}</div>
                  </div>
                ))}
              </div>
              <p className="text-xs text-zinc-400">{sup.recommendation}</p>
            </div>
          );
        })}
      </ResultBlock>
      <AgentBlock />
    </div>
  );

  if (workflowId === "wf-7") return (
    <div className="space-y-4">
      <PrioritySummaryBar summary={summary} />
      <AttentionBanner items={needs_attention} label="Purchase Orders Requiring Approval" />
      <ResultBlock label="Selected Supplier">
        <div className="flex items-center gap-3 p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
          <CheckCircle2 className="w-5 h-5 text-emerald-400" />
          <div>
            <p className="text-sm font-bold text-white">{outputs.selected_supplier?.name} ({outputs.selected_supplier?.id})</p>
            <p className="text-xs text-zinc-400">Lowest risk score: {outputs.selected_supplier?.score}</p>
          </div>
        </div>
      </ResultBlock>
      <ResultBlock label="All Purchase Orders">
        {(outputs.purchase_orders || []).map((po, i) => {
          const cfg = PRIORITY_CFG[po.priority] || PRIORITY_CFG.low;
          return (
            <div key={i} className="p-3 rounded-lg border space-y-1" style={{ backgroundColor: cfg.bg, borderColor: cfg.border }}>
              <div className="flex items-center gap-2">
                <PriorityBadge level={po.priority} />
                <span className="text-sm font-bold text-white font-mono">{po.po_number}</span>
                <span className="ml-auto text-sm font-bold text-white">${Number(po.total_cost).toLocaleString()}</span>
              </div>
              <p className="text-xs text-zinc-300">{po.product?.name} × {po.quantity} units</p>
              <p className="text-xs text-zinc-400">Delivery: {po.expected_delivery} · Trigger: {po.trigger}</p>
            </div>
          );
        })}
      </ResultBlock>
      <AgentBlock />
    </div>
  );

  if (workflowId === "wf-8") return (
    <div className="space-y-4">
      <PrioritySummaryBar summary={summary} />
      <AttentionBanner items={needs_attention} label="High-Priority Saving Opportunities" />
      <ResultBlock label="Cost Breakdown">
        {(outputs.cost_analysis?.cost_breakdown || []).map((row, i) => (
          <div key={i} className="flex items-center justify-between py-1.5 border-b border-zinc-800/50 last:border-0">
            <span className="text-sm text-zinc-300">{row.category}</span>
            <div className="flex items-center gap-3">
              <span className="text-xs text-zinc-500">{row.share_pct}%</span>
              <span className="text-sm font-mono text-white">${Number(row.total_cost).toLocaleString()}</span>
            </div>
          </div>
        ))}
      </ResultBlock>
      <ResultBlock label="Optimization Opportunities">
        <PriorityList items={outputs.cost_analysis?.optimization_opportunities || []}
          titleKey="category" subtitleKey="action"
          extraKeys={["potential_saving_pct","estimated_saving_usd"]} />
      </ResultBlock>
      <ResultBlock label="Shipping Mode Savings by Supplier">
        <PriorityList items={outputs.shipping_comparison || []}
          titleKey="supplier" subtitleKey="region"
          extraKeys={["normal_mode","economy_mode","cost_factor_saving"]} />
      </ResultBlock>
      <AgentBlock />
    </div>
  );

  if (workflowId === "wf-9") return (
    <div className="space-y-4">
      <PrioritySummaryBar summary={summary} />
      <AttentionBanner items={needs_attention} label="Products with Critical Production Gaps" />
      <ResultBlock label="All Products — 8-Week Production Plans">
        {(outputs.all_products || []).map((plan, i) => {
          const cfg = PRIORITY_CFG[plan.priority] || PRIORITY_CFG.low;
          const s = plan.summary || {};
          return (
            <div key={i} className="p-3 rounded-lg border space-y-2" style={{ backgroundColor: cfg.bg, borderColor: cfg.border }}>
              <div className="flex items-center gap-2 flex-wrap">
                <PriorityBadge level={plan.priority} />
                <span className="text-sm font-bold text-white">{plan.product_name}</span>
                <span className="ml-auto text-xs text-zinc-400">{plan.critical_weeks} critical wk(s)</span>
              </div>
              <div className="flex gap-4 text-xs text-zinc-400">
                <span>Plan: <b className="text-white">{s.total_planned_production?.toLocaleString()}</b></span>
                <span>Demand: <b className="text-white">{s.total_forecasted_demand?.toLocaleString()}</b></span>
                <span>Util: <b className="text-white">{s.avg_weekly_utilization_pct}%</b></span>
              </div>
              {(plan.alerts || []).slice(0, 2).map((a, j) => (
                <p key={j} className="text-xs text-amber-400">⚠ {a}</p>
              ))}
            </div>
          );
        })}
      </ResultBlock>
      <AgentBlock />
    </div>
  );

  if (workflowId === "wf-10") return (
    <div className="space-y-4">
      <PrioritySummaryBar summary={summary} />
      <AttentionBanner items={needs_attention} label="Non-Compliant Suppliers & High-Impact Actions" />
      <ResultBlock label="Supplier ESG Compliance">
        <PriorityList items={outputs.supplier_esg_compliance || []}
          titleKey="supplier_name" subtitleKey="action_needed"
          extraKeys={["region","iso_14001"]} />
      </ResultBlock>
      <ResultBlock label="Green Action Plan — Sorted by Impact">
        <PriorityList items={outputs.sustainability_overview?.green_action_plan || []}
          titleKey="action" extraKeys={["co2_reduction_pct","cost_impact"]} />
      </ResultBlock>
      <ResultBlock label="ESG News">
        {(outputs.esg_news?.news_items || []).slice(0, 4).map((n, i) => (
          <div key={i} className="flex gap-2 text-xs border-b border-zinc-800/50 pb-2 last:border-0">
            <span className={`shrink-0 font-semibold ${n.sentiment === "negative" ? "text-red-400" : "text-emerald-400"}`}>{n.sentiment}</span>
            <span className="text-zinc-300">{n.headline}</span>
          </div>
        ))}
      </ResultBlock>
      <AgentBlock />
    </div>
  );

  return <p className="text-zinc-500 text-sm">No renderer for this workflow.</p>;
};

// ── Workflow Card ─────────────────────────────────────────────────────────────
const WorkflowCard = ({ workflow, isSelected, onClick }) => {
  const Icon  = WORKFLOW_ICONS[workflow.name] || Package;
  const color = CATEGORY_COLORS[workflow.category] || "#6366f1";
  return (
    <div className={`workflow-card ${isSelected ? "active" : ""}`} onClick={onClick}
      data-testid={`workflow-card-${workflow.id}`}>
      <div className="flex items-start justify-between mb-3">
        <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ backgroundColor: `${color}15` }}>
          <Icon className="w-5 h-5" style={{ color }} />
        </div>
        <Badge variant="outline" className="text-xs capitalize" style={{ borderColor: color, color }}>{workflow.category}</Badge>
      </div>
      <h3 className="text-base font-semibold text-white mb-1">{workflow.name}</h3>
      <p className="text-sm text-zinc-500 mb-3 line-clamp-2">{workflow.description}</p>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1">
          {(workflow.agents_involved || []).map((agent, idx) => (
            <div key={idx} className="w-6 h-6 rounded-full bg-zinc-800 border border-zinc-700 flex items-center justify-center" title={agent}>
              <Bot className="w-3 h-3 text-zinc-400" />
            </div>
          ))}
        </div>
        <span className="text-xs text-zinc-600 capitalize">{TYPE_LABELS[workflow.workflow_type]}</span>
      </div>
    </div>
  );
};

// ── Workflow Detail Panel ─────────────────────────────────────────────────────
const WorkflowDetail = ({ workflow }) => {
  const [running, setRunning]     = useState(false);
  const [runResult, setRunResult] = useState(null);
  const [steps, setSteps]         = useState([]);
  const [error, setError]         = useState(null);

  useEffect(() => { setRunResult(null); setSteps([]); setError(null); }, [workflow?.id]);

  if (!workflow) return (
    <div className="flex items-center justify-center h-full text-zinc-500">Select a workflow to view details</div>
  );

  const Icon  = WORKFLOW_ICONS[workflow.name] || Package;
  const color = CATEGORY_COLORS[workflow.category] || "#6366f1";

  const handleRun = async () => {
    setRunning(true); setRunResult(null); setError(null);
    setSteps([{ step: "Connecting to agents…", status: "running" }]);
    try {
      const { data } = await axios.post(`${API}/workflows/${workflow.id}/run`, { params: {} });
      setSteps((data.steps || []).map(s => ({ ...s, status: "complete" })));
      setRunResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Workflow execution failed");
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="space-y-5" data-testid="workflow-detail">
      {/* Header */}
      <div className="flex items-start gap-4">
        <div className="w-14 h-14 rounded-xl flex items-center justify-center shrink-0" style={{ backgroundColor: `${color}15` }}>
          <Icon className="w-7 h-7" style={{ color }} />
        </div>
        <div className="flex-1 min-w-0">
          <h2 className="text-2xl font-bold text-white">{workflow.name}</h2>
          <p className="text-zinc-400 mt-1 text-sm">{workflow.description}</p>
          <div className="flex gap-2 mt-3 flex-wrap">
            <Badge style={{ backgroundColor: `${color}20`, color }}>{workflow.category}</Badge>
            <Badge variant="outline">{TYPE_LABELS[workflow.workflow_type]}</Badge>
          </div>
        </div>
        <Button onClick={handleRun} disabled={running} className="shrink-0 gap-2"
          style={{ backgroundColor: color, color: "#fff", border: "none" }}>
          {running ? <><Loader2 className="w-4 h-4 animate-spin" />Running…</> : <><Play className="w-4 h-4" />Run Workflow</>}
        </Button>
      </div>

      {/* Static metrics */}
      {workflow.metrics && (
        <div className="grid grid-cols-2 gap-3">
          {Object.entries(workflow.metrics).map(([k, v]) => (
            <div key={k} className="p-3 bg-zinc-900/50 rounded-lg border border-zinc-800/50">
              <p className="text-xs text-zinc-500 uppercase tracking-wider mb-1">{k.replace(/_/g, " ")}</p>
              <p className="text-lg font-mono font-bold text-white">
                {typeof v === "number" && v < 1 ? `${(v * 100).toFixed(1)}%` : v.toLocaleString?.() ?? v}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Agents + tools */}
      <div className="flex flex-wrap gap-2">
        {(workflow.agents_involved || []).map((a, i) => (
          <div key={i} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800">
            <Bot className="w-4 h-4 text-indigo-400" /><span className="text-sm text-zinc-300 capitalize">{a}</span>
          </div>
        ))}
        {(workflow.tools || []).map((t, i) => (
          <div key={i} className="flex items-center gap-1 px-2.5 py-1 rounded-md bg-zinc-900 border border-zinc-800">
            <Zap className="w-3 h-3 text-amber-400" /><span className="text-xs text-zinc-400 font-mono">{t}</span>
          </div>
        ))}
      </div>

      {/* Execution log */}
      {(running || steps.length > 0) && (
        <div className="bg-zinc-900/40 rounded-xl border border-zinc-800/50 p-4 space-y-1">
          <p className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">Execution Log</p>
          {steps.map((s, i) => (
            <div key={i} className={`flex items-center gap-3 py-1.5 px-2 rounded ${s.status === "running" ? "bg-indigo-500/10" : ""}`}>
              {s.status === "complete"
                ? <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                : <Loader2 className="w-4 h-4 text-indigo-400 animate-spin shrink-0" />}
              <span className={`text-sm ${s.status === "complete" ? "text-zinc-400" : "text-white"}`}>{s.step}</span>
            </div>
          ))}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-sm text-red-400">
          <XCircle className="w-4 h-4 shrink-0" />{error}
        </div>
      )}

      {/* Results */}
      {runResult?.status === "completed" && (
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-5 h-5 text-emerald-400" />
            <h3 className="text-sm font-semibold text-zinc-300">Results — Priority Ranked</h3>
            <span className="text-xs text-zinc-500 ml-auto">{new Date(runResult.completed_at).toLocaleTimeString()}</span>
          </div>
          <WorkflowResults workflowId={workflow.id} outputs={runResult.outputs} />
        </div>
      )}
    </div>
  );
};

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function Workflows() {
  const [workflows, setWorkflows] = useState([]);
  const [selected, setSelected]   = useState(null);
  const [loading, setLoading]     = useState(true);

  useEffect(() => {
    axios.get(`${API}/workflows`).then(r => {
      setWorkflows(r.data.workflows);
      if (r.data.workflows.length > 0) setSelected(r.data.workflows[0]);
    }).catch(console.error).finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="flex items-center justify-center h-96" data-testid="workflows-loading">
      <div className="text-zinc-500">Loading workflows…</div>
    </div>
  );

  return (
    <div className="space-y-6" data-testid="workflows-page">
      <div>
        <h1 className="text-3xl font-bold text-white tracking-tight">Workflows</h1>
        <p className="text-zinc-500 mt-1">
          10 workflows across all products & suppliers — sorted{" "}
          <span className="text-red-400 font-medium">Critical → High → Medium → Low</span>.
          Agent LLM analysis included after each run.
        </p>
      </div>
      <div className="grid grid-cols-12 gap-6">
        <div className="col-span-5 space-y-3" data-testid="workflow-list">
          {workflows.map(wf => (
            <WorkflowCard key={wf.id} workflow={wf}
              isSelected={selected?.id === wf.id} onClick={() => setSelected(wf)} />
          ))}
        </div>
        <div className="col-span-7 bg-zinc-900/30 rounded-xl border border-zinc-800/50 p-6 overflow-y-auto max-h-[88vh]">
          <WorkflowDetail workflow={selected} />
        </div>
      </div>
    </div>
  );
}
