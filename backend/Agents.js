import { useState, useEffect, useRef } from "react";
import axios from "axios";
import {
  Bot, Cpu, Package, AlertTriangle, Zap, Activity, MessageSquare,
  ArrowRight, Send, Loader2, CheckCircle2, XCircle, AlertCircle,
  Sparkles, TrendingUp, DollarSign, Factory, Leaf, Clock,
  ChevronDown, ChevronUp, Calendar, BarChart2, Shield,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// ── Agent config (all 8 real agents with real tools) ─────────────────────────
const AGENT_CONFIG = {
  orchestrator: {
    name: "Orchestrator", color: "#f8fafc", bg: "rgba(248,250,252,0.08)",
    Icon: Cpu,
    tools: ["coordinate_agents","decompose_task","route_to_agents"],
    description: "Routes queries, coordinates all agents, synthesises results"
  },
  demand: {
    name: "Demand Agent", color: "#a855f7", bg: "rgba(168,85,247,0.08)",
    Icon: TrendingUp,
    tools: ["forecast_demand","analyze_stockout_risk"],
    description: "Demand forecasting, stockout prediction, seasonality analysis"
  },
  inventory: {
    name: "Inventory Agent", color: "#06b6d4", bg: "rgba(6,182,212,0.08)",
    Icon: Package,
    tools: ["calculate_reorder_point","check_warehouse_levels"],
    description: "EOQ optimisation, safety stock, warehouse health"
  },
  supplier: {
    name: "Supplier Agent", color: "#f97316", bg: "rgba(249,115,22,0.08)",
    Icon: AlertTriangle,
    tools: ["score_supplier_risk","get_shipping_options","get_market_intelligence","predict_shipment_delay"],
    description: "Risk scoring, delay prediction, route optimisation, news analysis"
  },
  action: {
    name: "Action Agent", color: "#3b82f6", bg: "rgba(59,130,246,0.08)",
    Icon: Zap,
    tools: ["generate_purchase_order"],
    description: "Executes POs, raises alerts, logs decisions"
  },
  logistics: {
    name: "Logistics Agent", color: "#22c55e", bg: "rgba(34,197,94,0.08)",
    Icon: DollarSign,
    tools: ["optimize_logistics_cost"],
    description: "Cost analysis, freight mode optimisation, savings identification"
  },
  planning: {
    name: "Planning Agent", color: "#eab308", bg: "rgba(234,179,8,0.08)",
    Icon: Factory,
    tools: ["generate_production_plan"],
    description: "8-week production scheduling, capacity planning"
  },
  sustainability: {
    name: "Sustainability Agent", color: "#10b981", bg: "rgba(16,185,129,0.08)",
    Icon: Leaf,
    tools: ["track_sustainability"],
    description: "Carbon footprint, ESG compliance, green action planning"
  },
};

const PRIORITY_CFG = {
  critical: { color: "#ef4444", bg: "rgba(239,68,68,0.1)",  border: "rgba(239,68,68,0.25)",  Icon: XCircle,      label: "Critical" },
  high:     { color: "#f97316", bg: "rgba(249,115,22,0.1)", border: "rgba(249,115,22,0.25)", Icon: AlertTriangle, label: "High" },
  medium:   { color: "#f59e0b", bg: "rgba(245,158,11,0.1)", border: "rgba(245,158,11,0.25)", Icon: AlertCircle,  label: "Medium" },
  low:      { color: "#22c55e", bg: "rgba(34,197,94,0.1)",  border: "rgba(34,197,94,0.25)",  Icon: CheckCircle2, label: "Low" },
};

// ── Suggested prompts ─────────────────────────────────────────────────────────
const SUGGESTED_PROMPTS = [
  { label: "What needs my attention today?",        icon: Sparkles,  color: "#ef4444" },
  { label: "Give me a 7-day action plan",           icon: Calendar,  color: "#3b82f6" },
  { label: "Which suppliers are at risk this week?",icon: Shield,    color: "#f97316" },
  { label: "Where can I cut logistics costs?",      icon: DollarSign,color: "#22c55e" },
  { label: "Any inventory emergencies?",            icon: Package,   color: "#06b6d4" },
  { label: "ESG and sustainability status",         icon: Leaf,      color: "#10b981" },
  { label: "Production gaps in the next 8 weeks?",  icon: Factory,   color: "#eab308" },
  { label: "Which products face stockout risk?",    icon: AlertTriangle, color: "#a855f7" },
];

// ── Tiny components ───────────────────────────────────────────────────────────
const PriorityBadge = ({ level }) => {
  const cfg = PRIORITY_CFG[level] || PRIORITY_CFG.low;
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold"
      style={{ backgroundColor: cfg.bg, color: cfg.color, border: `1px solid ${cfg.border}` }}>
      <cfg.Icon className="w-3 h-3" />{cfg.label}
    </span>
  );
};

const AgentChip = ({ name }) => {
  const cfg = AGENT_CONFIG[name];
  if (!cfg) return null;
  const { Icon } = cfg;
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs"
      style={{ backgroundColor: cfg.bg, color: cfg.color, border: `1px solid ${cfg.color}30` }}>
      <Icon className="w-3 h-3" />{cfg.name}
    </span>
  );
};

// ── Agent network node ────────────────────────────────────────────────────────
const AgentNode = ({ agentKey, state, isActive }) => {
  const cfg = AGENT_CONFIG[agentKey];
  if (!cfg) return null;
  const { Icon } = cfg;
  const status = state?.status || "idle";
  const active = status !== "idle";
  return (
    <div className={`relative p-3 rounded-xl border transition-all duration-500 ${active ? "border-white/20" : "border-zinc-800/60"}`}
      style={{ backgroundColor: active ? cfg.bg : "rgba(24,24,27,0.4)", boxShadow: active ? `0 0 16px ${cfg.color}25` : "none" }}>
      {active && (
        <div className="absolute inset-0 rounded-xl animate-pulse"
          style={{ border: `1px solid ${cfg.color}40` }} />
      )}
      <div className="flex items-center gap-2 mb-1">
        <div className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0"
          style={{ backgroundColor: cfg.bg }}>
          <Icon className="w-4 h-4" style={{ color: cfg.color }} />
        </div>
        <div className="min-w-0">
          <p className="text-xs font-semibold text-white truncate">{cfg.name}</p>
          <div className="flex items-center gap-1">
            <div className={`w-1.5 h-1.5 rounded-full ${active ? "animate-pulse" : ""}`}
              style={{ backgroundColor: active ? cfg.color : "#52525b" }} />
            <span className="text-xs text-zinc-500 capitalize">{status}</span>
          </div>
        </div>
      </div>
      {state?.current_task && (
        <p className="text-xs text-zinc-500 bg-zinc-900/60 rounded px-1.5 py-0.5 truncate">{state.current_task}</p>
      )}
    </div>
  );
};

// ── Priority action card ──────────────────────────────────────────────────────
const ActionCard = ({ action, index }) => {
  const [open, setOpen] = useState(index < 3);
  const cfg = PRIORITY_CFG[action.priority] || PRIORITY_CFG.medium;
  return (
    <div className="rounded-xl overflow-hidden border transition-all"
      style={{ backgroundColor: cfg.bg, borderColor: cfg.border }}>
      <button onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-3 px-4 py-3 text-left hover:brightness-110 transition-all">
        <span className="text-base font-bold shrink-0" style={{ color: cfg.color }}>#{index + 1}</span>
        <PriorityBadge level={action.priority} />
        <span className="text-sm font-semibold text-white flex-1 truncate">{action.action}</span>
        <AgentChip name={action.agent} />
        {open ? <ChevronUp className="w-4 h-4 text-zinc-500 shrink-0" /> : <ChevronDown className="w-4 h-4 text-zinc-500 shrink-0" />}
      </button>
      {open && (
        <div className="px-4 pb-3 space-y-1.5">
          {action.detail && <p className="text-sm text-zinc-300">{action.detail}</p>}
          {action.estimated_impact && (
            <p className="text-xs text-zinc-500">
              <span className="text-zinc-400 font-medium">Impact: </span>{action.estimated_impact}
            </p>
          )}
        </div>
      )}
    </div>
  );
};

// ── 7-day plan ────────────────────────────────────────────────────────────────
const SevenDayPlan = ({ plan }) => {
  if (!plan || plan.length === 0) return null;
  const dayColors = ["#ef4444","#f97316","#f59e0b","#22c55e"];
  return (
    <div className="space-y-3">
      <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider flex items-center gap-2">
        <Calendar className="w-4 h-4" />7-Day Action Plan
      </h4>
      <div className="grid grid-cols-2 gap-2">
        {plan.map((day, i) => (
          <div key={i} className="p-3 rounded-lg bg-zinc-900/50 border border-zinc-800/50">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: dayColors[i] || "#6366f1" }} />
              <span className="text-xs font-bold text-white">{day.day}</span>
              <span className="text-xs text-zinc-500 truncate">{day.focus}</span>
            </div>
            <ul className="space-y-1">
              {(day.actions || []).slice(0, 2).map((action, j) => (
                <li key={j} className="text-xs text-zinc-400 flex gap-1.5">
                  <span className="text-zinc-600 shrink-0">•</span>{action}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
};

// ── Key metrics row ───────────────────────────────────────────────────────────
const KeyMetrics = ({ metrics }) => {
  if (!metrics) return null;
  const items = [
    { label: "Critical Items",     value: metrics.critical_items,      color: "#ef4444", icon: XCircle },
    { label: "High-Risk Suppliers",value: metrics.high_risk_suppliers,  color: "#f97316", icon: AlertTriangle },
    { label: "Savings Identified", value: metrics.logistics_savings,    color: "#22c55e", icon: DollarSign },
    { label: "Domains Covered",    value: metrics.domains_covered,      color: "#3b82f6", icon: BarChart2 },
    { label: "Tools Invoked",      value: metrics.tools_invoked,        color: "#a855f7", icon: Zap },
  ];
  return (
    <div className="flex flex-wrap gap-2">
      {items.map((m, i) => {
        const { icon: Icon } = m;
        return (
          <div key={i} className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-900/60 border border-zinc-800/50">
            <Icon className="w-3.5 h-3.5" style={{ color: m.color }} />
            <span className="text-xs text-zinc-400">{m.label}:</span>
            <span className="text-xs font-bold text-white">{m.value}</span>
          </div>
        );
      })}
    </div>
  );
};

// ── Full AI response renderer ─────────────────────────────────────────────────
const OrchestratorRouting = ({ routing }) => {
  if (!routing) return null;
  const urgencyColor = { immediate: "#ef4444", today: "#f97316", week: "#3b82f6" };
  const color = urgencyColor[routing.urgency] || "#94a3b8";
  return (
    <div className="space-y-2 p-3 rounded-lg bg-zinc-900/60 border border-zinc-800/40">
      {/* Row 1: label + agent chips + urgency badge */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1.5 shrink-0">
          <Cpu className="w-3.5 h-3.5 text-zinc-400" />
          <span className="text-xs font-semibold text-zinc-400">Orchestrator routed to:</span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {(routing.agents_selected || []).map((a, i) => <AgentChip key={i} name={a} />)}
        </div>
        <span className="ml-auto text-xs font-bold px-2 py-0.5 rounded-full whitespace-nowrap"
          style={{ color, backgroundColor: `${color}15`, border: `1px solid ${color}30` }}>
          {routing.urgency}
        </span>
      </div>
      {/* Row 2: reasoning on its own line */}
      {routing.reasoning && (
        <p className="text-xs text-zinc-500 leading-relaxed border-t border-zinc-800/60 pt-2">
          {routing.reasoning}
        </p>
      )}
    </div>
  );
};

const AIResponse = ({ response }) => {
  return (
    <div className="space-y-5">
      {/* Orchestrator routing decision — shown first */}
      {response.orchestrator_routing && (
        <OrchestratorRouting routing={response.orchestrator_routing} />
      )}

      {/* Executive summary */}
      {response.executive_summary && (
        <div className="p-4 rounded-xl bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border border-indigo-500/20">
          <p className="text-sm font-semibold text-zinc-200 leading-relaxed">{response.executive_summary}</p>
        </div>
      )}

      {/* Key metrics */}
      <KeyMetrics metrics={response.key_metrics} />

      {/* Priority actions */}
      {response.priority_actions?.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider flex items-center gap-2">
            <AlertTriangle className="w-4 h-4" />Priority Actions ({response.priority_actions.length})
          </h4>
          <div className="space-y-2">
            {response.priority_actions.map((action, i) => (
              <ActionCard key={i} action={action} index={i} />
            ))}
          </div>
        </div>
      )}

      {/* 7-day plan */}
      {response.seven_day_plan?.length > 0 && (
        <SevenDayPlan plan={response.seven_day_plan} />
      )}

      {/* Agent insights (collapsed) */}
      {response.agent_insights && Object.keys(response.agent_insights).length > 0 && (
        <AgentInsights insights={response.agent_insights} />
      )}
    </div>
  );
};

const AgentInsights = ({ insights }) => {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-zinc-800 rounded-xl overflow-hidden">
      <button onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 bg-zinc-900/50 hover:bg-zinc-900 transition-colors">
        <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Agent Reasoning</span>
        {open ? <ChevronUp className="w-4 h-4 text-zinc-500" /> : <ChevronDown className="w-4 h-4 text-zinc-500" />}
      </button>
      {open && (
        <div className="p-4 space-y-3 max-h-64 overflow-y-auto">
          {Object.entries(insights).map(([agent, insight], i) => {
            const cfg = AGENT_CONFIG[agent];
            return (
              <div key={i} className="space-y-1">
                <div className="flex items-center gap-2">
                  {cfg && <cfg.Icon className="w-3.5 h-3.5" style={{ color: cfg.color }} />}
                  <span className="text-xs font-semibold text-zinc-300">{cfg?.name || agent}</span>
                  <span className="text-xs text-zinc-600 ml-auto">confidence: {Math.round((insight.confidence || 0.8) * 100)}%</span>
                </div>
                <p className="text-xs text-zinc-400 pl-5">{insight.analysis}</p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

// ── Chat message ──────────────────────────────────────────────────────────────
const ChatMessage = ({ message }) => {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-lg px-4 py-3 rounded-2xl rounded-tr-sm bg-indigo-600/80 text-white text-sm">
          {message.content}
        </div>
      </div>
    );
  }
  if (message.role === "loading") {
    return (
      <div className="flex items-start gap-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shrink-0">
          <Sparkles className="w-4 h-4 text-white" />
        </div>
        <div className="flex-1 pt-1">
          <div className="flex items-center gap-2 mb-3">
            <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
            <span className="text-sm text-zinc-400">Activating agents and gathering real data…</span>
          </div>
          {message.activeAgents?.map((a, i) => {
            const agentCfg = AGENT_CONFIG[a];
            const AgentIcon = agentCfg?.Icon;
            return (
              <div key={i} className="flex items-center gap-2 text-xs text-zinc-500 mb-1">
                <div className="w-4 h-4 rounded flex items-center justify-center"
                  style={{ backgroundColor: agentCfg?.bg }}>
                  {AgentIcon && <AgentIcon className="w-2.5 h-2.5" style={{ color: agentCfg?.color }} />}
                </div>
                <span>{agentCfg?.name || a}</span>
                <Loader2 className="w-3 h-3 animate-spin text-zinc-600" />
              </div>
            );
          })}
        </div>
      </div>
    );
  }
  return (
    <div className="flex items-start gap-3">
      <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shrink-0 mt-0.5">
        <Sparkles className="w-4 h-4 text-white" />
      </div>
      <div className="flex-1 min-w-0">
        {message.error ? (
          <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-sm text-red-400">{message.error}</div>
        ) : (
          <AIResponse response={message.content} />
        )}
        <p className="text-xs text-zinc-600 mt-2">{new Date(message.timestamp).toLocaleTimeString()}</p>
      </div>
    </div>
  );
};

// ── Message bus ───────────────────────────────────────────────────────────────
const MSG_TYPE_CFG = {
  task_delegation:   { color: "#6366f1", label: "delegate",  dot: "bg-indigo-400" },
  analysis_complete: { color: "#22c55e", label: "analysis",  dot: "bg-emerald-400" },
  synthesis_complete:{ color: "#a855f7", label: "synthesis", dot: "bg-purple-400" },
  query_delegation:  { color: "#6366f1", label: "query",     dot: "bg-indigo-400" },
};
const getTypeCfg = (msgType) => {
  if (msgType?.startsWith("tool_call:")) return { color: "#f59e0b", label: msgType.replace("tool_call:",""), dot: "bg-amber-400" };
  return MSG_TYPE_CFG[msgType] || { color: "#71717a", label: msgType || "msg", dot: "bg-zinc-600" };
};

const MessageBus = ({ messages }) => (
  <div className="space-y-1.5 max-h-52 overflow-y-auto">
    {messages.length === 0 ? (
      <div className="text-center py-4">
        <MessageSquare className="w-6 h-6 mx-auto mb-1 text-zinc-700" />
        <p className="text-xs text-zinc-600">No messages yet — ask a question</p>
      </div>
    ) : (
      messages.slice(-30).reverse().map((msg, i) => {
        const fromCfg = AGENT_CONFIG[msg.from_agent];
        const toCfg   = AGENT_CONFIG[msg.to_agent];
        const typeCfg = getTypeCfg(msg.message_type);
        const isTool  = msg.message_type?.startsWith("tool_call:");
        return (
          <div key={msg.id || i}
            className={`flex items-center gap-1.5 text-xs rounded-lg px-2 py-1.5 ${isTool ? "bg-amber-500/5 border border-amber-500/10" : "bg-zinc-900/40"}`}>
            {/* from */}
            <div className="w-5 h-5 rounded flex items-center justify-center shrink-0"
              style={{ backgroundColor: fromCfg?.bg || "rgba(39,39,42,0.5)" }}>
              {fromCfg && (() => { const I = fromCfg.Icon; return <I className="w-3 h-3" style={{ color: fromCfg.color }} />; })()}
            </div>
            {/* arrow */}
            <ArrowRight className="w-3 h-3 text-zinc-700 shrink-0" />
            {/* to */}
            {!isTool && (
              <div className="w-5 h-5 rounded flex items-center justify-center shrink-0"
                style={{ backgroundColor: toCfg?.bg || "rgba(39,39,42,0.5)" }}>
                {toCfg && (() => { const I = toCfg.Icon; return <I className="w-3 h-3" style={{ color: toCfg.color }} />; })()}
              </div>
            )}
            {/* type badge */}
            <div className={`shrink-0 flex items-center gap-1`}>
              <div className={`w-1.5 h-1.5 rounded-full ${typeCfg.dot}`} />
              <span className="font-medium" style={{ color: typeCfg.color }}>{typeCfg.label}</span>
            </div>
            {/* content */}
            <span className="text-zinc-500 truncate flex-1">{msg.content?.substring(0, 60)}</span>
            {/* time */}
            <span className="text-zinc-700 shrink-0">{new Date(msg.timestamp).toLocaleTimeString()}</span>
          </div>
        );
      })
    )}
  </div>
);

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function Agents({ agentStates }) {
  const [chatHistory, setChatHistory]   = useState([]);
  const [inputValue, setInputValue]     = useState("");
  const [loading, setLoading]           = useState(false);
  const [messages, setMessages]         = useState([]);
  const [llmActive, setLlmActive]       = useState(false);
  const chatEndRef = useRef(null);
  const inputRef   = useRef(null);

  // Fetch messages + LLM status
  useEffect(() => {
    const fetch = async () => {
      try {
        const [msgRes, healthRes] = await Promise.all([
          axios.get(`${API}/agents/messages`),
          axios.get(`${API}/health`)
        ]);
        setMessages(msgRes.data.messages || []);
        setLlmActive(healthRes.data.ollama_connected || false);
      } catch {}
    };
    fetch();
    const iv = setInterval(fetch, 3000);
    return () => clearInterval(iv);
  }, []);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  const handleQuery = async (query) => {
    if (!query.trim() || loading) return;
    setInputValue("");

    // Determine likely agents from query
    const q = query.toLowerCase();
    const likelyAgents = [];
    if (["today","priority","urgent","week","plan","morning","focus"].some(kw => q.includes(kw)))
      likelyAgents.push("orchestrator","demand","inventory","supplier","logistics");
    else {
      if (["demand","forecast","stockout"].some(kw => q.includes(kw))) likelyAgents.push("demand");
      if (["inventory","stock","warehouse"].some(kw => q.includes(kw))) likelyAgents.push("inventory");
      if (["supplier","vendor","risk","delay"].some(kw => q.includes(kw))) likelyAgents.push("supplier");
      if (["cost","logistics","shipping","save"].some(kw => q.includes(kw))) likelyAgents.push("logistics");
      if (["production","plan","schedule"].some(kw => q.includes(kw))) likelyAgents.push("planning");
      if (["sustainability","carbon","esg"].some(kw => q.includes(kw))) likelyAgents.push("sustainability");
      if (!likelyAgents.length) likelyAgents.push("orchestrator","demand","supplier");
    }

    // Add user message + loading state
    const userMsg = { role: "user", content: query, timestamp: new Date().toISOString() };
    const loadMsg = { role: "loading", activeAgents: likelyAgents, timestamp: new Date().toISOString() };
    setChatHistory(prev => [...prev, userMsg, loadMsg]);
    setLoading(true);

    try {
      const { data } = await axios.post(`${API}/business/query`, { query });
      const aiMsg = { role: "assistant", content: data, timestamp: new Date().toISOString() };
      setChatHistory(prev => prev.filter(m => m.role !== "loading").concat(aiMsg));
    } catch (err) {
      const errMsg = { role: "assistant", error: err.response?.data?.detail || err.message || "Query failed", timestamp: new Date().toISOString() };
      setChatHistory(prev => prev.filter(m => m.role !== "loading").concat(errMsg));
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleQuery(inputValue); }
  };

  const activeAgentKeys = Object.entries(agentStates || {})
    .filter(([, s]) => s?.status !== "idle")
    .map(([k]) => k);

  return (
    <div className="flex gap-6 h-[calc(100vh-120px)]" data-testid="agents-page">

      {/* ── LEFT: Business Command Center ──────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
              <Sparkles className="w-6 h-6 text-indigo-400" />
              Business Command Center
            </h1>
            <p className="text-zinc-500 text-sm mt-0.5">
              Ask your supply chain AI anything — backed by {Object.keys(AGENT_CONFIG).length} agents and 12 live tools
            </p>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border"
            style={{ borderColor: llmActive ? "rgba(34,197,94,0.3)" : "rgba(245,158,11,0.3)",
                     backgroundColor: llmActive ? "rgba(34,197,94,0.08)" : "rgba(245,158,11,0.08)" }}>
            <div className={`w-2 h-2 rounded-full ${llmActive ? "bg-emerald-400 animate-pulse" : "bg-amber-400"}`} />
            <span className="text-xs font-medium" style={{ color: llmActive ? "#4ade80" : "#fbbf24" }}>
              {llmActive ? "LLM Online" : "Mock Mode"}
            </span>
          </div>
        </div>

        {/* Chat area */}
        <div className="flex-1 min-h-0 bg-zinc-900/30 rounded-xl border border-zinc-800/50 flex flex-col overflow-hidden">
          <ScrollArea className="flex-1 p-5">
            {chatHistory.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full py-12 space-y-6">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                  <Sparkles className="w-8 h-8 text-white" />
                </div>
                <div className="text-center">
                  <h2 className="text-lg font-bold text-white mb-1">Good morning, Supply Chain Leader</h2>
                  <p className="text-zinc-500 text-sm max-w-md">
                    Ask me anything about your supply chain. I'll activate the right agents,
                    analyse real data across all products and suppliers, and give you priority-ranked actions.
                  </p>
                </div>
                <div className="grid grid-cols-2 gap-2 w-full max-w-lg">
                  {SUGGESTED_PROMPTS.map((prompt, i) => {
                    const { icon: Icon } = prompt;
                    return (
                      <button key={i} onClick={() => handleQuery(prompt.label)}
                        className="flex items-center gap-2 px-3 py-2.5 rounded-xl border border-zinc-800 bg-zinc-900/40 hover:border-zinc-700 hover:bg-zinc-900 transition-all text-left group">
                        <Icon className="w-4 h-4 shrink-0" style={{ color: prompt.color }} />
                        <span className="text-xs text-zinc-400 group-hover:text-zinc-200 transition-colors">{prompt.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                {chatHistory.map((msg, i) => <ChatMessage key={i} message={msg} />)}
                <div ref={chatEndRef} />
              </div>
            )}
          </ScrollArea>

          {/* Input */}
          <div className="border-t border-zinc-800/50 p-4">
            <div className="flex gap-3 items-end">
              <textarea
                ref={inputRef}
                value={inputValue}
                onChange={e => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask anything: 'What should I focus on today?' or 'Which suppliers are at risk?'"
                rows={1}
                disabled={loading}
                className="flex-1 bg-zinc-900 border border-zinc-700 rounded-xl px-4 py-3 text-sm text-white placeholder-zinc-500 resize-none focus:outline-none focus:border-indigo-500/50 transition-colors disabled:opacity-50"
                style={{ minHeight: "48px", maxHeight: "120px" }}
              />
              <button
                onClick={() => handleQuery(inputValue)}
                disabled={loading || !inputValue.trim()}
                className="w-12 h-12 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-zinc-800 disabled:cursor-not-allowed flex items-center justify-center transition-colors shrink-0">
                {loading ? <Loader2 className="w-5 h-5 text-white animate-spin" /> : <Send className="w-5 h-5 text-white" />}
              </button>
            </div>
            <p className="text-xs text-zinc-600 mt-2">
              Press Enter to send · Agents call real tools on all {" "}
              <span className="text-zinc-500">10 products, 8 suppliers, 4 warehouses</span>
            </p>
          </div>
        </div>
      </div>

      {/* ── RIGHT: Agent Network + Message Bus ─────────────────────────────── */}
      <div className="w-80 shrink-0 flex flex-col gap-4">
        {/* Agent network */}
        <div className="bg-zinc-900/30 rounded-xl border border-zinc-800/50 p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Agent Network</h3>
            {activeAgentKeys.length > 0 && (
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-xs text-emerald-400">{activeAgentKeys.length} active</span>
              </div>
            )}
          </div>
          {/* Orchestrator */}
          <div className="mb-3">
            <AgentNode agentKey="orchestrator" state={agentStates?.orchestrator} />
          </div>
          {/* Connection line */}
          <div className="flex justify-center mb-3">
            <div className="w-px h-4 bg-gradient-to-b from-zinc-500 to-transparent" />
          </div>
          {/* All 7 specialist agents */}
          <div className="grid grid-cols-2 gap-2">
            {["demand","inventory","supplier","action","logistics","planning","sustainability"].map(key => (
              <AgentNode key={key} agentKey={key} state={agentStates?.[key]} />
            ))}
          </div>
        </div>

        {/* Tool registry */}
        <div className="bg-zinc-900/30 rounded-xl border border-zinc-800/50 p-4">
          <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">12 Active Tools</h3>
          <div className="space-y-1.5 max-h-40 overflow-y-auto">
            {Object.entries(AGENT_CONFIG).filter(([k]) => k !== "orchestrator").map(([agentKey, cfg]) =>
              cfg.tools.map((tool, i) => (
                <div key={`${agentKey}-${i}`} className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded flex items-center justify-center shrink-0"
                    style={{ backgroundColor: cfg.bg }}>
                    <cfg.Icon className="w-2.5 h-2.5" style={{ color: cfg.color }} />
                  </div>
                  <span className="text-xs text-zinc-500 font-mono truncate">{tool}()</span>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Message bus */}
        <div className="bg-zinc-900/30 rounded-xl border border-zinc-800/50 p-4 flex-1 min-h-0">
          <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Activity className="w-3.5 h-3.5" />Message Bus
          </h3>
          <MessageBus messages={messages} />
        </div>
      </div>
    </div>
  );
}
