import { useState, useEffect } from "react";
import axios from "axios";
import { toast } from "sonner";
import { 
  AlertTriangle, 
  Search, 
  Loader2, 
  CheckCircle2,
  XCircle,
  AlertCircle,
  Bot,
  Zap,
  FileText,
  ArrowRight,
  TrendingUp,
  TrendingDown,
  Building2,
  Globe,
  Shield,
  Truck
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { motion, AnimatePresence } from "framer-motion";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// Risk Level Badge
const RiskBadge = ({ level }) => {
  const config = {
    low: { color: "#10b981", bg: "rgba(16, 185, 129, 0.15)", icon: CheckCircle2 },
    medium: { color: "#f59e0b", bg: "rgba(245, 158, 11, 0.15)", icon: AlertCircle },
    high: { color: "#ef4444", bg: "rgba(239, 68, 68, 0.15)", icon: AlertTriangle },
    critical: { color: "#dc2626", bg: "rgba(220, 38, 38, 0.25)", icon: XCircle },
  };
  
  const cfg = config[level] || config.medium;
  const Icon = cfg.icon;
  
  return (
    <div 
      className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full"
      style={{ backgroundColor: cfg.bg }}
    >
      <Icon className="w-4 h-4" style={{ color: cfg.color }} />
      <span className="text-sm font-semibold capitalize" style={{ color: cfg.color }}>
        {level} Risk
      </span>
    </div>
  );
};

// Analysis Step Component
const AnalysisStep = ({ step, isActive, isComplete }) => (
  <div className={`flex items-center gap-3 p-3 rounded-lg ${isActive ? 'bg-indigo-500/10' : ''}`}>
    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
      isComplete ? 'bg-emerald-500/20' : isActive ? 'bg-indigo-500/20' : 'bg-zinc-800'
    }`}>
      {isComplete ? (
        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
      ) : isActive ? (
        <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
      ) : (
        <div className="w-2 h-2 rounded-full bg-zinc-600" />
      )}
    </div>
    <span className={`text-sm ${isActive ? 'text-white' : isComplete ? 'text-zinc-400' : 'text-zinc-600'}`}>
      {step}
    </span>
  </div>
);

export default function SupplierRiskDemo() {
  const [suppliers, setSuppliers] = useState([]);
  const [selectedSupplier, setSelectedSupplier] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);

  const analysisSteps = [
    "Initializing Supplier Agent (LangChain)",
    "Invoking score_supplier_risk tool",
    "Gathering market intelligence",
    "Processing with Ollama llama3.2:latest.2:latest.2:latest",
    "Running multi-agent orchestration",
    "Generating recommendations"
  ];

  // Fetch suppliers on mount
  useEffect(() => {
    const fetchSuppliers = async () => {
      try {
        const response = await axios.get(`${API}/suppliers`);
        setSuppliers(response.data.suppliers);
        if (response.data.suppliers.length > 0) {
          setSelectedSupplier(response.data.suppliers[0].id);
        }
      } catch (error) {
        console.error("Failed to fetch suppliers:", error);
        toast.error("Failed to load suppliers");
      } finally {
        setLoading(false);
      }
    };
    fetchSuppliers();
  }, []);

  const handleAnalyze = async () => {
    if (!selectedSupplier) {
      toast.error("Please select a supplier");
      return;
    }

    setAnalyzing(true);
    setResult(null);
    setCurrentStep(0);

    // Simulate step progression
    const stepInterval = setInterval(() => {
      setCurrentStep(prev => {
        if (prev < analysisSteps.length - 1) return prev + 1;
        clearInterval(stepInterval);
        return prev;
      });
    }, 700);

    try {
      const response = await axios.post(`${API}/supplier-risk/analyze`, {
        supplier_id: selectedSupplier,
        include_news: true
      });
      
      clearInterval(stepInterval);
      setCurrentStep(analysisSteps.length);
      setResult(response.data);
      toast.success("LangChain analysis complete!");
    } catch (error) {
      clearInterval(stepInterval);
      console.error("Analysis failed:", error);
      toast.error("Analysis failed. Please try again.");
    } finally {
      setAnalyzing(false);
    }
  };

  const selectedSupplierData = suppliers.find(s => s.id === selectedSupplier);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96" data-testid="supplier-risk-loading">
        <Loader2 className="w-8 h-8 animate-spin text-indigo-500" />
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="supplier-risk-page">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white tracking-tight">Supplier Risk Analysis</h1>
        <p className="text-zinc-500 mt-1">LangChain-powered multi-agent risk assessment with real supplier data</p>
      </div>

      <div className="grid grid-cols-12 gap-6">
        {/* Input Section */}
        <div className="col-span-5 space-y-6">
          {/* Supplier Selection Card */}
          <div className="p-6 bg-zinc-900/50 rounded-xl border border-zinc-800/50" data-testid="supplier-selection">
            <h3 className="text-lg font-semibold text-white mb-4">Select Supplier</h3>
            
            <div className="space-y-4">
              <Select value={selectedSupplier} onValueChange={setSelectedSupplier}>
                <SelectTrigger className="w-full bg-zinc-950 border-zinc-800" data-testid="supplier-select">
                  <SelectValue placeholder="Choose a supplier..." />
                </SelectTrigger>
                <SelectContent>
                  {suppliers.map((supplier) => (
                    <SelectItem key={supplier.id} value={supplier.id}>
                      <div className="flex items-center gap-2">
                        <Building2 className="w-4 h-4 text-zinc-500" />
                        <span>{supplier.name}</span>
                        <Badge variant="outline" className="ml-2 text-xs">
                          {supplier.country}
                        </Badge>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {/* Selected Supplier Info */}
              {selectedSupplierData && (
                <div className="p-4 bg-zinc-950 rounded-lg space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-zinc-500">Region</span>
                    <span className="text-sm text-zinc-300 flex items-center gap-1">
                      <Globe className="w-3 h-3" />
                      {selectedSupplierData.region}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-zinc-500">Reliability Score</span>
                    <span className="text-sm text-zinc-300">{selectedSupplierData.reliability_score}%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-zinc-500">On-Time Delivery</span>
                    <span className="text-sm text-zinc-300">{selectedSupplierData.on_time_delivery}%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-zinc-500">Financial Health</span>
                    <Badge variant={selectedSupplierData.financial_health === "Strong" ? "default" : "outline"}>
                      {selectedSupplierData.financial_health}
                    </Badge>
                  </div>
                </div>
              )}

              <Button 
                onClick={handleAnalyze}
                disabled={analyzing || !selectedSupplier}
                className="w-full bg-indigo-600 hover:bg-indigo-700"
                data-testid="analyze-btn"
              >
                {analyzing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Running LangChain Agents...
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4 mr-2" />
                    Analyze with LangChain
                  </>
                )}
              </Button>
            </div>
          </div>

          {/* LangChain Agent Activity */}
          <div className="p-6 bg-zinc-900/50 rounded-xl border border-zinc-800/50" data-testid="agent-activity">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Bot className="w-5 h-5 text-indigo-400" />
              LangChain Agent Activity
            </h3>
            
            <div className="space-y-1">
              {analysisSteps.map((step, idx) => (
                <AnalysisStep
                  key={idx}
                  step={step}
                  isActive={analyzing && currentStep === idx}
                  isComplete={currentStep > idx}
                />
              ))}
            </div>

            {/* Tools Used */}
            <div className="mt-4 pt-4 border-t border-zinc-800">
              <p className="text-xs text-zinc-500 mb-2">LangChain Tools Used:</p>
              <div className="flex flex-wrap gap-2">
                {["score_supplier_risk", "get_market_intelligence", "get_shipping_options"].map((tool) => (
                  <Badge key={tool} variant="outline" className="font-mono text-xs">
                    @tool {tool}()
                  </Badge>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Results Section */}
        <div className="col-span-7">
          <AnimatePresence mode="wait">
            {result ? (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-6"
                data-testid="analysis-result"
              >
                {/* Risk Summary Card */}
                <div className="p-6 bg-zinc-900/50 rounded-xl border border-zinc-800/50">
                  <div className="flex items-start justify-between mb-6">
                    <div>
                      <h2 className="text-2xl font-bold text-white">
                        {result.risk_assessment?.supplier_name || selectedSupplierData?.name}
                      </h2>
                      <p className="text-sm text-zinc-500 flex items-center gap-2 mt-1">
                        <Globe className="w-4 h-4" />
                        {result.risk_assessment?.country} • {result.risk_assessment?.region}
                      </p>
                    </div>
                    <RiskBadge level={result.risk_assessment?.risk_level || "medium"} />
                  </div>

                  {/* Risk Score */}
                  <div className="mb-6">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-zinc-400">Overall Risk Score</span>
                      <span className="text-2xl font-bold font-mono text-white">
                        {result.risk_assessment?.risk_score?.toFixed(1) || 0}%
                      </span>
                    </div>
                    <Progress 
                      value={result.risk_assessment?.risk_score || 0} 
                      className="h-3"
                    />
                  </div>

                  {/* Component Scores */}
                  {result.risk_assessment?.component_scores && (
                    <div className="grid grid-cols-4 gap-4 mb-6">
                      {Object.entries(result.risk_assessment.component_scores).map(([key, value]) => (
                        <div key={key} className="p-3 bg-zinc-950 rounded-lg text-center">
                          <p className="text-xs text-zinc-500 capitalize mb-1">{key}</p>
                          <p className="text-lg font-bold font-mono text-white">{value}%</p>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Risk Factors */}
                  <div className="mb-6">
                    <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3">
                      Risk Factors Identified
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {result.risk_assessment?.risk_factors?.map((factor, idx) => (
                        <Badge 
                          key={idx}
                          variant="outline"
                          className="border-amber-500/50 text-amber-400"
                        >
                          <AlertCircle className="w-3 h-3 mr-1" />
                          {factor}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  {/* Shipment Performance */}
                  {result.risk_assessment?.shipment_performance && (
                    <div className="p-4 bg-zinc-950 rounded-lg mb-6">
                      <h3 className="text-sm font-semibold text-zinc-400 mb-3 flex items-center gap-2">
                        <Truck className="w-4 h-4" />
                        Shipment Performance
                      </h3>
                      <div className="grid grid-cols-3 gap-4 text-center">
                        <div>
                          <p className="text-2xl font-bold text-white">{result.risk_assessment.shipment_performance.total_shipments}</p>
                          <p className="text-xs text-zinc-500">Total Shipments</p>
                        </div>
                        <div>
                          <p className="text-2xl font-bold text-emerald-400">{result.risk_assessment.shipment_performance.on_time_shipments}</p>
                          <p className="text-xs text-zinc-500">On Time</p>
                        </div>
                        <div>
                          <p className="text-2xl font-bold text-amber-400">{result.risk_assessment.shipment_performance.delay_rate_pct}%</p>
                          <p className="text-xs text-zinc-500">Delay Rate</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Recommendation */}
                  <div className="p-4 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
                    <h3 className="text-sm font-semibold text-emerald-400 mb-2 flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4" />
                      AI Recommendation
                    </h3>
                    <p className="text-sm text-zinc-300">{result.risk_assessment?.recommendation}</p>
                  </div>
                </div>

                {/* Agent Insights */}
                {result.agent_insights && (
                  <div className="p-6 bg-zinc-900/50 rounded-xl border border-zinc-800/50" data-testid="agent-insights">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <Bot className="w-5 h-5 text-purple-400" />
                      Supplier Agent Insights (LangChain)
                    </h3>
                    <div className="space-y-3">
                      <p className="text-sm text-zinc-300">{result.agent_insights.analysis}</p>
                      {result.agent_insights.recommendations && (
                        <div className="mt-4">
                          <p className="text-xs text-zinc-500 mb-2">Agent Recommendations:</p>
                          <ul className="space-y-2">
                            {result.agent_insights.recommendations.map((rec, idx) => (
                              <li key={idx} className="flex items-start gap-2 text-sm text-zinc-400">
                                <ArrowRight className="w-4 h-4 text-indigo-400 mt-0.5 flex-shrink-0" />
                                {rec}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Market Intelligence */}
                {result.market_intelligence?.news_items && (
                  <div className="p-6 bg-zinc-900/50 rounded-xl border border-zinc-800/50" data-testid="market-intel">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <FileText className="w-5 h-5 text-blue-400" />
                      Market Intelligence
                    </h3>
                    <ScrollArea className="h-48">
                      <div className="space-y-3">
                        {result.market_intelligence.news_items.map((news, idx) => (
                          <div key={idx} className="p-3 bg-zinc-950 rounded-lg">
                            <div className="flex items-start justify-between">
                              <p className="text-sm text-zinc-300">{news.headline}</p>
                              <Badge variant={news.sentiment === "positive" ? "default" : "destructive"} className="ml-2 flex-shrink-0">
                                {news.sentiment}
                              </Badge>
                            </div>
                            <p className="text-xs text-zinc-500 mt-1">{news.source} • {news.date}</p>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </div>
                )}
              </motion.div>
            ) : (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="h-96 flex flex-col items-center justify-center text-zinc-500 bg-zinc-900/30 rounded-xl border border-zinc-800/50"
                data-testid="empty-state"
              >
                <AlertTriangle className="w-16 h-16 mb-4 opacity-30" />
                <p className="text-lg mb-2">No Analysis Results</p>
                <p className="text-sm text-zinc-600">Select a supplier and click "Analyze with LangChain"</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* LangChain Architecture Info */}
      <div className="p-6 bg-zinc-900/30 rounded-xl border border-zinc-800/50" data-testid="architecture-info">
        <h3 className="text-lg font-semibold text-white mb-4">LangChain Multi-Agent Architecture</h3>
        <div className="grid grid-cols-4 gap-6">
          {[
            { 
              icon: Bot, 
              title: "LangChain Agents", 
              desc: "5 specialized agents built with LangChain's agent framework",
              color: "#a855f7"
            },
            { 
              icon: Zap, 
              title: "Custom Tools", 
              desc: "8 @tool decorated functions for supply chain operations",
              color: "#f59e0b"
            },
            { 
              icon: Shield, 
              title: "Ollama llama3.2:latest.2:latest.2:latest", 
              desc: "Local LLM inference via ChatOllama integration",
              color: "#06b6d4"
            },
            { 
              icon: FileText, 
              title: "Real Data", 
              desc: "500+ shipments, 8 suppliers, 24 months history",
              color: "#10b981"
            },
          ].map((item, idx) => (
            <div key={idx} className="relative">
              <div 
                className="w-10 h-10 rounded-lg flex items-center justify-center mb-3"
                style={{ backgroundColor: `${item.color}20` }}
              >
                <item.icon className="w-5 h-5" style={{ color: item.color }} />
              </div>
              <h4 className="text-sm font-semibold text-white mb-1">{item.title}</h4>
              <p className="text-xs text-zinc-500">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}