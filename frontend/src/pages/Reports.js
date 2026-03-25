import { useState, useEffect } from "react";
import axios from "axios";
import { toast } from "sonner";
import { 
  FileText, 
  Presentation, 
  Download, 
  Loader2,
  ChevronDown,
  ChevronRight,
  BookOpen,
  Target,
  Cpu,
  Workflow,
  DollarSign,
  Lightbulb,
  CheckCircle2
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import ReactMarkdown from 'react-markdown';

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const sectionIcons = {
  "executive-summary": BookOpen,
  "problem-statement": Target,
  "solution-architecture": Cpu,
  "workflows": Workflow,
  "roi-analysis": DollarSign,
  "patentability": Lightbulb,
};

// Report Section Component
const ReportSection = ({ section, isOpen, onToggle }) => {
  const Icon = sectionIcons[section.id] || FileText;
  
  return (
    <Collapsible open={isOpen} onOpenChange={onToggle}>
      <CollapsibleTrigger className="w-full" data-testid={`section-toggle-${section.id}`}>
        <div className="flex items-center justify-between p-4 bg-zinc-900/50 rounded-lg hover:bg-zinc-900/80 transition-colors">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/20 flex items-center justify-center">
              <Icon className="w-4 h-4 text-indigo-400" />
            </div>
            <span className="text-sm font-semibold text-white">{section.title}</span>
          </div>
          {isOpen ? (
            <ChevronDown className="w-5 h-5 text-zinc-400" />
          ) : (
            <ChevronRight className="w-5 h-5 text-zinc-400" />
          )}
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="p-4 mt-2 bg-zinc-900/30 rounded-lg border border-zinc-800/50">
          <div className="prose prose-invert prose-sm max-w-none">
            <div className="text-zinc-300 whitespace-pre-line leading-relaxed">
              {section.content}
            </div>
          </div>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
};

// Slide Preview Component
const SlidePreview = ({ slide, isSelected, onClick }) => (
  <div 
    className={`p-4 rounded-lg border cursor-pointer transition-all ${
      isSelected 
        ? 'border-indigo-500 bg-indigo-500/10' 
        : 'border-zinc-800 bg-zinc-900/50 hover:border-zinc-700'
    }`}
    onClick={onClick}
    data-testid={`slide-preview-${slide.number}`}
  >
    <div className="flex items-center gap-2 mb-2">
      <span className="w-6 h-6 rounded bg-zinc-800 text-xs flex items-center justify-center text-zinc-400">
        {slide.number}
      </span>
      <span className="text-sm font-medium text-white truncate">{slide.title}</span>
    </div>
    <div className="text-xs text-zinc-500 line-clamp-2">
      {slide.bullets?.[0] || slide.subtitle || ''}
    </div>
  </div>
);

// Full Slide View
const SlideView = ({ slide }) => {
  if (!slide) return null;
  
  return (
    <div 
      className="aspect-video bg-gradient-to-br from-zinc-900 to-zinc-950 rounded-xl border border-zinc-800 p-8 flex flex-col"
      data-testid="slide-view"
    >
      <div className="mb-auto">
        <h2 className="text-3xl font-bold text-white mb-2">{slide.title}</h2>
        {slide.subtitle && (
          <p className="text-lg text-zinc-400">{slide.subtitle}</p>
        )}
      </div>
      
      {slide.bullets && (
        <ul className="space-y-3 mt-6">
          {slide.bullets.map((bullet, idx) => (
            <li key={idx} className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-indigo-400 mt-0.5 flex-shrink-0" />
              <span className="text-zinc-300">{bullet}</span>
            </li>
          ))}
        </ul>
      )}
      
      <div className="mt-auto pt-6 flex justify-between items-center border-t border-zinc-800">
        <span className="text-xs text-zinc-600">SupplyMind</span>
        <span className="text-xs text-zinc-600">Slide {slide.number}/10</span>
      </div>
    </div>
  );
};

export default function Reports() {
  const [masterReport, setMasterReport] = useState(null);
  const [slides, setSlides] = useState([]);
  const [activeTab, setActiveTab] = useState("report");
  const [openSections, setOpenSections] = useState({});
  const [selectedSlide, setSelectedSlide] = useState(null);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    const fetchReports = async () => {
      try {
        const [reportRes, slidesRes] = await Promise.all([
          axios.get(`${API}/reports/master`),
          axios.get(`${API}/reports/slides`),
        ]);
        
        setMasterReport(reportRes.data);
        setSlides(slidesRes.data.slides);
        
        // Open first section by default
        if (reportRes.data.sections?.length > 0) {
          setOpenSections({ [reportRes.data.sections[0].id]: true });
        }
        
        // Select first slide by default
        if (slidesRes.data.slides?.length > 0) {
          setSelectedSlide(slidesRes.data.slides[0]);
        }
      } catch (error) {
        console.error("Failed to fetch reports:", error);
        toast.error("Failed to load reports");
      } finally {
        setLoading(false);
      }
    };

    fetchReports();
  }, []);

  const handleDownload = async (type) => {
    setDownloading(true);
    try {
      // For now, we'll create a text file with the content
      // In a real implementation, you'd generate PDFs server-side
      let content, filename;
      
      if (type === 'report') {
        content = `${masterReport.title}\n${'='.repeat(60)}\n\n`;
        masterReport.sections.forEach(section => {
          content += `\n${section.title}\n${'-'.repeat(40)}\n${section.content}\n`;
        });
        filename = 'SupplyMind_Master_Report.txt';
      } else {
        content = `${slides[0]?.title || 'SupplyMind Pitch Deck'}\n${'='.repeat(60)}\n\n`;
        slides.forEach(slide => {
          content += `\nSlide ${slide.number}: ${slide.title}\n${'-'.repeat(40)}\n`;
          if (slide.subtitle) content += `${slide.subtitle}\n`;
          if (slide.bullets) {
            slide.bullets.forEach(bullet => {
              content += `• ${bullet}\n`;
            });
          }
          content += '\n';
        });
        filename = 'SupplyMind_Pitch_Deck.txt';
      }
      
      const blob = new Blob([content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      toast.success(`${type === 'report' ? 'Report' : 'Slides'} downloaded successfully`);
    } catch (error) {
      toast.error("Download failed");
    } finally {
      setDownloading(false);
    }
  };

  const toggleSection = (sectionId) => {
    setOpenSections(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96" data-testid="reports-loading">
        <div className="text-zinc-500">Loading reports...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="reports-page">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Reports & Deliverables</h1>
          <p className="text-zinc-500 mt-1">Academic project deliverables and pitch deck</p>
        </div>
        
        <div className="flex gap-3">
          <Button
            variant="outline"
            onClick={() => handleDownload('report')}
            disabled={downloading}
            className="bg-zinc-900 border-zinc-800 hover:bg-zinc-800"
            data-testid="download-report-btn"
          >
            {downloading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Download className="w-4 h-4 mr-2" />}
            Download Report
          </Button>
          <Button
            onClick={() => handleDownload('slides')}
            disabled={downloading}
            className="bg-indigo-600 hover:bg-indigo-700"
            data-testid="download-slides-btn"
          >
            {downloading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Presentation className="w-4 h-4 mr-2" />}
            Download Slides
          </Button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-2 p-1 bg-zinc-900 rounded-lg w-fit">
        <button
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            activeTab === 'report' 
              ? 'bg-zinc-800 text-white' 
              : 'text-zinc-400 hover:text-white'
          }`}
          onClick={() => setActiveTab('report')}
          data-testid="tab-report"
        >
          <FileText className="w-4 h-4 inline mr-2" />
          Master Report
        </button>
        <button
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            activeTab === 'slides' 
              ? 'bg-zinc-800 text-white' 
              : 'text-zinc-400 hover:text-white'
          }`}
          onClick={() => setActiveTab('slides')}
          data-testid="tab-slides"
        >
          <Presentation className="w-4 h-4 inline mr-2" />
          Pitch Deck
        </button>
      </div>

      {/* Content */}
      {activeTab === 'report' ? (
        <div className="grid grid-cols-12 gap-6">
          {/* Report Content */}
          <div className="col-span-8 space-y-4" data-testid="report-content">
            <div className="p-6 bg-zinc-900/30 rounded-xl border border-zinc-800/50">
              <h2 className="text-xl font-bold text-white mb-2">{masterReport?.title}</h2>
              <p className="text-sm text-zinc-500">
                Generated: {masterReport?.generated_at ? new Date(masterReport.generated_at).toLocaleString() : 'N/A'}
              </p>
            </div>
            
            {masterReport?.sections?.map((section) => (
              <ReportSection
                key={section.id}
                section={section}
                isOpen={openSections[section.id] || false}
                onToggle={() => toggleSection(section.id)}
              />
            ))}
          </div>

          {/* Table of Contents */}
          <div className="col-span-4">
            <div className="p-4 bg-zinc-900/30 rounded-xl border border-zinc-800/50 sticky top-24">
              <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-4">
                Table of Contents
              </h3>
              <nav className="space-y-2">
                {masterReport?.sections?.map((section) => (
                  <button
                    key={section.id}
                    className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                      openSections[section.id]
                        ? 'bg-indigo-500/20 text-indigo-300'
                        : 'text-zinc-400 hover:bg-zinc-800 hover:text-white'
                    }`}
                    onClick={() => toggleSection(section.id)}
                  >
                    {section.title}
                  </button>
                ))}
              </nav>
            </div>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-12 gap-6">
          {/* Slide Preview */}
          <div className="col-span-8">
            <SlideView slide={selectedSlide} />
          </div>

          {/* Slide List */}
          <div className="col-span-4">
            <ScrollArea className="h-[500px] pr-4">
              <div className="space-y-3" data-testid="slides-list">
                {slides.map((slide) => (
                  <SlidePreview
                    key={slide.number}
                    slide={slide}
                    isSelected={selectedSlide?.number === slide.number}
                    onClick={() => setSelectedSlide(slide)}
                  />
                ))}
              </div>
            </ScrollArea>
          </div>
        </div>
      )}
    </div>
  );
}
