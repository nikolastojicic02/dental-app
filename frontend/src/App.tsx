import { useState, useRef } from 'react'; 
import { Upload, Activity, CheckCircle2, ChevronRight, Play, Info, Download, Maximize2, AlertTriangle } from 'lucide-react';
import ModelViewer from './components/ModelViewer';
import { generateDentalReport } from './utils/reportGenerator';

type Stage = 'upload' | 'detection' | 'cropping' | 'segmentation' | 'metrics';

interface StageData {
  upload?: any;
  detection?: any;
  cropping?: any;
  segmentation?: any;
  metrics?: any;
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentStage, setCurrentStage] = useState<Stage>('upload');
  const [loading, setLoading] = useState(false);
  const [stageData, setStageData] = useState<StageData>({});
  const [selectedMetric, setSelectedMetric] = useState<string | null>('pulpal_depth');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  // 1. Dodaj state na vrh App komponente
  const [isExporting, setIsExporting] = useState(false);
  const glRef = useRef<any>(null);

  const API_BASE = 'http://127.0.0.1:8000';

  // 2. Funkcija za eksport
  const handleExportClick = () => {
    setIsExporting(true); // Otvara "namesti kameru" modal
  };

 const finalizeExport = () => {
  const gl = glRef.current;
  const metrics = stageData.metrics;

  console.log("DEBUG EXPORT:");
  console.log("1. WebGL (glRef):", gl);
  console.log("2. Metrics (stageData):", metrics);

  if (!gl) {
    alert("3D model context is not captured. Please rotate the model once and try again.");
    return;
  }
  
  if (!metrics || !metrics.metrics) {
    alert("Metrics data is missing. Please run Stage 4 first.");
    return;
  }

  const screenshot = gl.domElement.toDataURL("image/png");
  generateDentalReport(
    { sessionId, filename: file?.name },
    metrics.metrics,
    screenshot
  );
  
  setIsExporting(false);
};



  // --- DRAG & DROP HANDLERS ---
  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.name.toLowerCase().endsWith('.stl')) {
      await processFile(droppedFile);
    }
  };

  // --- FILE PROCESSING ---
  const processFile = async (selectedFile: File) => {
    setFile(selectedFile);
    setStageData({});
    setSessionId(null);
    setSelectedMetric(null);
    setCurrentStage('upload');
    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
      const response = await fetch(`${API_BASE}/stage0/upload`, { method: 'POST', body: formData });
      const data = await response.json();
      if (data.status === 'success') {
        setSessionId(data.session_id);
        setStageData({ upload: data.data });
      }
    } catch (error) { console.error(error); } finally { setLoading(false); }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      await processFile(e.target.files[0]);
    }
  };

  const runStage = async (stageNum: number, endpoint: string, stageKey: Stage) => {
    if (!sessionId) return;
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/${endpoint}/${sessionId}`, { method: 'POST' });
      const data = await response.json();
      if (data.status === 'success') {
        setStageData(prev => ({ ...prev, [stageKey]: data.data }));
        setCurrentStage(stageKey);
        if (stageKey === 'metrics') setSelectedMetric('pulpal_depth');
      }
    } catch (error) { console.error(error); } finally { setLoading(false); }
  };

  const processAll = async () => {
    if (!sessionId || loading) return;
    if (!stageData.detection) await runStage(1, 'stage1/detect', 'detection');
    if (!stageData.cropping) await runStage(2, 'stage2/crop', 'cropping');
    if (!stageData.segmentation) await runStage(3, 'stage3/segment', 'segmentation');
    if (!stageData.metrics) await runStage(4, 'stage4/metrics', 'metrics');
  };

  const metricsConfig = [
    { id: 'pulpal_depth', label: 'PULPAL DEPTH' },
    { id: 'bl_ratio', label: 'B-L RATIO' },
    { id: 'axial_height', label: 'AXIAL HEIGHT' },
    { id: 'wall_taper', label: 'WALL TAPER' },
    { id: 'marginal_ridge', label: 'MARGINAL RIDGE' },
    { id: 'wall_smoothness', label: 'WALL SMOOTH' },
    { id: 'undercuts', label: 'UNDERCUTS' },
    { id: 'cusp_undermining', label: 'CUSP UND_MIN' },
    { id: 'floor_flatness', label: 'FLOOR FLATNESS' },
  ];

  const segmentationClasses = [
    { name: 'Intact Tooth', color: '#7E7E7E', count: 1 },
    { name: 'Pulpal Floor', color: '#0000FE', count: 2 },
    { name: 'Gingival Floor', color: '#00FEFE', count: 3 },
    { name: 'F-L Walls', color: '#FE0000', count: 4 },
    { name: 'Axial Wall', color: '#00FE00', count: 5 },
    { name: 'Distal Wall', color: '#FEFE00', count: 6 },
  ];

  const isMetricsReady = !!stageData.metrics;
  const showInsight = isMetricsReady && selectedMetric;

  // --- METRIC INFO HELPER ---
  const getMetricInfo = (metricId: string, grade: string) => {
    console.log('🔍 DEBUG - metricId:', metricId, 'grade:', grade);
    const metricData: Record<string, any> = {
      pulpal_depth: {
        ranges: {
          Excellent: { min: 1.5, max: 2.0, label: '1.5 - 2.0mm: Ideal', color: '#34C759' },
          'Clinically Acceptable': { min: 2.0, max: 4.0, label: '2.0 - 4.0mm: Acceptable', color: '#FFCC00' },
          'Standard Not Met': { min: null, max: null, label: '< 1.5mm or > 4.0mm: Critical', color: '#FF3B30' }
        },
        insights: {
          Excellent: 'Ideal depth. You successfully bypassed the DEJ into dentin while maintaining a safe distance from the pulp horn.',
          'Clinically Acceptable': 'Deeper than ideal. Use a liner/base (like Dycal or Vitrebond) to protect the pulp from thermal sensitivity and microleakage.',
          'Standard Not Met': 'If too shallow, the restorative material lacks bulk and will fracture. Use your bur head (e.g., #330 is ~1.5-2mm long) as a visual depth gauge.'
        }
      },
      bl_ratio: {
        ranges: {
          Excellent: { min: 0.25, max: 0.33, label: '0.25 - 0.33: Ideal', color: '#34C759' },
          'Clinically Acceptable': { min: 0.33, max: 0.66, label: '0.33 - 0.66: Acceptable', color: '#FFCC00' },
          'Standard Not Met': { min: null, max: null, label: '< 0.25 or > 0.66: Critical', color: '#FF3B30' }
        },
        insights: {
          Excellent: 'Perfect conservation of tooth structure. This width maintains the strength of the buccal and lingual cusps.',
          'Clinically Acceptable': 'The preparation is wide. Be cautious with condensation pressure to avoid stressing the remaining weakened cusps.',
          'Standard Not Met': 'Preparation is too wide. The "wedge effect" during chewing may fracture the cusps. Next time, keep your bur strictly within the central groove.'
        }
      },
      axial_height: {
        ranges: {
          Excellent: { min: 1.0, max: 1.5, label: '1.0 - 1.5mm: Ideal', color: '#34C759' },
          'Clinically Acceptable': { min: 0.8, max: 2.0, label: '0.8 - 2.0mm: Acceptable', color: '#FFCC00' },
          'Standard Not Met': { min: null, max: null, label: '< 0.8mm or > 2.0mm: Critical', color: '#FF3B30' }
        },
        insights: {
          Excellent: 'Correct vertical box dimension. Provides adequate resistance form without encroaching on the gingival attachment.',
          'Clinically Acceptable': 'Slightly high or low. Check your gingival floor level—ensure it clears the contact point with the adjacent tooth by 0.5mm.',
          'Standard Not Met': 'If too short, the box won\'t provide enough retention. If too deep, you risk invading "Biological Width." Use a periodontal probe to verify floor depth.'
        }
      },
      wall_taper: {
        ranges: {
          Excellent: { min: 85, max: 93, label: '85° - 93°: Ideal', color: '#34C759' },
          'Clinically Acceptable': { min: null, max: 93, label: '≤ 93°: Acceptable', color: '#FFCC00' },
          'Standard Not Met': { min: 93, max: null, label: '> 93°: Critical', color: '#FF3B30' }
        },
        insights: {
          Excellent: 'Excellent convergence. This "mechanical lock" prevents the filling from lifting out under sticky food forces.',
          'Clinically Acceptable': 'Near parallel. While acceptable, try to tilt the bur slightly (3-5 degrees) toward the center of the tooth for better retention.',
          'Standard Not Met': 'Divergent walls. The filling will not stay in place. Ensure the handpiece is parallel to the long axis of the tooth, not tilted outward.'
        }
      },
      marginal_ridge: {
        ranges: {
          Excellent: { min: 1.6, max: null, label: '≥ 1.6mm: Ideal', color: '#34C759' },
          'Clinically Acceptable': { min: 1.2, max: 1.6, label: '1.2 - 1.6mm: Acceptable', color: '#FFCC00' },
          'Standard Not Met': { min: null, max: 1.2, label: '< 1.2mm: Critical', color: '#FF3B30' }
        },
        insights: {
          Excellent: 'Strong ridge. This thickness is crucial to prevent the tooth from splitting during heavy occlusal loading.',
          'Clinically Acceptable': 'Borderline thin. Avoid any further distal/mesial extension. Ensure your internal line angles are rounded to reduce stress.',
          'Standard Not Met': 'Ridge is critically weak. In a clinical setting, this ridge would likely fracture. Practice keeping a "safety zone" of tooth structure near the proximal surfaces.'
        }
      },
      wall_smoothness: {
        ranges: {
          Excellent: { min: null, max: 0.08, label: '< 0.08: Ideal', color: '#34C759' },
          'Clinically Acceptable': { min: 0.08, max: 0.15, label: '0.08 - 0.15: Acceptable', color: '#FFCC00' },
          'Standard Not Met': { min: 0.15, max: null, label: '≥ 0.15: Critical', color: '#FF3B30' }
        },
        insights: {
          Excellent: 'Glass-like finish. This ensures the best marginal adaptation and prevents voids at the tooth-material interface.',
          'Clinically Acceptable': 'Slightly rough. Switch to a fine-grit finishing diamond or a multi-fluted carbide bur at low speed for a smoother finish.',
          'Standard Not Met': 'Visible steps or "chatter marks." These create stress concentrations. Use a "planing" motion with a hand instrument (like an enamel hatchet) to smooth the walls.'
        }
      },
      undercuts: {
        ranges: {
          Excellent: { min: null, max: 0.5, label: '< 0.5%: Ideal', color: '#34C759' },
          'Clinically Acceptable': { min: 0.5, max: 3.0, label: '0.5% - 3.0%: Acceptable', color: '#FFCC00' },
          'Standard Not Met': { min: 3.0, max: null, label: '≥ 3.0%: Critical', color: '#FF3B30' }
        },
        insights: {
          Excellent: 'No undercuts. This allows for perfect condensation of composite or amalgam into the line angles.',
          'Clinically Acceptable': 'Minor shadowing. Ensure you are not "burying" the head of the bur into the wall, which creates a bell-shaped prep.',
          'Standard Not Met': 'Significant undercuts found. These trap air bubbles and weaken the enamel. Keep the bur\'s path of motion straight up and down.'
        }
      },
      cusp_undermining: {
        ranges: {
          Excellent: { min: 2.0, max: null, label: '≥ 2.0mm: Ideal', color: '#34C759' },
          'Clinically Acceptable': { min: 1.5, max: 2.0, label: '1.5 - 2.0mm: Acceptable', color: '#FFCC00' },
          'Standard Not Met': { min: null, max: 1.5, label: '< 1.5mm: Critical', color: '#FF3B30' }
        },
        insights: {
          Excellent: 'Cusps are supported by a healthy "dentin foundation." This tooth can withstand high chewing forces.',
          'Clinically Acceptable': 'Dentin support is thin. Be careful when removing decay near the pulp horns to avoid "hollowing out" the cusp.',
          'Standard Not Met': 'The cusp is undermined (enamel with no dentin under it). This cusp will likely break. Consider "capping" the cusp or a more extensive restoration.'
        }
      },
      floor_flatness: {
        ranges: {
          Excellent: { min: null, max: 0.12, label: '< 0.12: Ideal', color: '#34C759' },
          'Clinically Acceptable': { min: 0.12, max: 0.25, label: '0.12 - 0.25: Acceptable', color: '#FFCC00' },
          'Standard Not Met': { min: 0.25, max: null, label: '≥ 0.25: Critical', color: '#FF3B30' }
        },
        insights: {
          Excellent: 'Perfectly flat floor. This "Resistance Form" prevents the restoration from rocking or shifting under pressure.',
          'Clinically Acceptable': 'Slightly uneven. Use a flat-ended bur like a #56 or #245 to gently level the pulpal floor.',
          'Standard Not Met': 'Sloping or "pitted" floor. This creates uneven pressure points. Move the bur in a sweeping, horizontal "mowing" motion rather than pushing down.'
        }
      }
    };

    const metric = metricData[metricId];
    if (!metric) return null;

    const ranges = metric.ranges;
    const insight = metric.insights[grade] || 'No insight available.';
    
    return {
      ranges: Object.values(ranges),
      insight,
      color: ranges[grade]?.color || '#34C759'
    };
  };

  return (
    <div className="h-screen w-full bg-[#F2F2F7] flex overflow-hidden font-sans select-none">
      
      {/* 1. LEVI SIDEBAR: DIAGNOSTICS */}
      <aside className={`w-[90px] bg-white border-r border-[#E5E5E7] flex flex-col items-center py-4 z-20 transition-all ${!isSidebarOpen && '-ml-[90px]'}`}>
        <span className="text-[9px] font-semibold text-slate-600 tracking-[1px] mb-3">METRICS</span>
        <div className="flex flex-col gap-1.5 w-full px-2 overflow-y-auto no-scrollbar">
          {metricsConfig.map((m) => {
            const mData = stageData.metrics?.metrics?.[m.id];
            const isActive = selectedMetric === m.id;
            return (
              <button
                key={m.id}
                disabled={!isMetricsReady}
                onClick={() => setSelectedMetric(isActive ? null : m.id)}
                className={`relative w-full py-2.5 rounded-lg flex flex-col items-center justify-center transition-all cursor-pointer ${
                  !isMetricsReady ? 'opacity-25 cursor-not-allowed' : 
                  isActive ? 'bg-[#0071E3]/8 ring-1 ring-[#0071E3]/30' : 'bg-slate-50 hover:bg-slate-100'
                }`}
              >
                {isMetricsReady && mData && (
                  <div className={`absolute top-1 right-1 w-1 h-1 rounded-full ${
                    mData.color === 'green' ? 'bg-[#34C759]' : 
                    mData.color === 'red' ? 'bg-[#FF3B30]' : 
                    mData.color === 'orange' ? 'bg-[#FFCC00]' : 'bg-slate-300'
                  }`} />
                )}
                <span className={`text-[8px] font-semibold text-center leading-[1.2] tracking-tight uppercase ${isActive ? 'text-[#0071E3]' : 'text-slate-800'}`}>
                  {m.label.split(' ')[0]}<br/>{m.label.split(' ')[1] || ''}
                </span>
              </button>
            );
          })}
        </div>
      </aside>

      {/* 2. CENTRALNI VIEWPORT */}
      <main className="flex-1 flex flex-col p-4 relative min-w-0">
        <div className="flex-1 bg-white rounded-3xl border border-[#D1D1D6] overflow-hidden relative group">
          
          {!file ? (
            /* --- ONBOARDING SCREEN --- */
/* --- ONBOARDING SCREEN --- */
<div className="h-full w-full flex flex-col items-center justify-center p-8">
  <label 
    className={`w-[420px] h-[200px] rounded-2xl border-2 border-dashed transition-all cursor-pointer flex flex-col items-center justify-center gap-3 group/upload mb-8 ${
      isDragging 
        ? 'border-[#0071E3] bg-[#0071E3]/5 scale-[1.02]' 
        : 'border-[#0071E3]/25 hover:border-[#0071E3] bg-white'
    }`}
    onDragEnter={handleDragEnter}
    onDragLeave={handleDragLeave}
    onDragOver={handleDragOver}
    onDrop={handleDrop}
  >
    <input type="file" className="hidden" accept=".stl" onChange={handleFileChange} />
    <div className="w-11 h-11 bg-[#0071E3]/5 rounded-full flex items-center justify-center transition-transform group-hover/upload:scale-105">
      <Upload size={20} className="text-[#0071E3]" />
    </div>
    <div className="text-center">
      <p className="text-base font-semibold text-slate-900">Upload STL Scan</p>
      <p className="text-sm font-medium text-slate-600 mt-0.5">Drag and drop or click to browse</p>
    </div>
    <span className="text-[10px] font-medium text-[#0071E3]/40 tracking-wider mt-0.5">SUPPORTED: .STL BINARY / ASCII</span>
  </label>

  <div className="w-full max-w-3xl">
  <div className="flex flex-col items-center mb-6">
    <span className="text-[11px] font-semibold text-slate-800 tracking-[2px] uppercase">
      Scan Guidelines
    </span>
    <div className="w-8 h-[2px] bg-[#0071E3] rounded-full mt-1.5" />
  </div>

  <div className="grid grid-cols-2 gap-6 px-8">
    
    {/* CARD 1 */}
    <div className="bg-slate-50 p-6 rounded-2xl flex flex-col items-center text-center">
      
      {/* IMAGE WRAPPER – FIXED HEIGHT */}
      <div className="h-28 flex items-center justify-center mb-4">
        <img
          src="/karta1.png"
          alt="Class II Extension"
          className="max-h-full object-contain"
        />
      </div>

      {/* TEXT */}
      <h3 className="font-semibold text-sm mb-1.5 text-slate-900">
        Class II Extension
      </h3>
      <p className="text-[11px] leading-relaxed text-slate-700 font-medium px-3">
        Proximal boxes extend approximately to the tooth’s mid-depth.
      </p>
    </div>

    {/* CARD 2 */}
    <div className="bg-slate-50 p-6 rounded-2xl flex flex-col items-center text-center">
      
      {/* IMAGE WRAPPER – SAME HEIGHT */}
      <div className="h-28 flex items-center justify-center mb-4">
        <img
          src="/karta2.png"
          alt="Centered View"
          className="max-h-full object-contain"
        />
      </div>

      {/* TEXT */}
      <h3 className="font-semibold text-sm mb-1.5 text-slate-900">
        Centered View
      </h3>
      <p className="text-[11px] leading-relaxed text-slate-700 font-medium px-3">
        The prepared tooth is centered and flanked by at least two neighboring teeth.
      </p>
    </div>

  </div>
</div>

</div>
          ) : (
            <>
              <ModelViewer 
                glRef={glRef}
                stageData={stageData} 
                currentStage={currentStage} 
                selectedMetric={selectedMetric}
                isSidebarOpen={isSidebarOpen}
                setIsSidebarOpen={setIsSidebarOpen}
              />
              
              {/* Analysis Card - Gore levo */}
              {selectedMetric && stageData.metrics?.metrics?.[selectedMetric] && (() => {
                const metricData = stageData.metrics.metrics[selectedMetric];
                const metricInfo = getMetricInfo(selectedMetric, metricData.grade);
                
                return (
                  <div className="absolute top-4 left-4 w-[190px] bg-white/80 backdrop-blur-xl rounded-2xl p-4 border border-white/60 ring-1 ring-black/5 animate-in fade-in zoom-in-95 duration-300 z-10">
                    <span className="text-[9px] font-semibold text-slate-600 tracking-wider uppercase">{metricsConfig.find(m => m.id === selectedMetric)?.label}</span>
                    <div className="text-2xl font-semibold mt-1.5 text-slate-900 tracking-tight">
                      {metricData.value}
                      <span className="text-sm font-medium ml-1 text-slate-600">{metricData.unit}</span>
                    </div>
                    <div 
                      className="inline-flex items-center px-2.5 py-0.5 text-[10px] font-semibold rounded-full mt-2 uppercase tracking-tight"
                      style={{
                        backgroundColor: metricInfo?.color ? `${metricInfo.color}15` : '#E8F5E9',
                        color: metricInfo?.color || '#2E7D32'
                      }}
                    >
                      {metricData.grade || 'Excellent'}
                    </div>
                    <div className="h-px bg-slate-200 w-full my-3" />
                    <span className="text-[9px] font-semibold block mb-2 uppercase tracking-wide text-slate-700">Reference Scale</span>
                    <div className="space-y-1.5">
                      {metricInfo?.ranges.map((range: any, idx: number) => (
                        <div key={idx} className="flex items-center gap-2 text-[10px] font-medium text-slate-800">
                          <div className="w-1 h-1 rounded-sm" style={{ backgroundColor: range.color }} />
                          {range.label}
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })()}
            </>
          )}
        </div>

        {/* Clinical Insight / Disclaimer */}
        {showInsight ? (() => {
          const metricData = stageData.metrics.metrics[selectedMetric];
          const metricInfo = getMetricInfo(selectedMetric, metricData.grade);
          
          return (
            <div className="mt-3 mx-1 bg-white/90 backdrop-blur-md border border-slate-200 rounded-2xl p-3 flex items-center gap-3">
              <div className="w-7 h-7 bg-[#0071E3] rounded-full flex items-center justify-center text-white font-semibold text-sm shadow-sm">?</div>
              <div className="flex flex-col">
                <span className="text-[10px] font-semibold text-slate-600 tracking-[1.2px] uppercase">Clinical Insight</span>
                <p className="text-[13px] font-medium text-slate-800 mt-0.5 leading-tight">
                  {metricInfo?.insight || 'No specific guidance available for this metric.'}
                </p>
              </div>
            </div>
          );
        })() : file ? (
          <div className="mt-3 mx-1 bg-[#FFF9E6] border border-[#FFE082] rounded-2xl p-3 flex items-center gap-3">
             <div className="w-6 h-6 bg-[#FBC02D] rounded-full flex items-center justify-center text-white font-semibold text-sm shadow-sm">!</div>
             <div className="flex flex-col">
                <span className="text-[9px] font-semibold text-[#A57900] tracking-wider uppercase">Clinical Disclaimer</span>
                <p className="text-[11px] font-medium text-[#5D4037] mt-0.5 leading-tight">
                  AI analysis provides educational insight. Metrics should be used with clinical reserve and expert verification.
                </p>
             </div>
          </div>
        ) : null}
      </main>

      {/* 3. DESNI SIDEBAR: PIPELINE CONTROL */}
      <aside className={`w-[260px] bg-white border-l border-[#E5E5E7] flex flex-col py-5 z-20 transition-all ${!isSidebarOpen && '-mr-[260px]'}`}>
        <div className="px-5 mb-5 flex flex-col gap-3">
          <span className="text-[11px] font-semibold text-slate-800 tracking-wider uppercase">Pipeline Control</span>
          <button 
            disabled={!sessionId || loading || isMetricsReady}
            onClick={processAll}
            className="w-full bg-[#0071E3] hover:bg-[#0077ED] disabled:bg-slate-100 disabled:text-slate-300 text-white py-3 rounded-xl font-semibold text-[13px] transition-all active:scale-[0.98] cursor-pointer"
          >
            {loading ? <Activity className="animate-spin mx-auto" size={16} /> : 'PROCESS FULL CASE'}
          </button>
        </div>

        {/* STEPPER */}
        <div className="flex-1 px-5 overflow-y-auto no-scrollbar">
          <div className="relative space-y-6 py-1">
            
            {/* Stage 0 */}
            <div className="relative flex gap-3">
              <div className="absolute top-2.5 bottom-[-24px] left-[10px] w-[1.5px] bg-[#34C759]" />
              <div className="relative z-10 w-5 h-5 bg-[#34C759] rounded-full flex items-center justify-center text-white">
                <CheckCircle2 size={12} strokeWidth={2.5} />
              </div>
              <div className="flex flex-col">
                <h3 className="text-[13px] font-semibold text-slate-900">0. STL Uploaded</h3>
                <span className="text-[10px] font-medium text-slate-600 mt-0.5 break-all pr-2">{stageData.upload?.filename || (file ? file.name : 'Waiting...')}</span>
                <span className="text-[10px] font-medium text-slate-600">{stageData.upload?.mesh_vertices?.length.toLocaleString() || '---'} Vertices</span>
              </div>
            </div>

         {/* Stage 1 */}
            <div className="relative flex gap-3">
              <div className={`absolute top-2.5 bottom-[-24px] left-[10px] w-[1.5px] ${stageData.detection ? 'bg-[#34C759]' : 'bg-slate-200'}`} />
              <div className={`relative z-10 w-5 h-5 rounded-full flex items-center justify-center transition-all ${
                stageData.detection ? 'bg-[#34C759] text-white' : 'bg-white border-2 border-slate-200 text-slate-300'
              }`}>
                {stageData.detection ? <CheckCircle2 size={12} strokeWidth={2.5} /> : <div className="w-1 h-1 rounded-full bg-current" />}
              </div>
              <div className="flex flex-col flex-1">
                <div className="flex items-center justify-between">
                   <h3 className={`text-[13px] font-semibold ${stageData.detection ? 'text-slate-900' : 'text-slate-400'}`}>1. Detection</h3>
                   {!stageData.detection && sessionId && (
                     <button onClick={() => runStage(1, 'stage1/detect', 'detection')} className="bg-[#0071E3] text-white px-2 py-0.5 rounded-md text-[9px] font-semibold uppercase hover:bg-blue-600 transition-all cursor-pointer">Run</button>
                   )}
                </div>
                <div className="text-[10px] font-medium text-slate-600 mt-0.5 uppercase">
                  {stageData.detection ? <>Jaw: {stageData.detection.jaw_type}<br/>Box: 12.0 × 14.0 × 12.0 mm</> : 'Waiting for scan...'}
                </div>
              </div>
            </div>

            {/* Stage 2 */}
            <div className="relative flex gap-3">
              <div className={`absolute top-2.5 bottom-[-24px] left-[10px] w-[1.5px] ${stageData.cropping ? 'bg-[#34C759]' : 'bg-slate-200'}`} />
              <div className={`relative z-10 w-5 h-5 rounded-full flex items-center justify-center transition-all ${
                stageData.cropping ? 'bg-[#34C759] text-white' : 'bg-white border-2 border-slate-200 text-slate-300'
              }`}>
                {stageData.cropping ? <CheckCircle2 size={12} strokeWidth={2.5} /> : <div className="w-1 h-1 rounded-full bg-current" />}
              </div>
              <div className="flex flex-col flex-1">
                 <div className="flex items-center justify-between">
                   <h3 className={`text-[13px] font-semibold ${stageData.cropping ? 'text-slate-900' : 'text-slate-400'}`}>2. Cropping</h3>
                   {!stageData.cropping && stageData.detection && (
                     <button onClick={() => runStage(2, 'stage2/crop', 'cropping')} className="bg-[#0071E3] text-white px-2 py-0.5 rounded-md text-[9px] font-semibold uppercase hover:bg-blue-600 transition-all cursor-pointer">Run</button>
                   )}
                </div>
                <div className="text-[10px] font-medium text-slate-600 mt-0.5 uppercase">
                  {stageData.cropping ? <>Retention: {stageData.cropping.retention_pct.toFixed(1)}%<br/>Points: {stageData.cropping.cropped_vertices.toLocaleString()}</> : 'Waiting for crop...'}
                </div>
              </div>
            </div>

            {/* Stage 3 */}
            <div className="relative flex gap-3">
              <div className={`absolute top-2.5 bottom-[-24px] left-[10px] w-[1.5px] ${stageData.segmentation ? 'bg-[#34C759]' : 'bg-slate-200'}`} />
              <div className={`relative z-10 w-5 h-5 rounded-full flex items-center justify-center transition-all ${
                stageData.segmentation ? 'bg-[#34C759] text-white' : 'bg-white border-2 border-slate-200 text-slate-300'
              }`}>
                {stageData.segmentation ? <CheckCircle2 size={12} strokeWidth={2.5} /> : <div className="w-1 h-1 rounded-full bg-current" />}
              </div>
              <div className="flex flex-col flex-1">
                 <div className="flex items-center justify-between">
                   <h3 className={`text-[13px] font-semibold ${stageData.segmentation ? 'text-slate-900' : 'text-slate-400'}`}>3. Segmentation</h3>
                   {!stageData.segmentation && stageData.cropping && (
                     <button onClick={() => runStage(3, 'stage3/segment', 'segmentation')} className="bg-[#0071E3] text-white px-2 py-0.5 rounded-md text-[9px] font-semibold uppercase hover:bg-blue-600 transition-all cursor-pointer">Run</button>
                   )}
                </div>
                <div className="text-[10px] font-medium text-slate-600 mt-0.5">
                  {stageData.segmentation ? (
                    <ul className="space-y-1">
                      {segmentationClasses.map((cls) => {
                        const count = stageData.segmentation.class_distribution?.[cls.count];
                        return count ? (
                          <li key={cls.name} className="flex items-center gap-2">
                            <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: cls.color }} />
                            <span>{cls.name}: {count.toLocaleString()} pts</span>
                          </li>
                        ) : null;
                      })}
                    </ul>
                  ) : 'Waiting for AI...'}
                </div>
              </div>
            </div>

            {/* Stage 4 */}
            <div className="relative flex gap-3 pb-4">
              <div className={`relative z-10 w-5 h-5 rounded-full flex items-center justify-center transition-all ${
                stageData.metrics ? 'bg-[#34C759] text-white' : 'bg-white border-2 border-slate-200 text-slate-300'
              }`}>
                {stageData.metrics ? <CheckCircle2 size={12} strokeWidth={2.5} /> : <div className="w-1 h-1 rounded-full bg-current" />}
              </div>
              <div className="flex flex-col flex-1">
                 <div className="flex items-center justify-between">
                   <h3 className={`text-[13px] font-semibold ${stageData.metrics ? 'text-slate-900' : 'text-slate-400'}`}>4. Metrics Analysis</h3>
                   {!stageData.metrics && stageData.segmentation && (
                     <button onClick={() => runStage(4, 'stage4/metrics', 'metrics')} className="bg-[#0071E3] text-white px-2 py-0.5 rounded-md text-[9px] font-semibold uppercase hover:bg-blue-600 transition-all cursor-pointer">Run</button>
                   )}
                </div>
                <div className="text-[10px] font-medium text-slate-600 mt-0.5">
                   {stageData.metrics ? 'Full diagnostic dataset ready.' : 'Waiting for results...'}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer Action */}
<div className="px-5 pt-4 border-t border-slate-200">
  <button 
    disabled={!isMetricsReady} 
    onClick={handleExportClick} // Ovo otvara modal
    className="w-full h-11 border border-[#0071E3] text-[#0071E3] hover:bg-[#0071E3]/5 disabled:opacity-25 rounded-xl font-semibold text-[13px] flex items-center justify-center gap-2 transition-all uppercase tracking-tight cursor-pointer"
  >
    <Download size={16} /> Export Report
  </button>
</div>
      </aside>
 {/* --- OVDE DODAJ MODAL --- */}
{isExporting && (
  <div className="fixed inset-0 z-[100] pointer-events-none animate-in fade-in duration-500">
    
    {/* 1. SIDEBAR BLURS */}
    <div className="absolute top-0 left-0 bottom-0 w-[90px] bg-white/40 backdrop-blur-md border-r border-slate-200 pointer-events-auto" />
    <div className="absolute top-0 right-0 bottom-0 w-[260px] bg-white/40 backdrop-blur-md border-l border-slate-200 pointer-events-auto" />

    {/* 2. BOTTOM BLUR (Preko disclaimera) */}
    <div className="absolute bottom-0 left-[90px] right-[260px] h-[105px] bg-[#F2F2F7]/60 backdrop-blur-md border-t border-slate-200 pointer-events-auto" />

    {/* 3. PLAVI OKVIR (Capture Zone) */}
    {/* bottom je smanjen sa 121px na 112px da bi se ivica spustila niže */}
    <div className="absolute top-4 left-[106px] right-[276px] bottom-[112px] rounded-[32px] ring-[3px] ring-[#0071E3] ring-offset-[3px] ring-offset-[#F2F2F7] shadow-[0_0_60px_-15px_rgba(0,113,227,0.4)]" />

    {/* 4. PLUTAJUĆA KONTROLA (Pill) */}
    <div className="absolute bottom-6 left-1/2 -translate-x-1/2 pointer-events-auto flex items-center h-[64px] bg-white/90 backdrop-blur-2xl border border-white px-6 rounded-[24px] shadow-[0_20px_50px_rgba(0,0,0,0.15)] animate-in slide-in-from-bottom-4 duration-500">
      <div className="flex flex-col justify-center pr-8 border-r border-slate-100 h-8">
        <span className="text-[9px] font-black text-[#0071E3] uppercase tracking-[1.5px] leading-none mb-1">Capture Mode</span>
        <span className="text-[12px] font-bold text-slate-700 leading-none">Position for PDF Report</span>
      </div>
      
      <div className="flex items-center gap-3 pl-6">
        <button 
          onClick={() => setIsExporting(false)}
          className="px-5 py-2.5 rounded-xl text-xs font-bold text-slate-500 hover:bg-slate-50 transition-all cursor-pointer active:scale-95"
        >
          Cancel
        </button>
        <button 
          onClick={finalizeExport}
          className="bg-[#0071E3] hover:bg-[#0077ED] text-white h-10 px-6 rounded-xl text-xs font-bold shadow-lg shadow-[#0071E3]/20 transition-all active:scale-95 flex items-center justify-center gap-2 cursor-pointer"
        >
          <Download size={14} strokeWidth={2.5} />
          GENERATE REPORT
        </button>
      </div>
    </div>

    {/* 5. POMERENI TOOLTIP (Gore desno) */}
    <div className="absolute top-7 right-[290px] bg-slate-900/90 text-white text-[9px] font-black px-4 py-2 rounded-2xl backdrop-blur-md shadow-xl tracking-[0.5px] uppercase border border-white/10 animate-in slide-in-from-right-4 duration-700">
      <div className="flex items-center gap-2">
        <div className="w-1.5 h-1.5 bg-[#0071E3] rounded-full animate-pulse" />
        View inside blue frame will be saved
      </div>
    </div>
  </div>
)}



    </div>
  );
}