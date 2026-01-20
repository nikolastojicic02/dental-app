import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Line, Html, PerspectiveCamera } from '@react-three/drei';
import { Suspense, useMemo, useRef, useState, useEffect } from 'react';
import * as THREE from 'three';
import { Maximize2, Minimize2, RotateCcw } from 'lucide-react';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

type Stage = 'upload' | 'detection' | 'cropping' | 'segmentation' | 'metrics';

interface ModelViewerProps {
  stageData: any;
  currentStage: Stage;
  selectedMetric?: string | null;
  isSidebarOpen: boolean;
  setIsSidebarOpen: (open: boolean) => void;
  glRef: React.RefObject<any>;
}

// --- POMOĆNA KOMPONENTA ZA KONTROLU KAMERE ---
function CameraController({ view, setView }: { view: string | null, setView: (v: string | null) => void }) {
  const { camera, controls } = useThree();
  
  useEffect(() => {
    if (!view || !controls) return;
    const orbit = controls as any;
    const dist = 16;

    switch (view) {
      case 'TOP':
        camera.position.set(0, 0, dist);
        break;
      case 'FRONT':
        camera.position.set(0, -dist, 2);
        break;
      case 'MESIAL':
        camera.position.set(-dist, 0, 2);
        break;
      case 'DISTAL':
        camera.position.set(dist, 0, 2);
        break;
      case 'RESET':
        camera.position.set(12, 12, 12);
        break;
    }
    
    orbit.target.set(0, 0, 0);
    orbit.update();
    
    const timer = setTimeout(() => setView(null), 100);
    return () => clearTimeout(timer);
  }, [view, camera, controls, setView]);

  return null;
}

// --- MESH KOMPONENTA ---
function ToothMesh({ vertices, faces, colors, isMetricView }: any) {
  const metricOpacity = 0.55;

  const geometry = useMemo(() => {
    if (!vertices || !faces || vertices.length === 0 || faces.length === 0) return new THREE.BufferGeometry();
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertices.flat()), 3));
    geom.setIndex(new THREE.BufferAttribute(new Uint32Array(faces.flat()), 1));
    
    if (colors && colors.length > 0) {
      const colorArray = new Float32Array(colors.length * 3);
      for (let i = 0; i < colors.length; i++) {
        colorArray[i * 3] = colors[i][0] / 255;
        colorArray[i * 3 + 1] = colors[i][1] / 255;
        colorArray[i * 3 + 2] = colors[i][2] / 255;
      }
      geom.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));
    }
    geom.computeVertexNormals();
    return geom;
  }, [vertices, faces, colors]);

  const hasColors = !!(colors && colors.length > 0);

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial 
        key={`mat-${hasColors}-${isMetricView}`}
        vertexColors={hasColors}
        color="#ffffff" 
        side={THREE.DoubleSide}
        transparent={isMetricView} 
        opacity={isMetricView ? metricOpacity : 1.0} 
        roughness={0.5}
        metalness={0.1}
        depthWrite={!isMetricView} 
      />
    </mesh>
  );
}

// ============================================================================
// --- METRIC VISUALIZERS  ---
// ============================================================================

function PulpalDepthVisualizer({ data }: { data: any }) {
  const { start, end, color, value_text } = data;
  const midPoint: [number, number, number] = [start[0], start[1], (start[2] + end[2]) / 2];
  const labelStyle = { color, fontWeight: '900', fontSize: '7px', textTransform: 'uppercase' as const, whiteSpace: 'nowrap' as const, pointerEvents: 'none' as const };
  return (
    <group>
      <Line points={[start, end]} color={color} lineWidth={5} />
      <mesh position={start}><sphereGeometry args={[0.045, 16, 16]} /><meshBasicMaterial color={color} /></mesh>
      <mesh position={end}><sphereGeometry args={[0.045, 16, 16]} /><meshBasicMaterial color={color} /></mesh>
      <mesh position={[start[0], start[1], end[2]]}>
        <planeGeometry args={[3.5, 3.5]} />
        <meshStandardMaterial color={color} transparent opacity={0.2} side={THREE.DoubleSide} />
      </mesh>
      <Html position={midPoint} center distanceFactor={15}><div style={labelStyle}>{value_text}</div></Html>
    </group>
  );
}

function BLRatioVisualizer({ data }: { data: any }) {
  const { isthmus_line, icd_line } = data;
  if (!isthmus_line) return null;
  const istMid: [number, number, number] = [(isthmus_line.start[0] + isthmus_line.end[0]) / 2, (isthmus_line.start[1] + isthmus_line.end[1]) / 2, (isthmus_line.start[2] + isthmus_line.end[2]) / 2];
  const icdMid: [number, number, number] = icd_line ? [(icd_line.start[0] + icd_line.end[0]) / 2, (icd_line.start[1] + icd_line.end[1]) / 2, (icd_line.start[2] + icd_line.end[2]) / 2] : [0,0,0];
  return (
    <group>
      <Line points={[isthmus_line.start, isthmus_line.end]} color="#ff0000" lineWidth={6} transparent opacity={0.8} />
      <Html position={istMid} center distanceFactor={15}><div style={{ color: '#ff0000', fontWeight: '900', fontSize: '9px' }}>{isthmus_line.value_text}</div></Html>
      {icd_line && (
        <>
          <Line points={[icd_line.start, icd_line.end]} color="#0000ff" lineWidth={4} dashed dashScale={0.5} dashSize={0.06} gapSize={0.04} transparent opacity={0.6} />
          <Html position={icdMid} center distanceFactor={15}><div style={{ color: '#0000ff', fontWeight: '900', fontSize: '9px' }}>{icd_line.value_text}</div></Html>
        </>
      )}
    </group>
  );
}

function AxialHeightVisualizer({ data }: { data: any }) {
  const { gingival_center, height_value, color } = data;
  if (!gingival_center || !height_value) return null;
  const start: [number, number, number] = [gingival_center[0], gingival_center[1], gingival_center[2]];
  const end: [number, number, number] = [gingival_center[0], gingival_center[1], gingival_center[2] + height_value];
  return (
    <group>
      <Line points={[start, end]} color={color} lineWidth={12} />
      <mesh position={start}><sphereGeometry args={[0.05, 16, 16]} /><meshBasicMaterial color={color} /></mesh>
      <mesh position={end}><sphereGeometry args={[0.05, 16, 16]} /><meshBasicMaterial color={color} /></mesh>
      <Html position={[(start[0]+end[0])/2, (start[1]+end[1])/2, (start[2]+end[2])/2]} center distanceFactor={15}>
        <div style={{ color, fontWeight: '900', fontSize: '9px' }}>{height_value.toFixed(2)} mm</div>
      </Html>
    </group>
  );
}

function WallTaperVisualizer({ data }: { data: any }) {
  const { measurement_lines } = data;
  return (
    <group>
      {measurement_lines?.map((line: any, idx: number) => (
        <group key={idx}>
          <Line points={[line.start, line.end]} color={line.color} lineWidth={12} />
          <Html position={line.end} center distanceFactor={15}>
            <div style={{ color: line.color, fontWeight: '900', fontSize: '8px' }}>{line.value_text}</div>
          </Html>
        </group>
      ))}
    </group>
  );
}

function MarginalRidgeVisualizer({ data }: { data: any }) {
  const { measurement_lines } = data;
  return (
    <group>
      {measurement_lines?.map((line: any, idx: number) => (
        <group key={idx}>
          <Line points={[line.start, line.end]} color={line.color} lineWidth={12} />
          <Html position={[(line.start[0]+line.end[0])/2, (line.start[1]+line.end[1])/2, (line.start[2]+line.end[2])/2]} center distanceFactor={15}>
            <div style={{ color: line.color, fontWeight: '900', fontSize: '8px' }}>{line.value_text}</div>
          </Html>
        </group>
      ))}
    </group>
  );
}

function getJetColor(v: number) {
  const max = 0.2; 
  let x = Math.min(Math.max(v / max, 0), 1);
  const r = x < 0.35 ? 0 : x < 0.66 ? (x - 0.35) / 0.31 : x < 0.89 ? 1 : 1 - (x - 0.89) / 0.11 * 0.5;
  const g = x < 0.125 ? 0 : x < 0.375 ? (x - 0.125) / 0.25 : x < 0.64 ? 1 : x < 0.91 ? 1 - (x - 0.64) / 0.27 : 0;
  const b = x < 0.11 ? 0.5 + x / 0.22 : x < 0.34 ? 1 : x < 0.6 ? 1 - (x - 0.34) / 0.26 : 0;
  return [r, g, b];
}

function WallSmoothnessVisualizer({ data }: { data: any }) {
  const { heatmap_vertices, heatmap_faces, heatmap_intensity, ghost_vertices, ghost_faces } = data;
  const heatmapGeom = useMemo(() => {
    if (!heatmap_vertices) return null;
    const geom = new THREE.BufferGeometry();
    const vertices = new Float32Array(heatmap_vertices.flat());
    geom.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    const colors = new Float32Array(vertices.length * 3);
    for (let i = 0; i < heatmap_intensity.length; i++) {
      const [r, g, b] = getJetColor(heatmap_intensity[i]);
      colors[i * 3] = r; colors[i * 3 + 1] = g; colors[i * 3 + 2] = b;
    }
    const filteredFaces: number[] = [];
    for (let i = 0; i < heatmap_faces.length; i++) {
      const [a, b, c] = heatmap_faces[i];
      if (heatmap_intensity[a] > 0 || heatmap_intensity[b] > 0 || heatmap_intensity[c] > 0) filteredFaces.push(a, b, c);
    }
    geom.setIndex(new THREE.BufferAttribute(new Uint32Array(filteredFaces), 1));
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geom.computeVertexNormals();
    return geom;
  }, [heatmap_vertices, heatmap_faces, heatmap_intensity]);

  const ghostGeom = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(ghost_vertices.flat()), 3));
    geom.setIndex(new THREE.BufferAttribute(new Uint32Array(ghost_faces.flat()), 1));
    geom.computeVertexNormals();
    return geom;
  }, [ghost_vertices, ghost_faces]);

  return (
    <group>
      <mesh geometry={ghostGeom}><meshStandardMaterial color="#ffffff" transparent opacity={0.05} side={THREE.DoubleSide} depthWrite={false} /></mesh>
      {heatmapGeom && <mesh geometry={heatmapGeom}><meshBasicMaterial vertexColors side={THREE.DoubleSide} polygonOffset polygonOffsetFactor={-1} /></mesh>}
    </group>
  );
}

function UndercutsVisualizer({ data }: { data: any }) {
  const { heatmap_vertices, heatmap_faces, heatmap_intensity, ghost_vertices, ghost_faces } = data;
  const heatmapGeom = useMemo(() => {
    if (!heatmap_vertices) return null;
    const geom = new THREE.BufferGeometry();
    const vertices = new Float32Array(heatmap_vertices.flat());
    geom.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    const colors = new Float32Array(vertices.length * 3);
    for (let i = 0; i < heatmap_intensity.length; i++) {
      const angle = heatmap_intensity[i];
      let r, g, b;
      if (angle < 92) { r = 0.2; g = 0.8; b = 0.2; }
      else if (angle < 100) { r = 1; g = 1; b = 0; }
      else { r = 0.9; g = 0.1; b = 0.1; }
      colors[i * 3] = r; colors[i * 3 + 1] = g; colors[i * 3 + 2] = b;
    }
    geom.setIndex(new THREE.BufferAttribute(new Uint32Array(heatmap_faces.flat()), 1));
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geom.computeVertexNormals();
    return geom;
  }, [heatmap_vertices, heatmap_faces, heatmap_intensity]);

  const ghostGeom = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(ghost_vertices.flat()), 3));
    geom.setIndex(new THREE.BufferAttribute(new Uint32Array(ghost_faces.flat()), 1));
    return geom;
  }, [ghost_vertices, ghost_faces]);

  return (
    <group>
      <mesh geometry={ghostGeom}><meshStandardMaterial color="#ffffff" transparent opacity={0.05} side={THREE.DoubleSide} depthWrite={false} /></mesh>
      {heatmapGeom && <mesh geometry={heatmapGeom}><meshBasicMaterial vertexColors side={THREE.DoubleSide} polygonOffset polygonOffsetFactor={-1} opacity={0.8} transparent /></mesh>}
    </group>
  );
}

function CuspUnderminingVisualizer({ data }: { data: any }) {
  const { measurement_lines } = data;
  return (
    <group>
      {measurement_lines?.map((line: any, idx: number) => (
        <group key={idx}>
          <Line points={[line.start, line.end]} color={line.color} lineWidth={11} />
          <mesh position={line.start}><sphereGeometry args={[0.06, 16, 16]} /><meshBasicMaterial color="black" /></mesh>
          <mesh position={line.end}><sphereGeometry args={[0.06, 16, 16]} /><meshBasicMaterial color="black" /></mesh>
          <Html position={[(line.start[0]+line.end[0])/2, (line.start[1]+line.end[1])/2, (line.start[2]+line.end[2])/2]} center distanceFactor={15}>
            <div style={{ color: line.color, fontWeight: '900', fontSize: '8px' }}>{line.value_text}</div>
          </Html>
        </group>
      ))}
    </group>
  );
}

function FloorFlatnessVisualizer({ data }: { data: any }) {
  const { heatmap_vertices, heatmap_faces, heatmap_intensity, ghost_vertices, ghost_faces } = data;
  const heatmapGeom = useMemo(() => {
    if (!heatmap_vertices) return null;
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(heatmap_vertices.flat()), 3));
    const colors = new Float32Array(heatmap_intensity.length * 3);
    for (let i = 0; i < heatmap_intensity.length; i++) {
      const val = Math.min(1, heatmap_intensity[i] / 0.4);
      let r, g, b;
      if (val < 0.5) { const t = val * 2; r = t; g = 1; b = 0; }
      else { const t = (val - 0.5) * 2; r = 1; g = 1 - t; b = 0; }
      colors[i * 3] = r; colors[i * 3 + 1] = g; colors[i * 3 + 2] = b;
    }
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const filteredFaces: number[] = [];
    for (let i = 0; i < heatmap_faces.length; i++) {
      const [a, b, c] = heatmap_faces[i];
      if (heatmap_intensity[a] > 0.001 || heatmap_intensity[b] > 0.001 || heatmap_intensity[c] > 0.001) filteredFaces.push(a, b, c);
    }
    geom.setIndex(new THREE.BufferAttribute(new Uint32Array(filteredFaces), 1));
    geom.computeVertexNormals();
    return geom;
  }, [heatmap_vertices, heatmap_faces, heatmap_intensity]);

  const ghostGeom = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(ghost_vertices.flat()), 3));
    geom.setIndex(new THREE.BufferAttribute(new Uint32Array(ghost_faces.flat()), 1));
    return geom;
  }, [ghost_vertices, ghost_faces]);

  return (
    <group>
      <mesh geometry={ghostGeom}><meshStandardMaterial color="#ffffff" transparent opacity={0.03} side={THREE.DoubleSide} depthWrite={false} /></mesh>
      {heatmapGeom && <mesh geometry={heatmapGeom}><meshBasicMaterial vertexColors side={THREE.DoubleSide} polygonOffset polygonOffsetFactor={-1} /></mesh>}
    </group>
  );
}

function BoundingBox({ corners }: { corners: number[][] }) {
  const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
  return (
    <group>
      {edges.map((edge, idx) => (
        <Line key={idx} points={[corners[edge[0]], corners[edge[1]]] as any} color="#0071E3" lineWidth={2} />
      ))}
    </group>
  );
}

// Dodaj ovo iznad komponente (izvan svih funkcija)
declare global {
  interface Window {
    dentalGL: any;
  }
}

// ============================================================================
// --- MAIN MODEL VIEWER COMPONENT ---
// ============================================================================

export default function ModelViewer({ stageData, currentStage, selectedMetric, isSidebarOpen, setIsSidebarOpen,  glRef }: ModelViewerProps) {
  const orbitRef = useRef<any>(null);
  const [activeView, setActiveView] = useState<string | null>(null);

  let meshData: any = null;
  let metricViz: any = null;
  let bboxCorners = null;

  if (currentStage === 'upload' && stageData.upload) {
    meshData = { vertices: stageData.upload.mesh_vertices, faces: stageData.upload.mesh_faces, colors: null };
  } else if (currentStage === 'detection' && stageData.detection) {
    meshData = { vertices: stageData.detection.mesh_vertices, faces: stageData.detection.mesh_faces, colors: null };
    bboxCorners = stageData.detection.bbox_corners;
  } else if (currentStage === 'cropping' && stageData.cropping) {
    meshData = { vertices: stageData.cropping.mesh_vertices, faces: stageData.cropping.mesh_faces, colors: null };
  } else if (stageData.segmentation) {
    const viz = (currentStage === 'metrics' && selectedMetric) ? stageData.metrics?.visualizations?.[selectedMetric] : null;
    meshData = {
      vertices: viz?.mesh_vertices || stageData.segmentation.mesh_vertices,
      faces: viz?.mesh_faces || stageData.segmentation.mesh_faces,
      colors: viz?.vertex_colors || stageData.segmentation.vertex_colors
    };
    metricViz = viz;
  }

  const isShowingMetric = currentStage === 'metrics' && !!selectedMetric;
  const showSegmentationLegend = stageData.segmentation && !isShowingMetric;
  const showHeatmapLegend = isShowingMetric && (selectedMetric === 'wall_smoothness' || selectedMetric === 'undercuts' || selectedMetric === 'floor_flatness');

  if (!meshData?.vertices) return null;

  const viewButtons = [
    { id: 'TOP', label: 'Occlusal' },
    { id: 'FRONT', label: 'Facial' },
    { id: 'MESIAL', label: 'Mesial' },
    { id: 'DISTAL', label: 'Distal' }
  ];

  const segmentationClasses = [
    { name: 'Intact Tooth', color: '#7E7E7E' },
    { name: 'Pulpal Floor', color: '#0000FE' },
    { name: 'Gingival Floor', color: '#00FEFE' },
    { name: 'F-L Walls', color: '#FE0000' },
    { name: 'Axial Wall', color: '#00FE00' },
    { name: 'Distal Wall', color: '#FEFE00' },
  ];

  return (
    <div className="h-full w-full relative">
      
      <Canvas 
        camera={{ position: [12, 12, 12], fov: 38 }} 
        gl={{ 
          antialias: true, 
          preserveDrawingBuffer: true 
        }} 
        onCreated={(state) => {
          // Proveravamo i setujemo ref na dva načina za svaki slučaj
          if (glRef) {
            glRef.current = state.gl;
            // Dodatno čuvamo i na window kao fallback
            (window as any).dentalGL = state.gl;
          }
          console.log("✅ WebGL Context successfully saved to Ref");
        }}
        dpr={[1, 2]}
      >
        <Suspense fallback={null}>
          <ambientLight intensity={0.9} />
          <directionalLight position={[10, 10, 20]} intensity={0.7} />
          <pointLight position={[-10, -10, 10]} intensity={0.4} />
          
          <ToothMesh vertices={meshData.vertices} faces={meshData.faces} colors={meshData.colors} isMetricView={isShowingMetric} />
          
          {bboxCorners && <BoundingBox corners={bboxCorners} />}
          
          {isShowingMetric && metricViz && (
            <>
              {selectedMetric === 'pulpal_depth' && metricViz.measurement && <PulpalDepthVisualizer data={metricViz.measurement} />}
              {selectedMetric === 'bl_ratio' && <BLRatioVisualizer data={metricViz} />}
              {selectedMetric === 'axial_height' && <AxialHeightVisualizer data={metricViz} />}
              {selectedMetric === 'marginal_ridge' && <MarginalRidgeVisualizer data={metricViz} />}
              {selectedMetric === 'wall_taper' && <WallTaperVisualizer data={metricViz} />}
              {selectedMetric === 'wall_smoothness' && <WallSmoothnessVisualizer data={metricViz} />}
              {selectedMetric === 'undercuts' && <UndercutsVisualizer data={metricViz} />}
              {selectedMetric === 'cusp_undermining' && <CuspUnderminingVisualizer data={metricViz} />}
              {selectedMetric === 'floor_flatness' && <FloorFlatnessVisualizer data={metricViz} />}
            </>
          )}

          <OrbitControls ref={orbitRef} makeDefault enableDamping dampingFactor={0.06} rotateSpeed={0.7} minDistance={5} maxDistance={40} />
          <CameraController view={activeView} setView={setActiveView} />
        </Suspense>
      </Canvas>

      {/* --- SEGMENTATION LEGEND (Gore desno, samo kada nema metrike) --- */}
      {showSegmentationLegend && (
        <div className="absolute top-4 right-4 bg-white/80 backdrop-blur-xl rounded-2xl p-4 border border-white/60 z-10">
          <span className="text-[9px] font-semibold text-slate-600 tracking-wider uppercase block mb-3">Segmentation Legend</span>
          <div className="space-y-2">
            {segmentationClasses.map((cls) => (
              <div key={cls.name} className="flex items-center gap-2.5 text-[10px] font-semibold text-slate-800">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: cls.color }} />
                <span>{cls.name}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* --- HEATMAP LEGENDE (Kontekstualne) --- */}
      {showHeatmapLegend && selectedMetric === 'wall_smoothness' && (
        <div className="absolute top-4 right-4 bg-white/80 backdrop-blur-xl rounded-2xl p-4 border border-white/60 z-10">
          <span className="text-[9px] font-semibold text-slate-600 tracking-wider uppercase block mb-3">Roughness (rad)</span>
          <div className="flex gap-3 items-center">
            <div className="w-3 h-32 rounded-full bg-gradient-to-t from-[#000080] via-[#00ff00] to-[#ff0000]" />
            <div className="flex flex-col justify-between h-32 text-[9px] font-semibold text-slate-700">
              <span>0.20</span>
              <span>0.10</span>
              <span>0.00</span>
            </div>
          </div>
        </div>
      )}

      {showHeatmapLegend && selectedMetric === 'undercuts' && (
        <div className="absolute top-4 right-4 bg-white/80 backdrop-blur-xl rounded-2xl p-4 border border-white/60 z-10 w-44">
          <span className="text-[9px] font-semibold text-slate-600 tracking-wider uppercase block mb-3">Undercut Angle</span>
          <div className="flex gap-4 items-center">
            <div className="w-3 h-36 rounded-full bg-gradient-to-t from-[#4ade80] via-[#facc15] to-[#ef4444]" />
            <div className="flex flex-col justify-between h-36 text-[9px] font-semibold text-slate-700">
              <div className="flex flex-col">
                <span className="text-[#ef4444]">110°+</span>
                <span className="opacity-50 text-[8px]">CRITICAL</span>
              </div>
              <div>100°</div>
              <div className="flex flex-col">
                <span>95°</span>
                <span className="opacity-50 text-[8px]">WARNING</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[#22c55e]">92°</span>
                <span className="opacity-50 text-[8px]">SAFE</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {showHeatmapLegend && selectedMetric === 'floor_flatness' && (
        <div className="absolute top-4 right-4 bg-white/80 backdrop-blur-xl rounded-2xl p-4 border border-white/60 z-10 w-44">
          <span className="text-[9px] font-semibold text-slate-600 tracking-wider uppercase block mb-3">Flatness (mm)</span>
          <div className="flex gap-4 items-center">
            <div className="w-3 h-36 rounded-full bg-gradient-to-t from-[#00ff00] via-[#ffff00] to-[#ff0000]" />
            <div className="flex flex-col justify-between h-36 text-[9px] font-semibold text-slate-700">
              <span className="text-[#ff0000]">0.40</span>
              <span>0.30</span>
              <span>0.20</span>
              <span>0.10</span>
              <span className="text-[#00aa00]">0.00</span>
            </div>
          </div>
        </div>
      )}

      {/* --- UNIFIED TOOLBAR (Dole desno) --- */}
      {/* --- UNIFIED TOOLBAR (Dole desno) --- */}
{/* --- UNIFIED TOOLBAR (Dole desno) - Kompaktna verzija --- */}
      <div className="absolute bottom-6 right-6 z-20 flex items-center gap-1 bg-white/80 backdrop-blur-lg border border-white/50 rounded-full p-1 shadow-xl">
        
        {/* Fullscreen / Sidebar Toggle */}
        <button 
          onClick={() => setIsSidebarOpen(!isSidebarOpen)} 
          className="w-8 h-8 rounded-full hover:bg-slate-200/60 flex items-center justify-center transition-colors cursor-pointer group"
          title={isSidebarOpen ? "Maximize View" : "Show Sidebar"}
        >
          {isSidebarOpen ? (
            <Maximize2 size={14} className="text-slate-600 group-hover:text-[#0071E3]" />
          ) : (
            <Minimize2 size={14} className="text-slate-600 group-hover:text-[#0071E3]" />
          )}
        </button>

        {/* Separator */}
        <div className="w-px h-4 bg-slate-300/60 mx-0.5" />

        {/* Camera Views */}
        <div className="flex items-center gap-0.5">
          {viewButtons.map((v) => {
            const isActive = activeView === v.id;
            return (
              <button
                key={v.id}
                onClick={() => setActiveView(v.id)}
                className={`px-2.5 py-1 rounded-full transition-all duration-200 cursor-pointer ${
                  isActive 
                    ? 'bg-[#0071E3] shadow-sm' 
                    : 'hover:bg-slate-200/60'
                }`}
              >
                <span className={`text-[9px] font-bold tracking-tight uppercase ${
                  isActive ? 'text-white' : 'text-slate-600'
                }`}>
                  {v.label}
                </span>
              </button>
            );
          })}
        </div>

        {/* Separator */}
        <div className="w-px h-4 bg-slate-300/60 mx-0.5" />

        {/* Reset Camera */}
        <button
          onClick={() => setActiveView('RESET')}
          className="w-8 h-8 rounded-full hover:bg-orange-50 flex items-center justify-center transition-colors cursor-pointer group"
          title="Reset Camera Position"
        >
          <RotateCcw size={14} className="text-slate-600 group-hover:text-orange-500" />
        </button>
      </div>
    </div>
  );
}