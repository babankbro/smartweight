"use client";

import { useState, useRef, Suspense } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { ArrowLeft, UploadCloud, Camera, Loader2, Save, Activity, Ruler, Banknote, ShieldCheck, CheckCircle2, X, AlertTriangle, Smartphone } from "lucide-react";

type AuxValue = { value: number; unit: string; typical_error: number | null; r2: number | null };

// Display order and wording for the MLP's non-weight outputs.
const AUX_FIELDS: {
  key: "height_cm" | "length_cm" | "age_years" | "ratio_lh";
  label: string;
  unit: string;
  fmt: (v: number) => string;
}[] = [
  { key: "height_cm", label: "ความสูง", unit: "ซม.", fmt: (v) => v.toFixed(0) },
  { key: "length_cm", label: "ความยาวลำตัว (L)", unit: "ซม.", fmt: (v) => v.toFixed(0) },
  { key: "age_years", label: "อายุ", unit: "ปี", fmt: (v) => v.toFixed(1) },
  { key: "ratio_lh", label: "สัดส่วน L / ความสูง", unit: "", fmt: (v) => v.toFixed(2) },
];

// R² from leakage-free validation: how much of the real variation the model
// explains. Below 0.3 the estimate is barely better than guessing the average.
const reliability = (r2: number | null) =>
  r2 == null
    ? { label: "ไม่ทราบ", cls: "bg-gray-100 text-gray-500" }
    : r2 >= 0.3
      ? { label: "พอใช้", cls: "bg-amber-100 text-amber-700" }
      : { label: "ต่ำ", cls: "bg-red-100 text-red-600" };

// Response shape of ai-api stage 2 (ai-api/app/weight.py -> WeightPredictor.predict)
type WeightResult = {
  kg: number;
  spread_kg: number;
  per_model: { svr: number; ridge: number; mlp: number };
  typical_error_kg: number | null;
  // Other outputs of the multi-task MLP (single model, not the 3-model ensemble)
  measurements?: Partial<Record<"height_cm" | "length_cm" | "age_years" | "ratio_lh", AuxValue>>;
  detector_found: boolean;
  model_version: string;
  inference_ms: number;
};

const HEADS: { key: keyof WeightResult["per_model"]; label: string; color: string }[] = [
  { key: "svr", label: "SVR", color: "#2563eb" },
  { key: "ridge", label: "Ridge", color: "#9333ea" },
  { key: "mlp", label: "Multi-task MLP", color: "#ea580c" },
];

function WeightPanel({ weight, zonesFound, note }: { weight: WeightResult; zonesFound: number; note: string }) {
  const err = weight.typical_error_kg ?? 0;
  const lo = weight.kg - err;
  const hi = weight.kg + err;
  // Shared axis for the range bar: wide enough for the band and every head.
  const values = Object.values(weight.per_model);
  const axisLo = Math.min(lo, ...values) - 10;
  const axisHi = Math.max(hi, ...values) + 10;
  const pos = (v: number) => `${((v - axisLo) / (axisHi - axisLo)) * 100}%`;
  const agree = weight.spread_kg < 15;

  const steps = [
    { label: "แยกส่วน 7 โซน", ok: zonesFound >= 5, detail: `${zonesFound}/7` },
    { label: "ตรวจจับตัวสัตว์", ok: weight.detector_found, detail: weight.detector_found ? "YOLO11n" : "ใช้ทั้งภาพ" },
    { label: "สกัดฟีเจอร์", ok: true, detail: "4 backbones" },
    { label: "รวมผล 3 โมเดล", ok: agree, detail: `±${weight.spread_kg.toFixed(0)} กก.` },
  ];

  return (
    <div className="bg-emerald-50 border border-emerald-200 rounded-2xl p-4 space-y-4">
      {/* Headline estimate */}
      <div className="text-center">
        <p className="text-[10px] text-emerald-700 font-bold mb-1">
          น้ำหนักประมาณ (AI)
          <span className="ml-1 px-1.5 py-0.5 rounded bg-amber-100 text-amber-700">ทดลอง</span>
        </p>
        <p className="text-4xl font-black text-emerald-700 leading-none">
          {weight.kg.toFixed(0)} <span className="text-base font-bold">กก.</span>
        </p>
        {weight.typical_error_kg != null && (
          <p className="text-xs text-emerald-800 mt-1">
            ช่วงที่น่าจะเป็น {lo.toFixed(0)}–{hi.toFixed(0)} กก. (คลาดเคลื่อนเฉลี่ย ±{err.toFixed(0)})
          </p>
        )}
      </div>

      {/* Pipeline steps */}
      <div className="grid grid-cols-4 gap-1.5">
        {steps.map((s, i) => (
          <div
            key={s.label}
            className={`rounded-xl px-1.5 py-2 text-center border ${
              s.ok ? "bg-white border-emerald-200" : "bg-amber-50 border-amber-200"
            }`}
          >
            <div className={`mx-auto mb-1 w-5 h-5 rounded-full text-[10px] font-bold flex items-center justify-center text-white ${
              s.ok ? "bg-emerald-500" : "bg-amber-500"
            }`}>
              {i + 1}
            </div>
            <p className="text-[9px] font-bold text-gray-700 leading-tight">{s.label}</p>
            <p className="text-[9px] text-gray-500 mt-0.5">{s.detail}</p>
          </div>
        ))}
      </div>

      {/* Each head on a shared axis, over the likely range */}
      <div className="bg-white rounded-xl border border-emerald-100 p-3">
        <p className="text-[10px] font-bold text-gray-600 mb-3">ผลจากแต่ละโมเดล (ค่าที่แสดงคือค่าเฉลี่ย)</p>
        <div className="relative h-6 mx-2">
          <div className="absolute top-1/2 -translate-y-1/2 inset-x-0 h-1 rounded bg-gray-200" />
          {err > 0 && (
            <div
              className="absolute top-1/2 -translate-y-1/2 h-3 rounded bg-emerald-200"
              style={{ left: pos(lo), width: `calc(${pos(hi)} - ${pos(lo)})` }}
            />
          )}
          {HEADS.map((h) => (
            <div
              key={h.key}
              title={`${h.label}: ${weight.per_model[h.key].toFixed(1)} กก.`}
              className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 w-3 h-3 rounded-full border-2 border-white shadow"
              style={{ left: pos(weight.per_model[h.key]), backgroundColor: h.color }}
            />
          ))}
          <div
            className="absolute top-0 bottom-0 -translate-x-1/2 w-0.5 bg-emerald-700"
            style={{ left: pos(weight.kg) }}
          />
        </div>
        <div className="mt-3 space-y-1">
          {HEADS.map((h) => (
            <div key={h.key} className="flex items-center justify-between text-[11px]">
              <span className="flex items-center gap-1.5 text-gray-600">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: h.color }} />
                {h.label}
              </span>
              <span className="font-bold text-gray-800 tabular-nums">{weight.per_model[h.key].toFixed(1)} กก.</span>
            </div>
          ))}
          <div className="flex items-center justify-between text-[11px] pt-1 border-t border-gray-100">
            <span className="flex items-center gap-1.5 text-emerald-700 font-bold">
              <span className="w-2.5 h-0.5 bg-emerald-700" /> เฉลี่ย (ensemble)
            </span>
            <span className="font-black text-emerald-700 tabular-nums">{weight.kg.toFixed(1)} กก.</span>
          </div>
        </div>
      </div>

      {!agree && (
        <div className="flex gap-2 items-start bg-amber-50 border border-amber-200 rounded-xl p-3">
          <AlertTriangle size={16} className="text-amber-600 shrink-0 mt-0.5" />
          <p className="text-xs text-amber-900">โมเดลทั้ง 3 ให้ค่าต่างกันมาก ผลนี้ไม่น่าเชื่อถือ ลองถ่ายด้านข้างให้เห็นเต็มตัวอีกครั้ง</p>
        </div>
      )}

      {/* Other multi-task regression outputs */}
      {weight.measurements && Object.keys(weight.measurements).length > 0 && (
        <div className="bg-white rounded-xl border border-emerald-100 p-3">
          <p className="text-[10px] font-bold text-gray-600">ค่าประมาณอื่นจาก Multi-task regression</p>
          <p className="text-[9px] text-gray-400 mb-2">ทำนายพร้อมน้ำหนักจากโมเดล MLP ตัวเดียว · ± คือค่าคลาดเคลื่อนเฉลี่ย</p>
          <div className="grid grid-cols-2 gap-2">
            {AUX_FIELDS.map((f) => {
              const m = weight.measurements?.[f.key];
              if (!m) return null;
              const rel = reliability(m.r2);
              return (
                <div key={f.key} className="rounded-lg bg-gray-50 border border-gray-100 p-2">
                  <div className="flex items-center justify-between gap-1">
                    <p className="text-[10px] text-gray-500 font-bold leading-tight">{f.label}</p>
                    <span className={`text-[8px] font-bold px-1 py-0.5 rounded ${rel.cls}`}>{rel.label}</span>
                  </div>
                  <p className="text-lg font-black text-gray-800 leading-tight mt-0.5 tabular-nums">
                    {f.fmt(m.value)} <span className="text-[10px] font-bold text-gray-500">{f.unit}</span>
                  </p>
                  {m.typical_error != null && (
                    <p className="text-[9px] text-gray-400 tabular-nums">
                      ±{f.fmt(m.typical_error)} {f.unit}
                      {m.r2 != null && <> · R² {m.r2.toFixed(2)}</>}
                    </p>
                  )}
                </div>
              );
            })}
          </div>
          <p className="text-[9px] text-gray-400 mt-2 leading-relaxed">
            ความน่าเชื่อถือ: &quot;พอใช้&quot; = อธิบายความแตกต่างจริงได้ราว 1 ใน 3 ·
            &quot;ต่ำ&quot; = แทบไม่ดีกว่าการเดาค่าเฉลี่ย ไม่ควรนำไปใช้ตัดสินใจ
          </p>
        </div>
      )}

      <p className="text-[10px] text-gray-500 leading-relaxed text-center">{note}</p>
      <p className="text-[9px] text-gray-400 text-center">
        {weight.model_version} · {(weight.inference_ms / 1000).toFixed(1)} วิ
      </p>
    </div>
  );
}

function ScanContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const targetAnimalId = searchParams.get("animalId");
  const targetAnimalName = searchParams.get("animalName");
  
  const [showGuidelines, setShowGuidelines] = useState(true);
  const [showSuccess, setShowSuccess] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [base64Image, setBase64Image] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  const [realGirth, setRealGirth] = useState("");
  const [animalName, setAnimalName] = useState(targetAnimalName || targetAnimalId || "");
  const [isSavingToFarm, setIsSavingToFarm] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [selectedType, setSelectedType] = useState("โคเนื้อ");

  const [showCamera, setShowCamera] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const startCamera = async () => {
    setShowCamera(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      // Force landscape if supported
      if (screen.orientation && (screen.orientation as any).lock) {
        try {
          await (screen.orientation as any).lock('landscape');
        } catch (e) {
          console.log("Orientation lock failed", e);
        }
      }
    } catch (err) {
      console.error("Camera access error:", err);
      alert("ไม่สามารถเข้าถึงกล้องได้ กรุณาตรวจสอบสิทธิ์การเข้าถึง");
      setShowCamera(false);
    }
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const MAX_WIDTH = 600;
      let targetWidth = video.videoWidth;
      let targetHeight = video.videoHeight;
      
      if (targetWidth > MAX_WIDTH) {
        targetHeight = (MAX_WIDTH / targetWidth) * targetHeight;
        targetWidth = MAX_WIDTH;
      }
      
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, targetWidth, targetHeight);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.6);
        setBase64Image(dataUrl);
        setSelectedImage(dataUrl);
        stopCamera();
        setResult(null);
        setAnalyzeError(null);
        setRealGirth("");
        
      }
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setShowCamera(false);
    if (screen.orientation && screen.orientation.unlock) {
      screen.orientation.unlock();
    }
  };

  const handleSave = async () => {
    if (isSavingToFarm && result && !showSuccess) {
      setIsSaving(true);
      try {
        const animalIdToSave = targetAnimalId || `M${Date.now()}`;
        const finalName = animalName || (targetAnimalId ? "ไม่มีชื่อ" : `${selectedType} #${animalIdToSave}`);
        
        // Fetch existing attempt count if it's an existing animal
        let nextAttempt = 1;
        if (targetAnimalId) {
          const res = await fetch(`/api/animals/${targetAnimalId}`, { cache: 'no-store' });
          if (res.ok) {
            const data = await res.json();
            if (data.measurements) {
              nextAttempt = data.measurements.length + 1;
            }
          }
        }

        const newMeasurement = {
          attempt: nextAttempt,
          date: "วันนี้",
          time: new Date().toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit' }) + " น.",
          // Experimental image-only estimate from ai-api stage 2. Buffalo stay
          // null: the weight model was trained on cattle only.
          aiWeight: aiWeightKg,
          aiGirth: null,
          aiHeight: aiWeightKg !== null ? (result.weight.measurements?.height_cm?.value ?? null) : null,
          realGirth: realGirth ? Number(realGirth) : null,
          realHeight: null,
          scanImage: base64Image || null,
          timestamp: Date.now()
        };

        const res = await fetch('/api/animals', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            id: animalIdToSave,
            name: finalName,
            type: selectedType,
            measurements: [newMeasurement]
          })
        });
        
        if (!res.ok) {
          const errorData = await res.json();
          alert(`เกิดข้อผิดพลาดในการบันทึก: ${errorData.error || res.statusText}`);
          setIsSaving(false);
          return;
        }
        
        // Show success modal ONLY after successful save
        setShowSuccess(true);
      } catch (err: any) {
        console.error(err);
        alert(`เกิดข้อผิดพลาดในการเชื่อมต่อ: ${err.message}`);
      } finally {
        setIsSaving(false);
      }
    } else if (!isSavingToFarm) {
      setShowSuccess(true);
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const imageUrl = URL.createObjectURL(file);
      setSelectedImage(imageUrl);
      
      const reader = new FileReader();
      reader.onloadend = () => {
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement('canvas');
          const MAX_WIDTH = 600;
          let targetWidth = img.width;
          let targetHeight = img.height;
          
          if (targetWidth > MAX_WIDTH) {
            targetHeight = (MAX_WIDTH / targetWidth) * targetHeight;
            targetWidth = MAX_WIDTH;
          }
          
          canvas.width = targetWidth;
          canvas.height = targetHeight;
          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
            setBase64Image(canvas.toDataURL('image/jpeg', 0.6));
          }
        };
        img.src = reader.result as string;
      };
      reader.readAsDataURL(file);

      setResult(null); // Reset previous result
      setAnalyzeError(null);
      setRealGirth("");
      
    }
  };

  const handleAnalyze = async () => {
    if (!base64Image) return;
    setIsProcessing(true);
    setAnalyzeError(null);
    try {
      const blob = await (await fetch(base64Image)).blob();
      const fd = new FormData();
      fd.append("file", blob, "scan.jpg");

      const res = await fetch("/api/analyze", { method: "POST", body: fd });
      const json = await res.json();

      if (!res.ok) {
        setAnalyzeError(json.error || "วิเคราะห์ภาพไม่สำเร็จ");
        setResult(null);
        return;
      }

      setResult({ ...json, animalType: selectedType });
    } catch (err: any) {
      setAnalyzeError(`เชื่อมต่อไม่สำเร็จ: ${err.message}`);
      setResult(null);
    } finally {
      setIsProcessing(false);
    }
  };

  const aiWeightKg: number | null =
    result?.weight && result.animalType !== "กระบือ" ? result.weight.kg : null;

  // Heart girth formula (Schaeffer): W(kg) = girth(cm)^2 x length(cm) / 10840.
  // Without a measured body length we fall back to the girth-only rule of thumb
  // that Thai extension officers use. This is the baseline the AI has to beat -
  // it is NOT the AI's output.
  const calculatedWeight = realGirth
    ? (Math.pow(Number(realGirth), 2) / 50).toFixed(1)
    : "-";

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-blue-100 text-gray-800 flex flex-col relative overflow-hidden">
      
      {/* Decorative background elements */}
      <div className="absolute top-[-20%] right-[-10%] w-96 h-96 bg-blue-200 rounded-full mix-blend-multiply filter blur-3xl opacity-50"></div>
      
      {/* Guidelines Modal */}
      {showGuidelines && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-300">
          <div className="bg-white rounded-[32px] w-full max-w-sm overflow-hidden shadow-2xl animate-in zoom-in-95 duration-300 relative">
            
            {/* Modal Header */}
            <div className="bg-gradient-to-r from-blue-900 to-blue-600 p-5 pt-8 text-center relative">
              <button 
                onClick={() => setShowGuidelines(false)}
                className="absolute top-4 right-4 text-white/80 hover:text-white p-1 bg-white/10 rounded-full transition-colors"
              >
                <X size={20} />
              </button>
              <h2 className="text-white font-bold text-xl mb-1">คำแนะนำในการถ่ายรูป<br/>และเลือกรูปภาพ</h2>
              <p className="text-blue-100 text-sm">ขั้นตอนการถ่ายภาพให้ AI วิเคราะห์ได้แม่นยำ</p>
            </div>

            {/* Modal Content */}
            <div className="p-6 space-y-5">
              <ul className="space-y-4">
                <li className="flex gap-3">
                  <div className="w-6 h-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold text-sm shrink-0">1</div>
                  <p className="text-sm text-gray-700 leading-relaxed"><strong className="text-gray-900">รูปแบบภาพถ่าย:</strong> ต้องเป็นภาพถ่ายแนวนอน</p>
                </li>
                <li className="flex gap-3">
                  <div className="w-6 h-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold text-sm shrink-0">2</div>
                  <p className="text-sm text-gray-700 leading-relaxed"><strong className="text-gray-900">เว้นระยะ:</strong> ต้องเว้นระยะให้ห่างจากตัวสัตว์ประมาณ 2-3 เมตร</p>
                </li>
                <li className="flex gap-3">
                  <div className="w-6 h-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold text-sm shrink-0">3</div>
                  <p className="text-sm text-gray-700 leading-relaxed"><strong className="text-gray-900">เช็กจุดโฟกัส:</strong> ตรวจสอบในจอว่าเห็นโค/กระบือ ให้เต็มตัวอย่าให้หัวหรือหางหลุดจากรูปภาพ</p>
                </li>
                <li className="flex gap-3">
                  <div className="w-6 h-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold text-sm shrink-0">4</div>
                  <p className="text-sm text-gray-700 leading-relaxed"><strong className="text-gray-900">ถือกล้องตรง:</strong> ถือสมาร์ตโฟนให้ตั้งฉากกับพื้น เล็งกลางลำตัวสัตว์ แล้วกดถ่าย</p>
                </li>
              </ul>

              {/* Warning Alert */}
              <div className="bg-red-50 border border-red-200 p-3 rounded-2xl flex items-start gap-3 mt-2">
                <AlertTriangle size={20} className="text-red-600 shrink-0 mt-0.5" />
                <p className="text-sm font-bold text-red-600 leading-tight">
                  ข้อควรระวัง: ภาพที่ถ่ายต้องเป็นภาพถ่ายแนวนอน
                </p>
              </div>

              <button 
                onClick={() => setShowGuidelines(false)}
                className="w-full py-3.5 bg-blue-700 text-white font-bold rounded-xl hover:bg-blue-800 transition-colors shadow-md mt-2"
              >
                รับทราบ และเริ่มใช้งาน
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Success Modal */}
      {showSuccess && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-300">
          <div className="bg-white rounded-[32px] w-full max-w-sm overflow-hidden shadow-2xl animate-in zoom-in-95 duration-300 p-8 flex flex-col items-center text-center relative">
            
            <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mb-5 relative">
              <div className="absolute inset-0 bg-blue-400 rounded-full animate-ping opacity-20"></div>
              <CheckCircle2 size={40} className="text-[#1e3a8a]" />
            </div>
            
            <h2 className="text-2xl font-black text-[#1e3a8a] mb-2">บันทึกสำเร็จ!</h2>
            <p className="text-gray-500 mb-8">ข้อมูลการประเมินถูกจัดเก็บเรียบร้อยแล้ว</p>
            
            <div className="w-full space-y-3">
              <button 
                onClick={() => { 
                  setShowSuccess(false); 
                  window.location.href = targetAnimalId ? `/animals/${targetAnimalId}` : '/dashboard';
                }}
                className="w-full py-4 bg-[#1e3a8a] text-white font-bold rounded-2xl hover:bg-[#1d4ed8] transition-all shadow-md"
              >
                {targetAnimalId ? "กลับไปหน้าโปรไฟล์สัตว์" : "กลับหน้าแรก"}
              </button>
              <button 
                onClick={() => { 
                  setShowSuccess(false); 
                  window.location.href = '/history';
                }}
                className="w-full py-4 bg-gray-100 text-[#1e3a8a] font-bold rounded-2xl hover:bg-gray-200 transition-all"
              >
                ดูประวัติการชั่งรวม
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Header */}
      <header className="bg-white/40 backdrop-blur-md text-[#1e3a8a] p-4 flex items-center justify-between z-10 border-b border-white/50">
        <div className="flex items-center space-x-4">
          <Link href="/dashboard" className="p-2.5 bg-white/60 shadow-sm rounded-full hover:bg-white transition-all active:scale-95 text-[#1d4ed8]">
            <ArrowLeft size={22} />
          </Link>
          <h1 className="text-xl font-bold tracking-wide">วิเคราะห์ภาพวัว</h1>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 p-5 flex flex-col items-center z-10 overflow-y-auto pb-10">
        
        {/* Upload Area */}
        <div className="w-full max-w-sm glass-panel rounded-3xl overflow-hidden flex flex-col transition-all duration-300">
          <div className="p-4 bg-white/40 border-b border-white/50 flex justify-between items-center">
            <h2 className="font-bold text-[#1e3a8a] flex items-center gap-2">
              <Camera size={18} />
              <span>ภาพสำหรับวิเคราะห์</span>
            </h2>
            {selectedImage && !isProcessing && (
               <button onClick={() => { setSelectedImage(null); setResult(null); setAnalyzeError(null); setRealGirth("");  }} className="text-red-500 text-sm font-semibold hover:underline">เปลี่ยนรูป</button>
            )}
          </div>

          {/* Animal Type Selection */}
          {!selectedImage && (
            <div className="px-5 pt-5 pb-1 bg-gradient-to-b from-white/40 to-white/30">
              <label className="block text-sm font-bold text-[#1e3a8a] mb-2">เลือกประเภทสัตว์</label>
              <div className="flex bg-white rounded-xl p-1 border border-blue-100 shadow-sm">
                <button 
                  onClick={() => setSelectedType('โคเนื้อ')}
                  className={`flex-1 py-2 text-sm font-bold rounded-lg transition-all ${selectedType === 'โคเนื้อ' ? 'bg-[#1e3a8a] text-white shadow-md' : 'text-gray-500 hover:bg-gray-50'}`}
                >
                  โคเนื้อ
                </button>
                <button 
                  onClick={() => setSelectedType('กระบือ')}
                  className={`flex-1 py-2 text-sm font-bold rounded-lg transition-all ${selectedType === 'กระบือ' ? 'bg-[#1e3a8a] text-white shadow-md' : 'text-gray-500 hover:bg-gray-50'}`}
                >
                  กระบือ
                </button>
              </div>
            </div>
          )}

          <div className="flex-1 min-h-[320px] flex flex-col items-center justify-center p-5 relative bg-gradient-to-b from-white/30 to-transparent">
            {selectedImage ? (
              <div className="relative w-full h-full rounded-2xl overflow-hidden shadow-inner border border-white/50">
                <img src={selectedImage} alt="Cow" className="w-full h-full object-cover" />
                
                {/* Scanning Animation Overlay */}
                {isProcessing && (
                  <div className="absolute inset-0 bg-[#1e3a8a]/80 backdrop-blur-sm flex flex-col items-center justify-center text-white transition-all duration-300">
                    <Loader2 size={48} className="animate-spin text-blue-400 mb-4" />
                    <p className="font-bold text-lg animate-pulse tracking-wide">กำลังประมวลผล AI...</p>
                    <p className="text-sm font-light text-blue-200 mt-2">จำแนกประเภทและสัดส่วน</p>
                    
                    {/* Scanner Line */}
                    <div className="absolute top-0 left-0 w-full h-1 bg-blue-400 shadow-[0_0_15px_#34d399] animate-[scan_2s_ease-in-out_infinite]"></div>
                  </div>
                )}
              </div>
            ) : (
              <div className="w-full flex gap-3 h-full">
                {/* Upload from Gallery */}
                <label className="flex-1 flex flex-col items-center justify-center h-full border-2 border-dashed border-[#1d4ed8]/40 rounded-2xl cursor-pointer hover:bg-white/50 hover:border-[#1d4ed8] transition-all bg-white/30 group">
                  <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-3 text-[#1d4ed8] group-hover:scale-110 transition-transform shadow-sm">
                    <UploadCloud size={32} />
                  </div>
                  <span className="text-[#1e3a8a] font-bold">อัปโหลดรูปภาพ</span>
                  <input type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
                </label>

                {/* Take Photo from Camera */}
                <button 
                  onClick={startCamera}
                  className="flex-1 flex flex-col items-center justify-center h-full border-2 border-dashed border-[#1d4ed8]/40 rounded-2xl cursor-pointer hover:bg-white/50 hover:border-[#1d4ed8] transition-all bg-white/30 group"
                >
                  <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-3 text-[#1d4ed8] group-hover:scale-110 transition-transform shadow-sm">
                    <Camera size={32} />
                  </div>
                  <span className="text-[#1e3a8a] font-bold">ถ่ายภาพจากกล้อง</span>
                </button>
              </div>
            )}
          </div>

          {/* Action Button */}
          <div className="p-5 bg-white/40 border-t border-white/50">
            <button 
              onClick={handleAnalyze}
              disabled={!selectedImage || isProcessing}
              className={`w-full py-4 rounded-2xl font-bold transition-all flex items-center justify-center gap-2 text-lg shadow-lg ${
                !selectedImage || isProcessing 
                  ? 'bg-gray-200/50 text-gray-400 cursor-not-allowed border border-gray-300/50' 
                  : 'bg-gradient-to-r from-[#1e3a8a] to-[#1d4ed8] text-white hover:opacity-95 active:scale-[0.98]'
              }`}
            >
              {isProcessing ? (
                <>
                  <Loader2 size={22} className="animate-spin" />
                  กำลังคำนวณ...
                </>
              ) : (
                <>
                  <Activity size={22} />
                  เริ่มวิเคราะห์ข้อมูลด้วย AI
                </>
              )}
            </button>
          </div>
        </div>

        {/* Analysis failed - show the service's own reason, not a generic one */}
        {analyzeError && !result && (
          <div className="w-full max-w-sm mt-6 bg-red-50 border border-red-200 rounded-2xl p-4 flex gap-3 items-start animate-in fade-in slide-in-from-bottom-4">
            <AlertTriangle size={20} className="text-red-500 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-bold text-red-700 mb-1">วิเคราะห์ภาพไม่สำเร็จ</p>
              <p className="text-xs text-red-600 leading-relaxed">{analyzeError}</p>
            </div>
          </div>
        )}

        {/* Result Area */}
        {result && (
          <div className="w-full max-w-sm mt-6 glass-panel rounded-3xl overflow-hidden animate-in fade-in slide-in-from-bottom-8 duration-700 shadow-xl border border-blue-300 relative">
             <div className="bg-gradient-to-r from-blue-500 to-teal-500 text-white p-4 pb-8 text-center font-bold flex flex-col items-center justify-center shadow-inner relative">
               <div className="flex items-center gap-2">
                 <ShieldCheck size={20} />
                 แยกส่วนตัวสัตว์สำเร็จ
               </div>
               <p className="text-[11px] font-normal text-blue-50 mt-0.5">
                 พบ {result.quality.zones_found}/7 โซน · ใช้เวลา {(result.inference_ms / 1000).toFixed(1)} วิ
               </p>
             </div>
             
             {/* Badge Overlapping Header and Content */}
             <div className="absolute top-11 left-1/2 -translate-x-1/2 bg-white px-6 py-2 rounded-full shadow-md border-2 border-blue-400 flex items-center gap-2 text-[#1e3a8a] font-bold z-10 w-max">
               <span className="text-xs text-gray-500">ประเภท:</span> 
               <span className="text-lg">{result.animalType}</span>
             </div>

             <div className="p-6 pt-10 space-y-5">
                {/* Segmentation overlay - lets the user judge whether the AI
                    actually outlined the right animal. */}
                {result.overlay_png_base64 && (
                  <img
                    src={result.overlay_png_base64}
                    alt="ผลการแยกส่วนตัวสัตว์"
                    className="w-full rounded-2xl border border-gray-200 shadow-sm"
                  />
                )}

                {/* Warnings from the model's own quality gate */}
                {(result.quality.reasons.length > 0 || result.quality.notes.length > 0) && (
                  <div className="space-y-2">
                    {result.quality.reasons.map((r: string, i: number) => (
                      <div key={`r${i}`} className="flex gap-2 items-start bg-amber-50 border border-amber-200 rounded-xl p-3">
                        <AlertTriangle size={16} className="text-amber-600 shrink-0 mt-0.5" />
                        <p className="text-xs text-amber-900">{r}</p>
                      </div>
                    ))}
                    {result.quality.notes.map((n: string, i: number) => (
                      <p key={`n${i}`} className="text-[11px] text-gray-500 px-1">{n}</p>
                    ))}
                  </div>
                )}

                {/* Experimental weight - shown with its typical error so it
                    is not mistaken for a scale reading. */}
                {aiWeightKg !== null ? (
                  <WeightPanel weight={result.weight} zonesFound={result.quality.zones_found} note={result.quality.weight_note} />
                ) : (
                  <div className="bg-gray-100 border border-gray-200 rounded-2xl p-4 text-center">
                    <p className="text-[10px] text-gray-500 font-bold mb-1">น้ำหนัก (AI)</p>
                    <p className="text-lg font-bold text-gray-400">ยังไม่พร้อมใช้งาน</p>
                    <p className="text-[10px] text-gray-500 mt-1 leading-relaxed">
                      {result.weight && result.animalType === "กระบือ"
                        ? "โมเดลน้ำหนักเทรนจากโคเท่านั้น ยังไม่รองรับกระบือ"
                        : result.quality.weight_note}
                    </p>
                  </div>
                )}

                {/* Pixel measurements - honest about their unit */}
                <div className="grid grid-cols-3 gap-3">
                  <div className="bg-white p-3 rounded-2xl border border-gray-100 text-center shadow-sm flex flex-col justify-center">
                    <p className="text-[10px] text-gray-500 font-bold mb-1">พื้นที่ลำตัว</p>
                    <p className="text-lg font-bold text-[#1e3a8a]">
                      {(result.total.area_px / 1000).toFixed(1)}k
                    </p>
                    <p className="text-[10px] text-gray-400">PX²</p>
                  </div>
                  <div className="bg-blue-50 p-3 rounded-2xl border border-blue-100 text-center shadow-sm flex flex-col justify-center">
                    <p className="text-[10px] text-blue-700 font-bold mb-1">ความยาว</p>
                    <p className="text-lg font-bold text-blue-700">{result.total.length_px}</p>
                    <p className="text-[10px] text-blue-600/70">PX</p>
                  </div>
                  <div className="bg-white p-3 rounded-2xl border border-gray-100 text-center shadow-sm flex flex-col justify-center">
                    <p className="text-[10px] text-gray-500 font-bold mb-1">ความลึกลำตัว</p>
                    <p className="text-lg font-bold text-[#1e3a8a]">{result.total.height_px}</p>
                    <p className="text-[10px] text-gray-400">PX</p>
                  </div>
                </div>

                {/* Per-zone shares. These ratios do not depend on how far away
                    the camera was, so they are the one number here that is
                    already meaningful without a scale reference. */}
                <div className="bg-white rounded-2xl border border-gray-100 p-4 shadow-sm">
                  <p className="text-xs font-bold text-[#1e3a8a] mb-3">สัดส่วนพื้นที่รายโซน</p>
                  <div className="space-y-2">
                    {result.zones.map((z: any) => (
                      <div key={z.id} className="flex items-center gap-2">
                        <span
                          className="w-3 h-3 rounded-sm shrink-0"
                          style={{ backgroundColor: `rgb(${z.color.join(",")})`, opacity: z.found ? 1 : 0.25 }}
                        />
                        <span className={`text-[11px] flex-1 truncate ${z.found ? "text-gray-700" : "text-gray-300"}`}>
                          {z.name_th}
                        </span>
                        {z.found ? (
                          <>
                            <div className="w-16 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                              <div
                                className="h-full rounded-full"
                                style={{ width: `${Math.min(z.share * 250, 100)}%`, backgroundColor: `rgb(${z.color.join(",")})` }}
                              />
                            </div>
                            <span className="text-[11px] font-bold text-gray-600 w-10 text-right">
                              {(z.share * 100).toFixed(1)}%
                            </span>
                          </>
                        ) : (
                          <span className="text-[11px] text-gray-300 w-10 text-right">ไม่พบ</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Compare Real Girth Input */}
                <div className="bg-blue-50 p-5 rounded-2xl border border-blue-100 space-y-4">
                  <div>
                    <label className="block text-sm font-bold text-[#1e3a8a] mb-2 flex items-center gap-2">
                      <Ruler size={16} /> ใส่ค่ารอบอกวัดจริง (ซม.)
                    </label>
                    <input 
                      type="number"
                      value={realGirth}
                      onChange={(e) => setRealGirth(e.target.value)}
                      placeholder="เช่น 150"
                      className="w-full px-4 py-3 bg-white border border-gray-200 rounded-xl focus:ring-2 focus:ring-[#1d4ed8] focus:border-transparent outline-none transition-all font-bold text-[#1e3a8a]"
                    />
                    
                    {realGirth && (
                      <div className="mt-4 pt-4 border-t border-blue-200 space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-gray-600">น้ำหนักจากสูตรรอบอก:</span>
                          <span className="font-bold text-[#1e3a8a]">{calculatedWeight} kg</span>
                        </div>
                        <p className="text-[11px] text-gray-500 leading-relaxed">
                          ค่านี้มาจากสูตรสายวัดรอบอก ไม่ใช่ผลจาก AI · เมื่อโมเดลขั้นที่ 2
                          พร้อมใช้งาน ระบบจะแสดงส่วนต่างระหว่างสองวิธีให้เปรียบเทียบ
                        </p>
                      </div>
                    )}
                  </div>
                  
                  <div className="pt-4 border-t border-blue-200">
                    <label className="block text-sm font-bold text-[#1e3a8a] mb-3">
                      ความต้องการบันทึกข้อมูล
                    </label>
                    
                    <div className="flex bg-white rounded-xl p-1 border border-gray-200 mb-4 shadow-sm">
                      <button 
                        onClick={() => setIsSavingToFarm(true)}
                        className={`flex-1 py-2 text-sm font-bold rounded-lg transition-colors ${isSavingToFarm ? 'bg-[#1e3a8a] text-white' : 'text-gray-500 hover:bg-gray-50'}`}
                      >
                        เก็บเข้าฟาร์ม
                      </button>
                      <button 
                        onClick={() => setIsSavingToFarm(false)}
                        className={`flex-1 py-2 text-sm font-bold rounded-lg transition-colors ${!isSavingToFarm ? 'bg-gray-500 text-white' : 'text-gray-500 hover:bg-gray-50'}`}
                      >
                        ไม่เก็บเข้าฟาร์ม
                      </button>
                    </div>

                    <div className="animate-in fade-in slide-in-from-top-2 duration-300">
                      {targetAnimalId ? (
                        <div className="w-full px-4 py-3 bg-blue-50 border border-blue-200 rounded-xl flex items-center justify-between text-sm">
                          <span className="font-bold text-[#1e3a8a]">อัปเดตข้อมูลสัตว์: #{targetAnimalId}</span>
                          <CheckCircle2 size={18} className="text-blue-500" />
                        </div>
                      ) : (
                        <input 
                          type="text"
                          value={animalName}
                          onChange={(e) => setAnimalName(e.target.value)}
                          placeholder="ชื่อ/รหัสสัตว์ (ไม่บังคับ) เช่น เจ้าบุญรอด"
                          className="w-full px-4 py-3 bg-white border border-gray-200 rounded-xl focus:ring-2 focus:ring-[#1d4ed8] focus:border-transparent outline-none transition-all text-sm"
                        />
                      )}
                    </div>
                  </div>
                </div>

                {/* Save Button */}
                {isSavingToFarm ? (
                  <button 
                    onClick={handleSave} 
                    disabled={isSaving}
                    className={`w-full py-4 text-white font-bold text-lg rounded-2xl transition-all shadow-md flex items-center justify-center gap-2 mt-4 ${isSaving ? 'bg-[#1e3a8a]/70 cursor-not-allowed' : 'bg-[#1e3a8a] hover:bg-[#1d4ed8] active:scale-[0.98]'}`}
                  >
                    {isSaving ? (
                      <>
                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                        กำลังบันทึก... กรุณารอสักครู่
                      </>
                    ) : (
                      <>
                        <Save size={20} />
                        บันทึกข้อมูลลงระบบฟาร์ม
                      </>
                    )}
                  </button>
                ) : (
                  <button onClick={handleSave} className="w-full py-4 bg-gray-600 text-white font-bold text-lg rounded-2xl hover:bg-gray-700 transition-all shadow-md active:scale-[0.98] flex items-center justify-center gap-2 mt-4">
                    <Save size={20} />
                    บันทึกเฉพาะประวัติ
                  </button>
                )}
             </div>
          </div>
        )}

      </div>

      {/* Custom Camera View */}
      {showCamera && (
        <div className="fixed inset-0 z-[100] bg-black flex flex-col">
          {/* Pure camera feed, no overlays */}
          <div className="relative flex-1 flex items-center justify-center overflow-hidden">
            <video ref={videoRef} autoPlay playsInline className="absolute w-full h-full object-cover" />
            
            {/* Minimal Instructions */}
            <div className="absolute bottom-32 flex flex-col items-center pointer-events-none">
              <p className="text-white text-2xl font-bold drop-shadow-lg">จัดตัวสัตว์ให้อยู่ภายในหน้าจอ</p>
            </div>

            {/* Portrait warning (if screen is tall) */}
            <div className="absolute inset-0 pointer-events-none flex items-center justify-center bg-black/80 portrait:flex landscape:hidden z-10 backdrop-blur-sm">
              <div className="text-white text-center p-6 flex flex-col items-center">
                <Smartphone className="w-20 h-20 animate-[spin_3s_ease-in-out_infinite] mb-6 text-blue-400" />
                <p className="text-2xl font-bold mb-2">กรุณาหมุนโทรศัพท์เป็นแนวนอน</p>
                <p className="text-blue-100/80">เพื่อให้ AI วิเคราะห์สัดส่วนได้อย่างแม่นยำ</p>
              </div>
            </div>
          </div>

          {/* Camera Controls */}
          <div className="bg-black/90 h-28 flex items-center justify-between px-10 pb-safe">
            <button onClick={stopCamera} className="text-white p-4 rounded-full hover:bg-white/20 transition-colors">
              <X size={32} />
            </button>
            <button 
              onClick={capturePhoto} 
              className="w-20 h-20 rounded-full bg-white border-4 border-blue-500 shadow-[0_0_20px_#3b82f6] active:scale-90 transition-transform focus:outline-none"
            >
              <div className="w-full h-full rounded-full border-4 border-black/10"></div>
            </button>
            <div className="w-16"></div> {/* Spacer for alignment */}
          </div>
          <canvas ref={canvasRef} className="hidden" />
        </div>
      )}
    </div>
  );
}

export default function ScanPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-blue-50 flex items-center justify-center"><Loader2 className="animate-spin text-[#1e3a8a]" size={40} /></div>}>
      <ScanContent />
    </Suspense>
  );
}
