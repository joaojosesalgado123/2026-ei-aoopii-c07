"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Eye,
  ImageUp,
  Loader2,
  RotateCcw,
  Shield,
  Sparkles,
  Upload,
  Zap
} from "lucide-react";

const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/webp"];

const featureCards = [
  {
    title: "Análise Instantânea",
    description: "Resultados em menos de 3 segundos com a nossa IA otimizada",
    icon: Zap
  },
  {
    title: "Alta Precisão",
    description: "Treinada com milhões de imagens reais e geradas por IA",
    icon: Eye
  },
  {
    title: "100% Privado",
    description: "As suas imagens não são armazenadas após a análise",
    icon: Shield
  }
];

function formatBytes(bytes) {
  if (!bytes) return "0 MB";
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function confidenceLabel(probability) {
  if (probability >= 0.75) return "Muito provável";
  if (probability >= 0.55) return "Possível";
  return "Pouco provável";
}

async function analyzeImage(file) {
  const apiUrl = process.env.NEXT_PUBLIC_DETECT_API_URL || "http://localhost:8000/predict";
  const formData = new FormData();
  formData.append("file", file);

  let response;

  try {
    response = await fetch(apiUrl, {
      method: "POST",
      body: formData
    });
  } catch {
    throw new Error(
      "API Python indisponível. Confirma se corriste `npm run api` e se http://localhost:8000/health responde."
    );
  }

  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(payload.error || "Não foi possível concluir a análise pela API.");
  }

  return payload;
}

export default function Home() {
  const inputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const resultState = useMemo(() => {
    if (!result) return null;

    const probabilityFake = Number(result.probabilityFake ?? result.prob_fake ?? 0);
    const probabilityReal = Number(result.probabilityReal ?? (1 - probabilityFake));
    const isFake = result.prediction
      ? result.prediction === "FAKE"
      : probabilityFake >= probabilityReal;
    const confidence = Number(result.confidence ?? (isFake ? probabilityFake : probabilityReal));

    return {
      isFake,
      confidence,
      probabilityFake,
      probabilityReal,
      headline: isFake ? "Provavelmente falsa" : "Provavelmente real",
      description: isFake
        ? "A imagem apresenta sinais compatíveis com geração ou manipulação por IA."
        : "A imagem apresenta sinais mais compatíveis com fotografia real.",
      label: confidenceLabel(confidence),
      mode: result.mode,
      device: result.device,
      fakeClassIndex: result.fakeClassIndex,
      realClassIndex: result.realClassIndex,
      decisionRule: result.decisionRule
    };
  }, [result]);

  const resetFile = useCallback(() => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(null);
    setPreviewUrl("");
    setError("");
    setResult(null);
    setIsAnalyzing(false);
    if (inputRef.current) inputRef.current.value = "";
  }, [previewUrl]);

  const handleFile = useCallback(
    async (file) => {
      if (!file) return;

      setError("");
      setResult(null);

      if (!ACCEPTED_TYPES.includes(file.type)) {
        setError("Use uma imagem PNG, JPG ou WEBP.");
        return;
      }

      if (file.size > MAX_FILE_SIZE) {
        setError("A imagem deve ter no máximo 10MB.");
        return;
      }

      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }

      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setIsAnalyzing(true);

      try {
        const analysis = await analyzeImage(file);
        setResult(analysis);
      } catch (analysisError) {
        setError(analysisError.message || "Ocorreu um erro durante a análise.");
      } finally {
        setIsAnalyzing(false);
      }
    },
    [previewUrl]
  );

  const openFilePicker = () => inputRef.current?.click();

  const onDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    handleFile(event.dataTransfer.files?.[0]);
  };

  return (
    <main className="page-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="AI Detector">
          <span className="brand-icon">
            <Shield aria-hidden="true" />
          </span>
          <span>AI Detector</span>
        </a>

        <nav className="site-nav" aria-label="Navegação principal">
          <a href="#como-funciona">Como funciona</a>
        </nav>
      </header>

      <section id="top" className="hero-section">
        <div className="hero-glow" aria-hidden="true" />

        <div className="hero-copy">
          <span className="ai-badge">
            <Sparkles aria-hidden="true" />
            Powered by AI
          </span>
          <h1>Detete imagens falsas em segundos</h1>
          <p>
            Faça upload de qualquer imagem e a nossa IA irá analisar se
            foi gerada artificialmente ou se é uma fotografia real.
          </p>
        </div>

        <section className="upload-panel" aria-label="Análise de imagem">
          <input
            ref={inputRef}
            className="file-input"
            type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={(event) => handleFile(event.target.files?.[0])}
          />

          <button
            className={`dropzone ${isDragging ? "is-dragging" : ""} ${previewUrl ? "has-preview" : ""}`}
            type="button"
            onClick={openFilePicker}
            onDragOver={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={onDrop}
          >
            {previewUrl ? (
              <span className="preview-wrap">
                <img src={previewUrl} alt="Pré-visualização da imagem selecionada" />
                <span className="preview-overlay">
                  <ImageUp aria-hidden="true" />
                  Trocar imagem
                </span>
              </span>
            ) : (
              <span className="empty-upload">
                <span className="upload-icon">
                  <Upload aria-hidden="true" />
                </span>
                <strong>Arraste uma imagem ou clique para selecionar</strong>
                <small>PNG, JPG ou WEBP (máx. 10MB)</small>
              </span>
            )}
          </button>

          <div className="upload-meta">
            <span>{selectedFile ? selectedFile.name : "Nenhuma imagem selecionada"}</span>
            <span>{selectedFile ? formatBytes(selectedFile.size) : "Pronto para analisar"}</span>
          </div>

          {error ? (
            <p className="message error-message">
              <AlertTriangle aria-hidden="true" />
              {error}
            </p>
          ) : null}

          {isAnalyzing ? (
            <div className="analysis-card is-loading" aria-live="polite">
              <Loader2 aria-hidden="true" />
              <div>
                <strong>A analisar imagem...</strong>
                <span>A verificar sinais de geração artificial.</span>
              </div>
            </div>
          ) : null}

          {resultState ? (
            <div className={`analysis-card ${resultState.isFake ? "is-fake" : "is-real"}`}>
              <div className="result-icon">
                {resultState.isFake ? (
                  <AlertTriangle aria-hidden="true" />
                ) : (
                  <CheckCircle2 aria-hidden="true" />
                )}
              </div>
              <div className="result-content">
                <span>{resultState.label}</span>
                <strong>{resultState.headline}</strong>
                <p>{resultState.description}</p>
                <div className="confidence-row">
                  <span>Confiança</span>
                  <b>{Math.round(resultState.confidence * 100)}%</b>
                </div>
                <div className="confidence-track">
                  <span style={{ width: `${Math.round(resultState.confidence * 100)}%` }} />
                </div>
                <div className="probability-grid">
                  <span>
                    <b>REAL</b>
                    {Math.round(resultState.probabilityReal * 100)}%
                  </span>
                  <span>
                    <b>FAKE</b>
                    {Math.round(resultState.probabilityFake * 100)}%
                  </span>
                </div>
                {resultState.mode === "model" ? (
                  <small>
                    Modelo Python: FAKE=classe {resultState.fakeClassIndex}, decisão por maior probabilidade.
                  </small>
                ) : null}
                {resultState.mode === "demo" ? (
                  <small>
                    Resultado simulado. Ainda não está ligado ao modelo Python/Streamlit.
                  </small>
                ) : null}
              </div>
            </div>
          ) : null}

          {selectedFile ? (
            <button className="reset-button" type="button" onClick={resetFile}>
              <RotateCcw aria-hidden="true" />
              Nova análise
            </button>
          ) : null}
        </section>
      </section>

      <section id="como-funciona" className="features-section" aria-label="Vantagens">
        {featureCards.map((feature) => {
          const Icon = feature.icon;
          return (
            <article className="feature-card" key={feature.title}>
              <span className="feature-icon">
                <Icon aria-hidden="true" />
              </span>
              <h2>{feature.title}</h2>
              <p>{feature.description}</p>
            </article>
          );
        })}
      </section>

      <footer className="site-footer">
        <span>© 2026 AI Detector. Todos os direitos reservados.</span>
      </footer>
    </main>
  );
}
