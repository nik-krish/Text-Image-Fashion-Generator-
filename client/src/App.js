import React, { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [prompt, setPrompt] = useState("");
  const [imageUrl, setImageUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showOriginal, setShowOriginal] = useState(false);
  const [theme] = useState("cyber");

  const suggestions = {
    cyber: [
      "dress lilac pink",
      "blue surplice jersey maxi dress",
      "pink haley tee",
      "sheath dress gray cut-out dress",
    ],
  };

  const generateImage = async () => {
    if (!prompt.trim()) {
      setError("Please enter a description");
      return;
    }

    setLoading(true);
    setImageUrl(null);
    setError(null);
    setShowOriginal(false);

    try {
      const response = await fetch("http://localhost:5000/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt, theme }),
      });

      if (!response.ok) {
        throw new Error("Failed to generate image");
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);

      const img = new Image();
      img.onload = function () {
        const canvas = document.createElement("canvas");
        const scaleFactor = 4;
        canvas.width = this.width * scaleFactor;
        canvas.height = this.height * scaleFactor;

        const ctx = canvas.getContext("2d");
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        const upscaledUrl = canvas.toDataURL("image/png");
        setImageUrl({ original: url, upscaled: upscaledUrl });
        setLoading(false);
      };
      img.src = url;
    } catch (err) {
      console.error(err);
      setError("Error generating image. Please try again.");
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      generateImage();
    }
  };

  const themeStyles = {
    cyber: {
      primary: "#00f2ff",
      secondary: "#ff00e6",
      background: "#0a0a20",
      accent: "#ffcc00",
      text: "#ffffff",
    },
  };

  const currentTheme = themeStyles[theme];

  useEffect(() => {
    document.documentElement.style.setProperty(
      "--primary-color",
      currentTheme.primary
    );
    document.documentElement.style.setProperty(
      "--secondary-color",
      currentTheme.secondary
    );
    document.documentElement.style.setProperty(
      "--background-color",
      currentTheme.background
    );
    document.documentElement.style.setProperty(
      "--accent-color",
      currentTheme.accent
    );
    document.documentElement.style.setProperty(
      "--text-color",
      currentTheme.text
    );
  }, [currentTheme]);

  return (
    <div className="app-container">
      <div className="app-background">
        <div className="grid-pattern"></div>
        <div className="glow-effect"></div>
      </div>

      <header className="app-header">
        <div className="logo-container">
          <div className="pixel-logo"></div>
          <h1>
            Fashion Generator<span>Powered by AI</span>
          </h1>
        </div>
      </header>

      <main className="main-content">
        <div className="generator-panel">
          <div className="input-container">
            <div className="input-wrapper">
              <input
                type="text"
                placeholder="Describe your pixel fashion creation..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={loading}
                className="prompt-input"
              />
              <button
                onClick={generateImage}
                disabled={loading}
                className="generate-btn"
              >
                {loading ? (
                  <div className="loading-icon">
                    <div className="pixel-spinner"></div>
                  </div>
                ) : (
                  <>
                    GENERATE<span className="btn-effect"></span>
                  </>
                )}
              </button>
            </div>

            {error && <div className="error-message">{error}</div>}

            <div className="prompt-suggestions">
              <div className="suggestion-chips">
                {suggestions.cyber.map((suggestion) => (
                  <div
                    key={suggestion}
                    className="chip"
                    onClick={() => setPrompt(suggestion)}
                  >
                    {suggestion}
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="result-container">
            {loading ? (
              <div className="loading-display">
                <div className="pixel-matrix">
                  {Array.from({ length: 64 }).map((_, i) => (
                    <div key={i} className="matrix-pixel"></div>
                  ))}
                </div>
                <p className="loading-text">Generating fashion</p>
              </div>
            ) : imageUrl ? (
              <div className="image-showcase">
                <div className="display-frame">
                  <img
                    src={showOriginal ? imageUrl.original : imageUrl.upscaled}
                    alt="Generated pixel fashion"
                    className={`pixel-display ${
                      showOriginal ? "original" : "upscaled"
                    }`}
                  />
                  <div className="frame-corners">
                    <span className="corner top-left"></span>
                    <span className="corner top-right"></span>
                    <span className="corner bottom-left"></span>
                    <span className="corner bottom-right"></span>
                  </div>
                </div>

                <div className="image-controls">
                  <div className="control-buttons">
                    <button
                      onClick={() => setShowOriginal(!showOriginal)}
                      className="view-toggle"
                    >
                      {showOriginal ? "ENHANCED" : "ORIGINAL"}
                    </button>

                    <a
                      href={
                        showOriginal ? imageUrl.original : imageUrl.upscaled
                      }
                      download={`pixel-fashion-${theme}-${Date.now()}.png`}
                      className="download-btn"
                    >
                      DOWNLOAD
                    </a>
                  </div>

                  <div className="image-info">
                    <span className="info-tag">
                      {showOriginal ? "64×64" : "256×256"}
                    </span>
                    <span className="info-tag">{theme.toUpperCase()}</span>
                    <span className="info-tag">FASHION ART</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="empty-display">
                <div className="pixel-canvas">
                  <div className="canvas-animation">
                    {Array.from({ length: 100 }).map((_, i) => (
                      <div
                        key={i}
                        className="canvas-pixel"
                        style={{
                          animationDelay: `${Math.random() * 2}s`,
                          backgroundColor: `hsl(${
                            Math.random() * 360
                          }, 80%, 60%, 0.3)`,
                        }}
                      ></div>
                    ))}
                  </div>
                </div>
                <p className="empty-text">Your creation will appear here</p>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <div className="footer-bottom">
          <p>Design your digital fashion identity</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
