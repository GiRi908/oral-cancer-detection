import React, { useState, useEffect } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState('');
  const [confidence, setConfidence] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [serverStatus, setServerStatus] = useState(null);
  const [backendUrl] = useState('http://localhost:5000');

  // Check server status on mount and periodically
  useEffect(() => {
    checkServerStatus();
    const interval = setInterval(checkServerStatus, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const checkServerStatus = async () => {
    try {
      const response = await fetch(`${backendUrl}/health`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        setServerStatus(data);
        console.log('✅ Server status:', data);
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('❌ Server not reachable:', error);
      setServerStatus({ status: 'offline', error: error.message });
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setResult('');
    setConfidence(null);
    
    if (selectedFile) {
      // Validate file size (max 10MB)
      if (selectedFile.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB');
        e.target.value = '';
        return;
      }

      // Validate file type
      const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/webp'];
      if (!validTypes.includes(selectedFile.type)) {
        alert('Please upload a valid image file (PNG, JPG, JPEG, GIF, BMP, WEBP)');
        e.target.value = '';
        return;
      }

      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(selectedFile);
    } else {
      setPreview(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert('Please select an image first.');
      return;
    }

    if (!serverStatus || serverStatus.status === 'offline') {
      alert('Backend server is not connected. Please ensure Flask is running on port 5000.');
      return;
    }

    if (!serverStatus.model_loaded) {
      alert('Model is not loaded on the server. Please check server logs.');
      return;
    }

    setIsLoading(true);
    setResult('');
    setConfidence(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      console.log(`📤 Uploading to: ${backendUrl}/predict`);
      console.log(`📁 File: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`);
      
      const response = await fetch(`${backendUrl}/predict`, {
        method: 'POST',
        body: formData,
        // Don't set Content-Type header - browser will set it with boundary
      });

      console.log(`📥 Response status: ${response.status}`);

      if (!response.ok) {
        let errorData;
        try {
          errorData = await response.json();
        } catch {
          errorData = { error: `HTTP ${response.status}: ${response.statusText}` };
        }
        throw new Error(errorData.error || `Server Error: ${response.status}`);
      }

      const data = await response.json();
      console.log('✅ Response data:', data);

      if (data.error) {
        setResult(`Error: ${data.error}`);
        setConfidence(null);
      } else if (data.prediction) {
        setResult(data.prediction);
        setConfidence(data.confidence);
      } else {
        setResult('Unexpected response format from server');
        setConfidence(null);
      }
    } catch (error) {
      console.error('❌ Upload error:', error);
      
      let errorMessage;
      
      if (error.message.includes('Failed to fetch')) {
        errorMessage = '⚠️ Cannot connect to server.\n\nPlease check:\n1. Flask server is running (python app.py)\n2. Server is on http://localhost:5000\n3. No firewall blocking the connection';
      } else if (error.message.includes('NetworkError')) {
        errorMessage = '⚠️ Network error occurred.\n\nTry:\n1. Refreshing the page\n2. Restarting Flask server\n3. Checking browser console (F12)';
      } else {
        errorMessage = `⚠️ ${error.message}`;
      }
      
      setResult(errorMessage);
      setConfidence(null);
    } finally {
      setIsLoading(false);
    }
  };

  const getResultStyle = () => {
    if (result.includes('No Cancer')) return 'alert alert-success mt-4';
    if (result.includes('Cancer Detected')) return 'alert alert-danger mt-4';
    if (result.includes('Error') || result.includes('⚠️')) return 'alert alert-warning mt-4';
    return 'alert alert-info mt-4';
  };

  const getStatusBadge = () => {
    if (!serverStatus) {
      return <span className="badge bg-secondary">Checking...</span>;
    }
    
    if (serverStatus.status === 'offline') {
      return <span className="badge bg-danger">Server Offline</span>;
    }
    
    if (serverStatus.model_loaded) {
      return <span className="badge bg-success">✓ Ready</span>;
    }
    
    return <span className="badge bg-warning">Model Not Loaded</span>;
  };

  return (
    <div className="container py-5">
      <div className="card shadow-lg p-4">
        {/* Header */}
        <div className="d-flex justify-content-between align-items-center mb-4">
          <h1 className="text-primary mb-0">
            🩺 Oral Cancer Detection
          </h1>
          <div className="d-flex align-items-center gap-2">
            {getStatusBadge()}
            <button 
              className="btn btn-sm btn-outline-secondary"
              onClick={checkServerStatus}
              title="Refresh server status"
              disabled={isLoading}
            >
              🔄
            </button>
          </div>
        </div>

        {/* Server Status Warning */}
        {serverStatus && serverStatus.status === 'offline' && (
          <div className="alert alert-danger" role="alert">
            <strong>⚠️ Backend Server Not Connected</strong>
            <p className="mb-0 mt-2">Please ensure Flask server is running:</p>
            <code className="d-block mt-2 p-2 bg-dark text-white rounded">python app.py</code>
          </div>
        )}

        {serverStatus && !serverStatus.model_loaded && serverStatus.status !== 'offline' && (
          <div className="alert alert-warning" role="alert">
            <strong>⚠️ Model Not Loaded</strong>
            <p className="mb-0 mt-2">Check that model file exists at: <code>model/oral_cancer_final_model.h5</code></p>
          </div>
        )}

        {/* File Upload */}
        <div className="mb-3">
          <label className="form-label fw-bold">Upload Medical Image</label>
          <input
            type="file"
            className="form-control"
            onChange={handleFileChange}
            accept="image/*"
            disabled={isLoading}
          />
          <small className="text-muted">
            Supported: PNG, JPG, JPEG, GIF, BMP, WEBP (Max: 10MB)
          </small>
        </div>

        {/* File Info */}
        {file && (
          <div className="alert alert-info py-2">
            📄 <strong>{file.name}</strong> — {(file.size / 1024).toFixed(2)} KB
          </div>
        )}

        {/* Image Preview */}
        {preview && (
          <div className="text-center my-4">
            <div className="d-inline-block border rounded p-2 bg-light">
              <img 
                src={preview} 
                alt="Preview" 
                style={{ 
                  maxWidth: '400px', 
                  maxHeight: '400px', 
                  borderRadius: '4px',
                  objectFit: 'contain'
                }}
              />
            </div>
          </div>
        )}

        {/* Analyze Button */}
        <div className="text-center my-3">
          <button
            onClick={handleUpload}
            disabled={
              isLoading || 
              !file || 
              !serverStatus || 
              serverStatus.status === 'offline' || 
              !serverStatus.model_loaded
            }
            className="btn btn-primary btn-lg px-5 py-2"
          >
            {isLoading ? (
              <>
                <span className="spinner-border spinner-border-sm me-2" role="status"></span>
                Analyzing Image...
              </>
            ) : (
              <>🔍 Analyze Image</>
            )}
          </button>
        </div>

        {/* Results */}
        {result && (
          <div className={getResultStyle()}>
            <h5 className="mb-3">📊 Analysis Result</h5>
            <div style={{ whiteSpace: 'pre-line', fontSize: '1.1rem' }}>
              {result}
            </div>
            {confidence !== null && (
              <div className="mt-4">
                <div className="d-flex justify-content-between align-items-center mb-2">
                  <strong>Confidence Level:</strong>
                  <span className="badge bg-primary fs-6">{confidence}%</span>
                </div>
                <div className="progress" style={{ height: '25px' }}>
                  <div 
                    className={`progress-bar progress-bar-striped ${
                      confidence > 80 ? 'bg-success' : 
                      confidence > 60 ? 'bg-warning' : 
                      'bg-info'
                    }`}
                    role="progressbar" 
                    style={{ width: `${confidence}%` }}
                  >
                    {confidence}%
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Footer Disclaimer */}
        <div className="mt-4 pt-3 border-top text-center">
          <small className="text-muted">
            <strong>⚕️ Medical Disclaimer:</strong> This tool is for research and educational purposes only. 
            Always consult qualified healthcare professionals for medical diagnosis and treatment.
          </small>
        </div>

        {/* Debug Info (dev mode) */}
        {process.env.NODE_ENV === 'development' && (
          <div className="mt-3">
            <details className="text-muted small">
              <summary style={{ cursor: 'pointer' }}>Debug Info</summary>
              <pre className="mt-2 p-2 bg-light rounded" style={{ fontSize: '0.75rem' }}>
                {JSON.stringify({ 
                  backendUrl, 
                  serverStatus,
                  fileSelected: !!file,
                  fileName: file?.name 
                }, null, 2)}
              </pre>
            </details>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;