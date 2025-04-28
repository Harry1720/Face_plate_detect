import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { BrowserRouter as Router, Route, Routes, useNavigate, useLocation } from 'react-router-dom';
import './App.css';

function MainApp() {
  const [activeMode, setActiveMode] = useState(null);
  const [logs, setLogs] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [faceStreamUrl, setFaceStreamUrl] = useState('');
  const [plateStreamUrl, setPlateStreamUrl] = useState('');
  const [recentData, setRecentData] = useState({ recent_plate: null, recent_face: null });
  const faceImgRef = useRef(null);
  const plateImgRef = useRef(null);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    // Set activeMode based on the current route
    const path = location.pathname;
    if (path === '/checkin') {
      setActiveMode('checkin');
    } else if (path === '/checkout') {
      setActiveMode('checkout');
    } else if (path === '/logs') {
      setActiveMode('logs');
    } else {
      setActiveMode(null);
    }
  }, [location]);

  const handleCheckin = async () => {
    setActiveMode('checkin');
    setStatusMessage('Starting check-in process...');
    setError(null);
    navigate('/checkin');
    try {
      setIsLoading(true);
      const response = await axios.post('http://localhost:5000/start-checkin');
      if (response.data.success) {
        const timestamp = new Date().getTime();
        setFaceStreamUrl(`http://localhost:5000/video_feed/face?t=${timestamp}`);
        setPlateStreamUrl(`http://localhost:5000/video_feed/plate?t=${timestamp}`);
        setStatusMessage('Check-in mode active - Please look at the camera and position your vehicle');
      } else {
        throw new Error(response.data.message);
      }
    } catch (err) {
      setError('Failed to start check-in: ' + err.message);
      setStatusMessage('Error: Could not start check-in');
      setActiveMode(null);
      navigate('/');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCheckout = async () => {
    setActiveMode('checkout');
    setStatusMessage('Starting check-out process...');
    setError(null);
    navigate('/checkout');
    try {
      setIsLoading(true);
      const response = await axios.post('http://localhost:5000/start-checkout');
      if (response.data.success) {
        const timestamp = new Date().getTime();
        setFaceStreamUrl(`http://localhost:5000/video_feed/face?t=${timestamp}`);
        setPlateStreamUrl(`http://localhost:5000/video_feed/plate?t=${timestamp}`);
        setStatusMessage('Check-out mode active - Please look at the camera and position your vehicle');
      } else {
        throw new Error(response.data.message);
      }
    } catch (err) {
      setError('Failed to start check-out: ' + err.message);
      setStatusMessage('Error: Could not start check-out');
      setActiveMode(null);
      navigate('/');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCheckLogs = async () => {
    setActiveMode('logs');
    setStatusMessage('Fetching logs...');
    setFaceStreamUrl('');
    setPlateStreamUrl('');
    setError(null);
    navigate('/logs');
    try {
      setIsLoading(true);
      const response = await axios.get('http://localhost:5000/logs');
      if (response.data.success) {
        setLogs(response.data.logs);
        setStatusMessage(`Found ${response.data.logs.length} log entries`);
      } else {
        throw new Error(response.data.message);
      }
    } catch (err) {
      setError('Failed to fetch logs: ' + err.message);
      setStatusMessage('Error: Could not retrieve logs');
      setLogs([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleStop = async () => {
    try {
      await axios.post('http://localhost:5000/stop');
      setActiveMode(null);
      setFaceStreamUrl('');
      setPlateStreamUrl('');
      setStatusMessage('Process stopped');
      setRecentData({ recent_plate: null, recent_face: null });
      setError(null);
      navigate('/');
    } catch (err) {
      setError('Failed to stop process: ' + err.message);
    }
  };

  useEffect(() => {
    let interval;
    if (activeMode === 'checkin' || activeMode === 'checkout') {
      interval = setInterval(async () => {
        try {
          const response = await axios.get('http://localhost:5000/recent-data');
          if (response.data.success) {
            setRecentData(response.data.data);
            if (response.data.data.recent_plate) {
              setStatusMessage(`Detected plate: ${response.data.data.recent_plate.plate_text}`);
            } else if (response.data.data.recent_face) {
              setStatusMessage(`Detected face: User ${response.data.data.recent_face.user_id}`);
            }
          }
        } catch (err) {
          console.error('Failed to fetch recent data:', err.message);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [activeMode]);

  useEffect(() => {
    const checkServerStatus = async () => {
      const maxRetries = 3;
      let attempts = 0;
      while (attempts < maxRetries) {
        try {
          const response = await axios.get('http://localhost:5000/status', { timeout: 5000 });
          setIsConnected(response.data.success);
          setError(null);
          return;
        } catch (err) {
          attempts++;
          console.error(`Server status check failed (attempt ${attempts}/${maxRetries}):`, err.message);
          if (attempts === maxRetries) {
            setIsConnected(false);
            setError('Cannot connect to server. Please ensure the Python API server is running: `python api.py`');
          }
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      }
    };
    checkServerStatus();
    const interval = setInterval(checkServerStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const handleImageError = (elem, streamType) => {
      if (elem) {
        elem.onerror = () => {
          setError(`Failed to load ${streamType} stream. Make sure cameras are connected.`);
          const timestamp = new Date().getTime();
          if (streamType === 'face') {
            setFaceStreamUrl(`http://localhost:5000/video_feed/face?t=${timestamp}`);
          } else {
            setPlateStreamUrl(`http://localhost:5000/video_feed/plate?t=${timestamp}`);
          }
        };
      }
    };
    handleImageError(faceImgRef.current, 'face');
    handleImageError(plateImgRef.current, 'plate');
  }, [faceStreamUrl, plateStreamUrl]);

  return (
    <div className="app-container">
      <header>
        <h1>Smart Parking System</h1>
        <div className={`server-status ${isConnected ? 'connected' : 'disconnected'}`}>
          Server status: {isConnected ? 'Connected' : 'Disconnected'}
        </div>
      </header>

      <div className="controls">
        <button
          className={`control-button ${activeMode === 'checkin' ? 'active' : ''}`}
          onClick={handleCheckin}
          disabled={!isConnected || isLoading || activeMode === 'checkin'}
        >
          Check In
        </button>
        <button
          className={`control-button ${activeMode === 'checkout' ? 'active' : ''}`}
          onClick={handleCheckout}
          disabled={!isConnected || isLoading || activeMode === 'checkout'}
        >
          Check Out
        </button>
        <button
          className={`control-button ${activeMode === 'logs' ? 'active' : ''}`}
          onClick={handleCheckLogs}
          disabled={!isConnected || isLoading}
        >
          View Logs
        </button>
        {(activeMode === 'checkin' || activeMode === 'checkout') && (
          <button className="stop-button" onClick={handleStop} disabled={isLoading}>
            Stop
          </button>
        )}
      </div>

      {statusMessage && <div className="status-message">{statusMessage}</div>}

      {error && (
        <div className="error-message">
          {error}
          {!isConnected && (
            <div className="error-details">
              <p>Please ensure the Python API server is running:</p>
              <code>python api.py</code>
            </div>
          )}
        </div>
      )}

      {(activeMode === 'checkin' || activeMode === 'checkout') && (
        <div className="camera-container">
          <div className="status-bar">
            <div className="status-indicator active">
              Mode: {activeMode === 'checkin' ? 'CHECK IN' : 'CHECK OUT'}
            </div>
            <div className="status-indicator active">Face Camera</div>
            <div className="status-indicator active">Plate Camera</div>
          </div>

          <div className="cameras-display">
            <div className="camera-section">
              <h2>Face Recognition</h2>
              <div className="camera-feed">
                {faceStreamUrl ? (
                  <img
                    ref={faceImgRef}
                    src={faceStreamUrl}
                    alt="Face recognition stream"
                    onError={() => setError('Face camera feed unavailable')}
                  />
                ) : (
                  <div className="no-stream">Waiting for face camera stream...</div>
                )}
              </div>
              {recentData.recent_face && (
                <div className="detection-info">
                  <p><strong>User ID:</strong> {recentData.recent_face.user_id}</p>
                  <p><strong>Detected:</strong> {new Date(recentData.recent_face.last_access).toLocaleString()}</p>
                </div>
              )}
            </div>

            <div className="camera-section">
              <h2>License Plate Detection</h2>
              <div className="camera-feed">
                {plateStreamUrl ? (
                  <img
                    ref={plateImgRef}
                    src={plateStreamUrl}
                    alt="Plate detection stream"
                    onError={() => setError('Plate camera feed unavailable')}
                  />
                ) : (
                  <div className="no-stream">Waiting for plate camera stream...</div>
                )}
              </div>
              {recentData.recent_plate && (
                <div className="detection-info">
                  <p><strong>Plate:</strong> {recentData.recent_plate.plate_text}</p>
                  <p><strong>User ID:</strong> {recentData.recent_plate.user_id || 'Not assigned'}</p>
                  <p><strong>Detected:</strong> {new Date(recentData.recent_plate.updated_at).toLocaleString()}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {activeMode === 'logs' && (
        <div className="logs-container">
          <h2>System Logs</h2>
          {isLoading ? (
            <div className="loading">Loading logs...</div>
          ) : logs.length > 0 ? (
            <table className="logs-table">
              <thead>
                <tr>
                  <th>User ID</th>
                  <th>Plate</th>
                  <th>Status</th>
                  <th>Date & Time</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((log, index) => (
                  <tr key={index}>
                    <td>{log.user_id}</td>
                    <td>{log.plate_text}</td>
                    <td>{log.status}</td>
                    <td>{new Date(log.updated_at).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="no-logs">No logs found</div>
          )}
        </div>
      )}

      <footer>
        <p>Smart Parking System - {new Date().getFullYear()}</p>
      </footer>
    </div>
  );
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainApp />} />
        <Route path="/checkin" element={<MainApp />} />
        <Route path="/checkout" element={<MainApp />} />
        <Route path="/logs" element={<MainApp />} />
      </Routes>
    </Router>
  );
}

export default App;