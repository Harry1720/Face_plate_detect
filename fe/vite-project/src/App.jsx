import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [activeMode, setActiveMode] = useState(null) // 'checkin', 'checkout', or 'logs'
  const [logs, setLogs] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const [faceStreamUrl, setFaceStreamUrl] = useState('')
  const [plateStreamUrl, setPlateStreamUrl] = useState('')
  const [statusMessage, setStatusMessage] = useState('')
  const faceImgRef = useRef(null)
  const plateImgRef = useRef(null)

  // Function to start the checkin process
  const handleCheckin = async () => {
    setActiveMode('checkin')
    setStatusMessage('Starting checkin process...')
    try {
      setIsLoading(true)
      const response = await axios.post('http://localhost:5000/start-checkin')
      if (response.data.success) {
        // Use timestamp to prevent browser caching
        const timestamp = new Date().getTime()
        setFaceStreamUrl(`http://localhost:5000/video_feed/face?t=${timestamp}`)
        setPlateStreamUrl(`http://localhost:5000/video_feed/plate?t=${timestamp}`)
        setStatusMessage('Checkin mode active - Please look at the camera and position your vehicle')
      }
    } catch (err) {
      setError('Failed to start checkin process: ' + err.message)
      setStatusMessage('Error: Could not start checkin')
    } finally {
      setIsLoading(false)
    }
  }
  
  // Function to start the checkout process
  const handleCheckout = async () => {
    setActiveMode('checkout')
    setStatusMessage('Starting checkout process...')
    try {
      setIsLoading(true)
      const response = await axios.post('http://localhost:5000/start-checkout')
      if (response.data.success) {
        // Use timestamp to prevent browser caching
        const timestamp = new Date().getTime()
        setFaceStreamUrl(`http://localhost:5000/video_feed/face?t=${timestamp}`)
        setPlateStreamUrl(`http://localhost:5000/video_feed/plate?t=${timestamp}`)
        setStatusMessage('Checkout mode active - Please look at the camera and position your vehicle')
      }
    } catch (err) {
      setError('Failed to start checkout process: ' + err.message)
      setStatusMessage('Error: Could not start checkout')
    } finally {
      setIsLoading(false)
    }
  }
  
  // Function to fetch logs
  const handleCheckLogs = async () => {
    setActiveMode('logs')
    setStatusMessage('Fetching logs...')
    setFaceStreamUrl('')
    setPlateStreamUrl('')
    
    try {
      setIsLoading(true)
      const response = await axios.get('http://localhost:5000/logs')
      if (response.data.success) {
        setLogs(response.data.logs)
        setStatusMessage(`Found ${response.data.logs.length} log entries`)
      } else {
        setError('Error retrieving logs: ' + response.data.message)
        setLogs([])
      }
    } catch (err) {
      setError('Failed to fetch logs: ' + err.message)
      setStatusMessage('Error: Could not retrieve logs')
      setLogs([])
    } finally {
      setIsLoading(false)
    }
  }
  
  // Stop current process
  const handleStop = async () => {
    try {
      await axios.post('http://localhost:5000/stop')
      setActiveMode(null)
      setFaceStreamUrl('')
      setPlateStreamUrl('')
      setStatusMessage('Process stopped')
    } catch (err) {
      setError('Failed to stop process: ' + err.message)
    }
  }
  
  // Effect to handle image error
  useEffect(() => {
    // Add error handlers to image elements
    const handleImageError = (elem, streamType) => {
      if (elem) {
        elem.onerror = () => {
          setError(`Failed to load ${streamType} stream. Make sure cameras are connected.`);
          // Try to reload the image with a new timestamp to bypass cache
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
  
  // Check if server is running on component mount
  useEffect(() => {
    const checkServerStatus = async () => {
      try {
        const response = await axios.get('http://localhost:5000/status', { timeout: 3000 })
        setIsConnected(true)
        setError(null)
      } catch (err) {
        setIsConnected(false)
        setError('Cannot connect to server. Please make sure the Python server is running.')
      }
    }
    
    checkServerStatus()
    const interval = setInterval(checkServerStatus, 10000) // Check every 10 seconds
    
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="app-container">
      <header>
        <h1>Face & Plate Recognition System</h1>
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
          Check Logs
        </button>
        {(activeMode === 'checkin' || activeMode === 'checkout') && (
          <button 
            className="stop-button" 
            onClick={handleStop}
            disabled={isLoading}
          >
            Stop
          </button>
        )}
      </div>
      
      {statusMessage && (
        <div className="status-message">
          {statusMessage}
        </div>
      )}
      
      {error && (
        <div className="error-message">
          {error}
          {!isConnected && (
            <div className="error-details">
              <p>Please make sure the Python API server is running:</p>
              <code>cd face && python api.py</code>
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
            <div className={`status-indicator ${faceStreamUrl ? 'active' : 'inactive'}`}>
              Face Camera
            </div>
            <div className={`status-indicator ${plateStreamUrl ? 'active' : 'inactive'}`}>
              Plate Camera
            </div>
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
            </div>
            
            <div className="camera-section">
              <h2>Plate Detection</h2>
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
        <p>Face & Plate Recognition System - {new Date().getFullYear()}</p>
      </footer>
    </div>
  )
}

export default App