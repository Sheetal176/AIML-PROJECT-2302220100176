import React, { useState, useRef } from 'react';
import './App.css';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

function App() {
  const [formData, setFormData] = useState({
    gender: 'M',
    ssc_p: 70,
    hsc_p: 70,
    hsc_s: 'Commerce',
    degree_p: 70,
    degree_t: 'Comm&Mgmt',
    workex: 'No',
    etest_p: 70,
    specialisation: 'None',
    mba_p: ''
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const reportRef = useRef(null);

  const downloadPDF = async () => {
    if (!reportRef.current) return;
    try {
      const canvas = await html2canvas(reportRef.current, { scale: 2, backgroundColor: '#1e2a38' });
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
      pdf.save('Placement_Prediction_Report.pdf');
    } catch (err) {
      console.error(err);
      alert('Failed to generate PDF report.');
    }
  };

  const getChartData = () => {
    if (!result || !result.feature_importance) return null;
    const labels = Object.keys(result.feature_importance);
    const data = Object.values(result.feature_importance);
    
    const backgroundColors = data.map(val => val >= 0 ? 'rgba(46, 204, 113, 0.8)' : 'rgba(231, 76, 60, 0.8)');

    return {
      labels,
      datasets: [
        {
          label: 'Feature Impact on Probability',
          data,
          backgroundColor: backgroundColors,
        },
      ],
    };
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // Convert strings to floats for percentage fields
        body: JSON.stringify({
          ...formData,
          ssc_p: parseFloat(formData.ssc_p),
          hsc_p: parseFloat(formData.hsc_p),
          degree_p: parseFloat(formData.degree_p),
          etest_p: parseFloat(formData.etest_p),
          mba_p: parseFloat(formData.mba_p),
        })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Prediction failed');
      setResult(data);
    } catch (error) {
      console.error(error);
      alert('Failed to connect to ML Backend. Make sure FastAPI is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Campus Placement Prediction</h1>
        <p>AI-Powered Analytics & Career Recommendations</p>
      </header>

      <div className="content-grid">
        <div className="glass-panel">
          <h2>Student Profile</h2>
          <form onSubmit={handleSubmit}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div className="form-group">
                <label>Gender</label>
                <select name="gender" value={formData.gender} onChange={handleChange} className="input-field">
                  <option value="M">Male</option>
                  <option value="F">Female</option>
                </select>
              </div>
              <div className="form-group">
                <label>Work Experience</label>
                <select name="workex" value={formData.workex} onChange={handleChange} className="input-field">
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>
              
              <div className="form-group">
                <label>SSC % (10th)</label>
                <input type="number" name="ssc_p" value={formData.ssc_p} onChange={handleChange} className="input-field" min="0" max="100" step="0.1" required />
              </div>
              <div className="form-group">
                <label>HSC % (12th)</label>
                <input type="number" name="hsc_p" value={formData.hsc_p} onChange={handleChange} className="input-field" min="0" max="100" step="0.1" required />
              </div>

              <div className="form-group">
                <label>HSC Specialisation</label>
                <select name="hsc_s" value={formData.hsc_s} onChange={handleChange} className="input-field">
                  <option value="Commerce">Commerce</option>
                  <option value="Science">Science</option>
                  <option value="Arts">Arts</option>
                </select>
              </div>
              <div className="form-group">
                <label>Degree %</label>
                <input type="number" name="degree_p" value={formData.degree_p} onChange={handleChange} className="input-field" min="0" max="100" step="0.1" required />
              </div>

              <div className="form-group">
                <label>Degree Type</label>
                <select name="degree_t" value={formData.degree_t} onChange={handleChange} className="input-field">
                  <option value="Comm&Mgmt">Comm&Mgmt</option>
                  <option value="Sci&Tech">Sci&Tech</option>
                  <option value="Others">Others</option>
                </select>
              </div>
              <div className="form-group">
                <label>E-Test %</label>
                <input type="number" name="etest_p" value={formData.etest_p} onChange={handleChange} className="input-field" min="0" max="100" step="0.1" required />
              </div>

              <div className="form-group">
                <label>MBA Specialisation (Optional)</label>
                <select name="specialisation" value={formData.specialisation} onChange={handleChange} className="input-field">
                  <option value="None">None / Not Applicable</option>
                  <option value="Mkt&HR">Mkt&HR</option>
                  <option value="Mkt&Fin">Mkt&Fin</option>
                </select>
              </div>
              <div className="form-group">
                <label>MBA % (Optional)</label>
                <input type="number" name="mba_p" value={formData.mba_p} onChange={handleChange} className="input-field" min="0" max="100" step="0.1" />
              </div>
            </div>
            
            <button type="submit" className="btn-primary" style={{ width: '100%', marginTop: '1rem' }}>
              {loading ? 'Analyzing...' : 'Predict Placement Probability'}
            </button>
          </form>
        </div>

        <div className="glass-panel" ref={reportRef}>
          <h2>Prediction Results</h2>
          {result ? (
            <div className="result-card">
              <h3 style={{ color: result.prediction === 'Placed' ? '#2ecc71' : '#e74c3c' }}>
                Status: {result.prediction}
              </h3>
              <div className="probability-circle" style={{ borderColor: result.prediction === 'Placed' ? '#2ecc71' : '#e74c3c' }}>
                {result.placement_probability}%
              </div>
              
              <div className="recommendations">
                <h4>Explainable AI (Feature Impact)</h4>
                {result.feature_importance && Object.keys(result.feature_importance).length > 0 && (
                  <div style={{ height: '300px', width: '100%', marginBottom: '2rem' }}>
                    <Bar 
                      data={getChartData()} 
                      options={{ 
                        maintainAspectRatio: false,
                        indexAxis: 'y',
                        plugins: {
                          legend: { display: false },
                          title: { display: true, text: 'SHAP Feature Importance', color: '#fff' }
                        },
                        scales: {
                          x: { ticks: { color: '#ccc' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                          y: { ticks: { color: '#ccc' }, grid: { display: false } }
                        }
                      }} 
                    />
                  </div>
                )}
                
                <h4>Actionable Recommendations:</h4>
                <ul>
                  {result.recommendations.map((rec, i) => (
                    <li key={i}>{rec}</li>
                  ))}
                </ul>
              </div>

              <button onClick={downloadPDF} className="btn-primary" style={{ marginTop: '1.5rem', background: '#3498db' }}>
                Download Personalized PDF Report
              </button>
            </div>
          ) : (
            <div style={{ textAlign: 'center', marginTop: '3rem', color: '#ccc' }}>
              <p>Enter your profile details and click predict to see your placement probability and AI recommendations.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
