import React, { useEffect, useRef } from 'react';
import Chart from 'chart.js/auto';

const GapAnalysis = ({ docId, framework }) => {
  const chartRef = useRef(null);
  const [chartInstance, setChartInstance] = useState(null);
  const [insights, setInsights] = useState(null);

  useEffect(() => {
    const fetchInsights = async () => {
      if (docId && framework) {
        try {
          const response = await fetch(`http://localhost:8000/api/compliance-insights/?doc_id=${docId}&framework=${framework}`);
          const data = await response.json();
          if (data.length > 0) {
            const insight = JSON.parse(data[0].insight);
            setInsights(insight);
          }
        } catch (error) {
          console.error('Error fetching insights:', error);
        }
      }
    };
    fetchInsights();
  }, [docId, framework]);

  useEffect(() => {
    if (chartRef.current && !chartInstance && insights) {
      const ctx = chartRef.current.getContext('2d');
      const newChartInstance = new Chart(ctx, {
        type: 'radar',
        data: {
          labels: ['Compliance', 'Risk', 'Implementation'],
          datasets: [{
            label: framework,
            data: [
              insights.key_requirements ? 80 : 0,
              insights.risk_classification === 'High' ? 100 : insights.risk_classification === 'Medium' ? 50 : 0,
              insights.implementation_actions ? 70 : 0,
            ],
            backgroundColor: 'rgba(59, 130, 246, 0.2)',
            borderColor: 'rgba(59, 130, 246, 1)',
            borderWidth: 2,
          }],
        },
        options: {
          scales: { r: { beginAtZero: true, max: 100, ticks: { stepSize: 20 } } },
          plugins: { legend: { position: 'top' } },
        },
      });
      setChartInstance(newChartInstance);
    }
    return () => {
      if (chartInstance) chartInstance.destroy();
    };
  }, [chartInstance, insights]);

  return (
    <div className="gap-analysis">
      <h3 className="gap-title">Gap Analysis: {docId}</h3>
      <div className="gap-metrics">
        <div className="overall-score">
          <div className="score-circle">{insights?.risk_classification ? '87%' : 'N/A'}</div>
          <span className="score-label">Overall</span>
        </div>
        <div className="metrics-list">
          <p>Fairness: <span className="metric-green">{insights?.key_requirements ? '100%' : 'N/A'}</span></p>
          <p>Transparency: <span className="metric-yellow">{insights?.implementation_actions ? '50%' : 'N/A'}</span></p>
          <p>Privacy: <span className="metric-red">{insights?.risk_classification ? '85%' : 'N/A'}</span></p>
        </div>
      </div>
      <canvas ref={chartRef} className="chart-canvas"></canvas>
      <button className="view-report-btn">View Report</button>
    </div>
  );
};

export default GapAnalysis;