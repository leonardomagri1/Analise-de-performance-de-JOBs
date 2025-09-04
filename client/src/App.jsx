import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import './App.css'; // Sinta-se à vontade para estilizar aqui

function App() {
  const [selectedJob, setSelectedJob] = useState('OCGJD400');

  const [chartCpuData, setChartCpuData] = useState(null);
  const [chartSalaData, setChartSalaData] = useState(null);
  const [chartExecucaoData, setChartExecucaoData] = useState(null);

  const [imageCpuUrl, setImageCpuUrl] = useState(null);
  const [imageSalaUrl, setImageSalaUrl] = useState(null);

  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const carregarGraficos = async () => {
      setIsLoading(true);
      setChartCpuData(null); setChartSalaData(null); setChartExecucaoData(null);
      setImageCpuUrl(null); setImageSalaUrl(null);

      const tipos = ['cpu', 'sala', 'execucao'];
      for (const tipo of tipos) {
        try {
          const response = await fetch(`http://localhost:5000/api/grafico/${tipo}/${selectedJob}`);
          // Tenta parsear JSON
          const json = await response.json();

          // Caso a API retorne { urlImagem: "/grafico_xxx.png" }
          if (json && json.urlImagem) {
            if (tipo === 'cpu') setImageCpuUrl(json.urlImagem);
            if (tipo === 'sala') setImageSalaUrl(json.urlImagem);
            continue;
          }

          // Caso a API retorne um objeto com campo 'fig' (nosso caso ideal)
          let figObj = null;
          if (json && json.fig) {
            // 'fig' pode ser string JSON (pio.to_json) ou já um objeto
            if (typeof json.fig === 'string') {
              try {
                figObj = JSON.parse(json.fig);
              } catch (e) {
                // fallback: se falhar, tenta parsear como objeto literal
                figObj = null;
              }
            } else {
              figObj = json.fig;
            }
          } else if (json && json.data && json.layout) {
            // retorno direto { data: [...], layout: {...} }
            figObj = { data: json.data, layout: json.layout };
          } else if (typeof json === 'string') {
            // Se a API devolveu string JSON pura
            try {
              const parsed = JSON.parse(json);
              if (parsed.data && parsed.layout) figObj = parsed;
            } catch (e) {
              figObj = null;
            }
          }

          if (figObj) {
            if (tipo === 'cpu') setChartCpuData(figObj);
            if (tipo === 'sala') setChartSalaData(figObj);
            if (tipo === 'execucao') setChartExecucaoData(figObj);
          } else {
            // sem figObj: logar para debug
            console.warn(`Resposta não reconhecida do endpoint /api/grafico/${tipo}/${selectedJob}:`, json);
          }
        } catch (error) {
          console.error(`Erro ao carregar gráfico ${tipo}:`, error);
        }
      }

      setIsLoading(false);
    };

    carregarGraficos();
  }, [selectedJob]);

  const renderPlotOrImage = (chartData, imageUrl) => {
    if (imageUrl) {
      // imagem retornada pelo servidor (PNG)
      return <img src={imageUrl} alt="Gráfico" style={{ width: '100%', maxHeight: 600 }} />;
    }
    if (chartData && chartData.data && chartData.layout) {
      return (
        <Plot
          data={chartData.data}
          layout={{ ...chartData.layout, autosize: true }}
          style={{ width: '100%', height: '500px' }}
          useResizeHandler={true}
        />
      );
    }
    return null;
  };

  return (
    <div className="App" style={{ padding: 20 }}>
      <h1>Análise de Performance de Jobs</h1>
      <div className="seletor-container" style={{ marginBottom: 16 }}>
        <label htmlFor="job-select">Selecione o Job: </label>
        <select
          id="job-select"
          value={selectedJob}
          onChange={e => setSelectedJob(e.target.value)}
          style={{ marginLeft: 8 }}
        >
          <option value="OCGJD400">OCGJD400</option>
          <option value="DBA21P01">DBA21P01</option>
        </select>
      </div>

      <div className="grafico-container">
        {isLoading ? (
          <p>Gerando gráficos interativos...</p>
        ) : (
          <div>
            <h2>CPU</h2>
            {renderPlotOrImage(chartCpuData, imageCpuUrl)}
            <hr style={{ margin: '24px 0' }} />

            <h2>SALA</h2>
            {renderPlotOrImage(chartSalaData, imageSalaUrl)}
            <hr style={{ margin: '24px 0' }} />

            <h2>Hora de Execução</h2>
            {chartExecucaoData ? (
              <Plot
                data={chartExecucaoData.data}
                layout={{ ...chartExecucaoData.layout, title: `Hora de Execução — ${selectedJob}`, autosize: true }}
                style={{ width: '100%', height: '500px' }}
                useResizeHandler={true}
              />
            ) : (
              <p>Gráfico de horário não disponível.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
