document.addEventListener('DOMContentLoaded', () => {
    const seletorJob = document.getElementById('job-select');

    // Função para buscar e desenhar o gráfico
    const carregarGrafico = (nomeJob) => {
        // Mostra uma mensagem de "carregando"
        const container = document.getElementById('grafico-container');
        container.innerHTML = 'Carregando gráfico...';

        // Faz a chamada para a nossa API no back-end
        fetch(`/api/grafico/tempo/${nomeJob}`)
            .then(response => response.json())
            .then(figuraJson => {
                // Usa a biblioteca Plotly para desenhar o gráfico
                // A figuraJson já vem com os dados (data) e o layout (layout)
                Plotly.newPlot('grafico-container', figuraJson.data, figuraJson.layout);
            })
            .catch(error => {
                console.error('Erro ao buscar o gráfico:', error);
                container.innerHTML = 'Erro ao carregar o gráfico. Verifique o console.';
            });
    };

    // Evento que dispara a função quando o usuário muda o job no seletor
    seletorJob.addEventListener('change', () => {
        carregarGrafico(seletorJob.value);
    });

    // Carrega o gráfico do primeiro job da lista ao iniciar a página
    carregarGrafico(seletorJob.value);
});