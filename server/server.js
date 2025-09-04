const express = require('express');
const fs = require('fs');
const path = require('path');
const cors = require('cors');
const { exec } = require('child_process');
const app = express();
const port = 5000;

app.use(cors()); // Habilita o CORS para todas as rotas

// Esta função está aqui para referência futura, mas não é usada pela rota atual
function carregarDadosJob(nomeJob) {
    try {
        const caminhoArquivo = path.join(__dirname, 'dados_mock.json');
        const data = fs.readFileSync(caminhoArquivo, 'utf-8');
        const todosOsDados = JSON.parse(data);
        const dadosJob = todosOsDados.filter(reg => reg.nome_job === nomeJob);
        if (!dadosJob || dadosJob.length === 0) return null;
        const dadosParaGrafico = { datas: [], cpu: [], sala: [] };
        dadosJob.sort((a, b) => new Date(a.data_execucao.split('/').reverse().join('-')) - new Date(b.data_execucao.split('/').reverse().join('-')));
        dadosJob.forEach(reg => {
            const dataFormatada = reg.data_execucao.split('/').reverse().join('-');
            const horaFormatada = reg.hora_execucao.replace(/\./g, ':');
            dadosParaGrafico.datas.push(`${dataFormatada} ${horaFormatada}`);
            dadosParaGrafico.cpu.push(parseFloat(reg.tempo_cpu.replace(',', '.')));
            dadosParaGrafico.sala.push(parseFloat(reg.tempo_sala.replace(',', '.')));
        });
        return dadosParaGrafico;
    } catch (error) {
        console.error("Erro ao carregar dados:", error);
        return null;
    }
}

const fetch = require('node-fetch'); // npm install node-fetch@2

app.get('/api/grafico/:tipo/:nomeJob', async (req, res) => {
    const { tipo, nomeJob } = req.params;
    const caminhoJson = path.join(__dirname, 'dados_mock.json');

    // Se for gráfico 'execucao' ou 'horario' -> proxy para Flask (app.py) que gera JSON Plotly
    if (tipo === 'execucao' || tipo === 'horario') {
        try {
            const flaskUrl = `http://localhost:5000/api/grafico/${encodeURIComponent(tipo)}/${encodeURIComponent(nomeJob)}`;
            const resp = await fetch(flaskUrl, { method: 'GET' });
            if (!resp.ok) {
                const texto = await resp.text();
                console.error(`[server.js] erro proxy para Flask: ${resp.status} - ${texto}`);
                return res.status(502).json({ erro: 'Falha ao gerar gráfico via serviço Python (Flask).' });
            }
            const json = await resp.json();
            // Retorna diretamente o JSON recebido do Flask (fig object ou string JSON dependendo da implementação)
            return res.json(json);
        } catch (err) {
            console.error(`[server.js] exceção ao conectar no Flask: ${err}`);
            return res.status(502).json({ erro: 'Serviço de previsão indisponível (Flask).' });
        }
    }

    // Para os demais tipos mantém o comportamento original (gera PNG com visualizacao.py)
    const nomeArquivoImagem = `grafico_${tipo}_${nomeJob}.png`;
    const caminhoImagem = path.join(__dirname, '..', 'client', 'public', nomeArquivoImagem); // Salva na pasta public do React

    // Comando para executar o script Python, passando o caminho de saída da imagem
    const comando = `python visualizacao.py "${caminhoJson}" "${nomeJob}" "${tipo}" "${caminhoImagem}"`;

    exec(comando, { cwd: path.join(__dirname, '..') }, (error, stdout, stderr) => {
        if (error) {
            console.error(`Erro ao executar Python: ${stderr}`);
            return res.status(500).json({ erro: 'Falha ao gerar o gráfico.' });
        }

        console.log(`Python stdout: ${stdout}`);

        // Retorna o nome do arquivo de imagem para o front-end
        res.json({ urlImagem: `/${nomeArquivoImagem}` });
    });
});


// Inicia o servidor e o faz escutar na porta definida
// ESTE BLOCO DEVE ESTAR NO FIM DO ARQUIVO, FORA DE QUALQUER ROTA.
app.listen(port, () => {
    console.log(`API Server rodando em http://localhost:${port}`);
});