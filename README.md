# Deepfake Detector

| Nome | Numero |
| --- | --- |
| João José Martins Salgado | 29109 |
| Pedro Afonso de Morais | 31391 |

Aplicacao web para analisar imagens faciais e estimar se parecem reais ou geradas/manipuladas por IA.

## Estrutura

- Frontend em Next.js, na pasta `src/app/`
- API Python em `src/api.py`
- Modelo PyTorch esperado em `best_model.pth`

## Requisitos

- Node.js 18+
- Python 3.10+
- Modelo treinado `best_model.pth`

## Instalacao

Instalar dependencias do frontend:

```bash
npm install
```

Criar e ativar ambiente Python:

```bash
python3 -m venv venv
source venv/bin/activate
```

Instalar dependencias Python:

```bash
pip install -r requirements.txt
```

## Modelo

Coloca o ficheiro do modelo na raiz do projeto com este nome:

```text
best_model.pth
```

Por defeito, o modelo nao e enviado para o GitHub porque esta ignorado no `.gitignore`.
Se quiseres usar outro caminho, define a variavel `MODEL_PATH`:

```bash
MODEL_PATH=/caminho/para/modelo.pth npm run api
```

## Como correr

Arrancar a API Python:

```bash
npm run api
```

A API fica disponivel em:

```text
http://localhost:8000
```

Verificar se a API esta ligada:

```bash
curl http://localhost:8000/health
```

Noutro terminal, arrancar o frontend:

```bash
npm run dev
```

Abrir no browser:

```text
http://localhost:3000
```

## Endpoints da API

### `GET /health`

Verifica se a API esta online e se encontra o modelo.

### `POST /predict`

Analisa uma imagem enviada como `multipart/form-data`, no campo `file`.

Exemplo:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@imagem.jpg"
```

## Configuracao

Variaveis suportadas:

```bash
PORT=8000
MODEL_PATH=best_model.pth
FAKE_CLASS_INDEX=1
NEXT_PUBLIC_DETECT_API_URL=http://localhost:8000/predict
```

## Scripts

```bash
npm run dev      # frontend Next.js
npm run api      # API Python
npm run build    # build do frontend
npm run start    # correr build do frontend
```
