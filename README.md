# Deepfake Detector

## Membros do Grupo

| Nome | Número de Aluno |
|------|----------------|
| João José Martins Salgado | 29109 |
| Pedro Afonso de Morais | 31391 |

## Track

**Track A - Deep Learning**

## Descrição do Projeto

Aplicação web que analisa imagens faciais e classifica-as como **REAL** ou **FAKE**.

O modelo é um **ResNet-50** fine-tuned, treinado num conjunto alargado de datasets públicos de faces reais e deepfakes. Utiliza **MTCNN** para detetar e recortar a face da imagem antes de a passar ao classificador. O resultado inclui a probabilidade de a imagem ser falsa ou real, juntamente com o score de confiança.

A aplicação é composta por:
- Um **frontend em Next.js** onde o utilizador carrega uma imagem e vê o resultado
- Uma **API HTTP em Python** que corre a inferência com o modelo PyTorch treinado

## Tech Stack

| Camada | Tecnologia |
|--------|-----------|
| Frontend | Next.js (React) |
| API | Python com FastAPI e Uvicorn |
| Modelo | PyTorch, ResNet-50 |
| Deteção facial | MTCNN via `facenet-pytorch` |
| Treino | Kaggle (notebook em `/notebooks`) |
| Linguagem | Python 3.10+, JavaScript (Node.js 18+) |

## Modelo e Treino

### Arquitetura

O sistema usa dois componentes principais que trabalham em sequência:

**1. MTCNN — detetor de faces**

MTCNN (Multi-task Cascaded Convolutional Networks) é um algoritmo especializado em encontrar rostos em fotografias. Quando o utilizador envia uma imagem, o MTCNN procura automaticamente onde está a face, recorta essa região e descarta o resto da imagem (fundo, roupa, etc.). Isto garante que o classificador analisa apenas o rosto, e não elementos irrelevantes da fotografia.

**2. ResNet-50 — classificador**

ResNet-50 é uma rede neuronal profunda com 50 camadas, amplamente utilizada em visão computacional. Funciona de forma semelhante ao cérebro humano: foi "ensinada" a reconhecer padrões visuais em milhões de imagens.

Na prática, a rede aprendeu a detetar artefactos imperceptíveis ao olho humano que costumam aparecer em imagens fakes: inconsistências na textura da pele, irregularidades nos contornos do rosto, padrões anómalos nos olhos ou cabelo, entre outros.

**Pipeline completo:**
1. O utilizador envia uma imagem
2. **MTCNN** encontra e recorta o rosto
3. O rosto é redimensionado para 224×224 píxeis e normalizado
4. O **ResNet-50** analisa o rosto e devolve a probabilidade de ser REAL ou FAKE
5. O resultado é apresentado com o score de confiança

### Datasets utilizados

O modelo foi treinado com **6 datasets** de naturezas distintas, cobrindo diferentes técnicas de geração e manipulação de faces:

| Dataset | Tipo | Descrição |
|---------|------|-----------|
| **CelebDF** | Face Swap | Deepfakes de alta qualidade de celebridades |
| **FaceForensics++ (FF++)** | Face Swap | Manipulações faciais com múltiplos métodos (DeepFakes, Face2Face, FaceSwap, NeuralTextures) |
| **DFDC** | Face Swap | Dataset da competição Deepfake Detection Challenge da Meta |
| **140K Real and Fake Faces** | GAN | Faces geradas por GAN vs. faces reais do Flickr |
| **StyleGAN3** | GAN | Faces sintéticas geradas pelo StyleGAN3 da NVIDIA |
| **FFHQ** | Real | Flickr-Faces-HQ — dataset de referência de faces reais de alta qualidade |

### Limitações do modelo

> **Importante:** O modelo só consegue detetar aquilo para o qual foi treinado.

O modelo aprendeu padrões visuais específicos presentes nos datasets acima. Isto significa que:

- **Consegue detetar** deepfakes produzidos com técnicas semelhantes às dos datasets de treino (face swap, GANs como StyleGAN)
- **Pode falhar** em deepfakes gerados com tecnologias mais recentes ou métodos muito diferentes dos usados no treino
- **Pode falhar** em imagens com condições muito diferentes das do treino (iluminação extrema, ângulos incomuns, rostos parcialmente ocultados)
- **Não é infalível** — como qualquer modelo de classificação, tem uma taxa de erro e não deve ser usado como única fonte de decisão em contextos críticos

Em suma, o modelo é tão bom quanto os dados com que foi treinado. Técnicas de deepfake não representadas nos datasets de treino podem não ser detetadas.

## Como Executar

### 1. Clonar o repositório

```bash
git clone <repo-url>
cd deepfake-app
```

### 2. Obter o modelo treinado

O ficheiro do modelo (`best_model.pth`) não está incluído no repositório devido ao seu tamanho (~94 MB).

Abre o notebook em `/notebooks` no Kaggle. Após o treino, descarrega o `best_model.pth` gerado e coloca-o na raiz como indicado a seguir.

```
deepfake-app/
└── best_model.pth   ← colocar aqui
```

### 3. Instalar dependências Python

```bash
python3 -m venv venv
source venv/bin/activate   # no Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Instalar dependências do frontend

```bash
npm install
```

### 5. Iniciar a API

```bash
npm run api
```

A API ficará disponível em `http://localhost:8000`. Podes verificar se está a correr com:

```bash
curl http://localhost:8000/health
```

### 6. Iniciar o frontend

Noutro terminal:

```bash
npm run dev
```

Abre `http://localhost:3000` no browser, carrega uma imagem com uma face e a app indicará se parece **REAL** ou **FAKE**.

## Estrutura do Projeto

```
deepfake-app/
├── notebooks/          # Notebooks de treino e avaliação (correm no Kaggle)
├── src/
│   ├── api.py          # API de inferência em Python
│   └── app/            # Frontend Next.js
├── requirements.txt    # Dependências Python
├── package.json        # Scripts e dependências Node.js
└── best_model.pth      # Modelo treinado (não está no repo - ver "Obter o modelo treinado")
```
