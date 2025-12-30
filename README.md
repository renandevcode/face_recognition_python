# Face Recognition with Eigenfaces and LBPH

Este projeto implementa um sistema de **reconhecimento facial clássico**, utilizando dois métodos consagrados da visão computacional:

- **Eigenfaces (PCA)**  
- **LBPH (Local Binary Patterns Histograms)**  

O objetivo é testar implementação e comparar abordagens **estatísticas e baseadas em textura**, entendendo seus fundamentos matemáticos e limitações, sem o uso de deep learning.

---

## 📌 Motivação

Antes de modelos baseados em redes neurais convolucionais, técnicas como Eigenfaces e LBPH foram amplamente utilizadas em sistemas reais de reconhecimento facial.

Este projeto foi desenvolvido com foco em:
- Compreensão teórica dos algoritmos
- Implementação prática
- Análise das vantagens e desvantagens de cada método

---

## 🧠 Métodos Utilizados

### 🔹 Eigenfaces (PCA)
- Baseado em **Análise de Componentes Principais**
- Reduz a dimensionalidade das imagens
- Representa faces como combinações lineares de autovetores
- Sensível a variações de iluminação e pose

### 🔹 LBPH
- Baseado em **padrões locais de textura**
- Calcula histogramas de padrões binários
- Mais robusto a variações de iluminação
- Muito utilizado em aplicações em tempo real

---

## 🛠️ Tecnologias e Bibliotecas

- Python 3
- OpenCV
- NumPy
- Matplotlib (para visualização)

---

## Uso do programa
1 - Primeiro faça as capturas salvando as imagens das faces detectadas:

`python detect_face.py`

digite: um numero para ser o identificador da face e o nome do responsável daquela face
clique na tecla TAB para iniciar a sessão de capturas
clique na tecla "q" para salvar a imagem da face detectada.

2 - Faça o apredizado das faces detectadas:

`python treinamento.py`

3 - Execute o reconhecedor no terminal:

`python reconhecedor_eigenfaces.py` | `python reconhecedor_lbph.py`


---

## 📊 Resultados e Observações

- **Eigenfaces**
  - Bom desempenho em ambientes controlados
  - Sensível a ruído, iluminação e expressões faciais

- **LBPH**
  - Melhor desempenho em cenários mais variados
  - Mais robusto para aplicações simples e em tempo real

O projeto demonstra por que métodos clássicos ainda são importantes para estudo e compreensão da evolução do reconhecimento facial.

---

## 📁 Estrutura do Projeto

