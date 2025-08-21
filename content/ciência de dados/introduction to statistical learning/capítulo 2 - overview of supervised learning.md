## Introdução
Os tipos de variáveis de saída (outputs) podem variar significativamente dependendo do exemplo. Em um caso de predição de glicose, por exemplo, o output é **quantitativo**, ou seja, uma medida numérica contínua em que valores maiores indicam maior magnitude e valores próximos possuem similaridade natural. Já no famoso exemplo da classificação de espécies de íris, desenvolvido por R. A. Fisher, o output é **qualitativo**, assumindo valores em um conjunto finito $G = {Virginica, Setosa, Versicolor}$. Outro exemplo é a classificação de dígitos manuscritos, onde o output pertence a um conjunto de 10 classes $G = {0, 1, \dots, 9}$. Nestes casos, não há uma ordenação explícita entre as classes; muitas vezes utilizam-se **labels descritivos** em vez de números. Por isso, variáveis qualitativas também são chamadas de **categóricas**, **discretas** ou **fatores**.

A distinção entre tipos de saída também levou a diferentes abordagens de predição. Quando o objetivo é prever saídas **quantitativas**, o problema é denominado **regressão**. Quando se deseja prever saídas **qualitativas**, o problema é denominado **classificação**. Apesar dessa diferença, ambos os tipos de tarefa podem ser entendidos como um problema de **aproximação de função**, onde queremos mapear entradas para saídas com base em dados observados.

As variáveis de entrada (inputs) também podem ser de tipos variados. Podemos ter entradas **quantitativas**, **qualitativas**, ou uma combinação de ambas. A escolha do método de predição geralmente depende do tipo de input: alguns métodos são mais adequados para entradas quantitativas, outros para qualitativas, e alguns funcionam bem para ambos os tipos.

Além disso, existe um tipo especial de variável categórica conhecida como **categórica ordenada**, na qual há uma ordem natural entre os níveis, mas as diferenças não possuem significado métrico uniforme. Por exemplo, uma variável com níveis "pequeno", "médio" e "grande" é ordenada, mas a diferença entre "médio" e "pequeno" não precisa ser igual à diferença entre "grande" e "médio". Este tipo de variável é discutido com mais detalhes no [[capítulo 4 - linear methods for classification]].

Para lidar com variáveis qualitativas numericamente, utiliza-se a **codificação**. O caso mais simples é quando há apenas duas classes, como "sucesso" ou "falha", ou "sobreviveu" ou "morreu". Nestes casos, a variável pode ser codificada como $0/1$ ou $-1/1$, sendo este valor chamado de **target**. Para variáveis com mais de duas categorias, a abordagem mais comum é a utilização de **[[dummy variables]]**, onde uma variável qualitativa com K níveis é representada por um vetor de K bits, com apenas um bit "on" de cada vez. Embora existam formas mais compactas de codificação, o uso de dummy variables mantém simetria entre os níveis da variável.

A notação formal utilizada é a seguinte: denotamos variáveis de entrada por $X$, e, se $X$ for um vetor, suas componentes são acessadas como $X_j$. A saída quantitativa é denotada por $Y$, e a saída qualitativa por $G$. Quando nos referimos a valores observados, utilizamos letras minúsculas, como $x_i$, $y_i$ ou $g_i$, representando a i-ésima observação de cada variável. Matrizes são representadas por letras maiúsculas em negrito, por exemplo, o conjunto de N vetores de entrada de dimensão p é representado por $\mathbf{X}$, cuja i-ésima linha é $x_i^T$, a transposta do vetor de entrada. Vetores que representam todas as observações de uma variável, como $X_j$, têm N componentes e são também considerados vetores coluna.

O objetivo do aprendizado é, dado um vetor de entrada $X$, fazer uma boa predição da saída $Y$ ou $G$. Para saídas quantitativas, a predição é denotada por $\hat{Y}$ ("y-hat"), e para saídas qualitativas, por $\hat{G}$. No caso binário, é comum tratar a saída como quantitativa e predizer $\hat{Y} \in [0,1]$, atribuindo a classe correspondente de acordo com um limiar, geralmente 0,5:

$$\hat{G} = \left\{ \begin{aligned} 1 &\quad \text{se } \hat{y} > 0.5 \\ 0 &\quad \text{caso contrário} \end{aligned} \right.$$

Este método se generaliza para variáveis qualitativas com K níveis.

Para construir regras de predição, são necessários **dados de treinamento**, consistindo em conjuntos de medições $(x_i, y_i)$ ou $(x_i, g_i)$, para $i = 1, \dots, N$. Estes dados são utilizados para aproximar a função que mapeia entradas para saídas, permitindo realizar previsões em novos casos.

---
## Modelo Linear

O modelo linear tem sido fundamental em estatística nos últimos 30 anos e continua sendo uma das ferramentas mais importantes. Dado um vetor de entradas $X^T = (X_1, X_2, \ldots, X_p)$, predizemos a saída $Y$ através do modelo:

$$\hat{Y} = \hat{\beta}_0 + \sum_{j=1}^p X_j \hat{\beta}_j$$

onde $\hat{\beta}_0$ é o intercepto, também conhecido como **bias** em machine learning.

Para simplificar a notação, é comum incluir uma variável constante 1 em $X$ e incorporar $\hat{\beta}_0$ no vetor de coeficientes $\hat{\beta}$, permitindo escrever o modelo linear em forma vetorial como um produto interno: $\hat{Y} = X^T \hat{\beta}$. Este formato considera uma única saída escalar, mas pode ser generalizado para $K$ saídas, caso em que $\beta$ seria uma matriz $p \times K$ de coeficientes.

#### Interpretação Geométrica

Geometricamente, no espaço $(p+1)$-dimensional entrada-saída, $(X, \hat{Y})$ representa um **[[hiperplano]]**. Se a constante está incluída em $X$, o hiperplano passa pela origem formando um [[subespaço]]; caso contrário, é um conjunto afim que corta o eixo $Y$ no ponto $(0, \hat{\beta}_0)$. 
Visto como uma função sobre o espaço $p$-dimensional de entrada, $f(X) = X^T \beta$ é linear, e o gradiente $f'(X) = \beta$ é um vetor no espaço de entrada que aponta na direção de maior crescimento. 
explicação: [[interpretação geométrica da regressão linear]]

#### Ajuste por Mínimos Quadrados

O método mais popular para ajustar o modelo linear aos dados de treinamento é o **[[método dos mínimos quadrados]]**. Nesta abordagem, escolhemos os coeficientes $\beta$ para minimizar a soma residual dos quadrados:

$$RSS(\beta) = \sum_{i=1}^N (y_i - x_i^T \beta)^2$$

A função $RSS(\beta)$ é quadrática nos parâmetros, garantindo que seu mínimo sempre existe, embora possa não ser único. A solução é mais facilmente caracterizada em notação matricial como $RSS(\beta) = (y - X\beta)^T (y - X\beta)$, onde $X$ é uma matriz $N \times p$ com cada linha sendo um vetor de entrada, e $y$ é um vetor $N$-dimensional das saídas no conjunto de treinamento.

Diferenciando em relação a $\beta$, obtemos as **equações normais**: $X^T (y - X\beta) = 0$. Se $X^T X$ for [[não-singular]], a solução única é dada por:

$$\hat{\beta} = (X^T X)^{-1} X^T y$$

O valor ajustado na $i$-ésima entrada $x_i$ é $\hat{y}_i = x_i^T \hat{\beta}$, e para uma entrada arbitrária $x_0$, a predição é $\hat{y}(x_0) = x_0^T \hat{\beta}$. Toda a superfície ajustada é caracterizada pelos $p$ parâmetros $\hat{\beta}$, sugerindo intuitivamente que não precisamos de um conjunto de dados muito grande para ajustar tal modelo.



Considere um exemplo de classificação com **modelo linear**. Suponha que temos dados de treinamento em duas entradas, $X_1$ e $X_2$, representados em um scatterplot. A variável de saída $G$ assume duas classes: **AZUL** e **LARANJA**, com 100 pontos em cada classe. 
![[Imagem 2.1.png]]
Para aplicar regressão linear, a saída $Y$ é codificada como $Y=0$ para AZUL e $Y=1$ para LARANJA. O modelo linear é ajustado aos dados e gera predições $\hat{Y}$. A predição da classe $\hat{G}$ é obtida pelo critério:

$$
\hat{G} =
\begin{cases}
\text{LARANJA} & \text{se } \hat{Y} > 0.5\\
\text{AZUL} & \text{se } \hat{Y} \leq 0.5
\end{cases}
$$

A fronteira de decisão linear é definida por $\{x : x^T \hat{\beta} = 0.5\}$, onde a região laranja no espaço de entrada representa $\{x : x^T \hat{\beta} > 0.5\}$ (classe LARANJA) e a região azul representa $\{x : x^T \hat{\beta} \leq 0.5\}$ (classe AZUL).
Mesmo com o modelo ajustado, algumas **classificações incorretas** podem ocorrer nos dados de treinamento. Isso pode indicar que o modelo linear é muito rígido ou que certos erros são inevitáveis.

#### Cenários de geração de dados

1. **Distribuição Gaussiana por classe**: cada classe é gerada por uma [[distribuição bivariada Gaussiana]], com médias diferentes e componentes não correlacionados. Para este caso, a fronteira de decisão linear é quase ótima; a região de sobreposição é inevitável e erros são esperados mesmo para novos dados.

2. **Mistura de Gaussianas**: cada classe é gerada por uma mistura de 10 [[distribuições Gaussianas]] de baixa variância. O modelo linear não é ótimo; a fronteira ideal é **não linear** e **disjunta**, sendo mais difícil de estimar.

O modelo linear é adequado para o **primeiro cenário**, mas pode falhar no **segundo cenário**, onde métodos mais flexíveis, como vizinhos mais próximos ([[k-nearest neighbors (KNN)]]) ou outros modelos não lineares, são mais indicados.
