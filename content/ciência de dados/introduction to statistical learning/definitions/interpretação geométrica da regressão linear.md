

### Hiperplano com constante incluída

Quando adicionamos a coluna de 1 em $X$ (ou seja, incluímos o intercepto dentro do vetor de coeficientes), estamos dizendo ao modelo: "posso deslocar o hiperplano livremente no eixo Y".
**[[Subespaço]]**: se não tivermos intercepto (ou seja, nenhuma constante), o [[hiperplano]] **passa pela origem** $(0,0,...,0)$. Isso é um subespaço, porque não há deslocamento no eixo Y.

### Hiperplano sem constante

Nesse caso, o modelo força o hiperplano a passar por zero quando todas as entradas $X = 0$. Então o hiperplano corta o eixo Y exatamente no ponto $\hat{\beta}_0$, que é o valor da saída quando $X = 0$. Esse tipo de conjunto é chamado de **affine set**, ou seja, um hiperplano deslocado do zero.

**Exemplo em 2D** (uma variável X e uma saída Y):
- Com intercepto: $Y = 2 + 3X$ → corta Y em 2
- Sem intercepto: $Y = 3X$ → passa pela origem $(0,0)$

### Gradiente e Direção de Crescimento

#### Gradiente $\nabla f(X)$

Dada a função linear:
$f(X) = X^T \beta = \beta_1 x_1 + ... + \beta_p x_p$
O [[gradiente]] de uma função é um vetor que **aponta na direção em que a função cresce mais rápido**. Para uma função linear, o gradiente não depende de $X$; ele é simplesmente $\beta$.

Cada componente $\beta_i$ diz: "se eu aumentar $x_i$ em 1 unidade, a saída $f(X)$ aumenta em $\beta_i$".

**Exemplo em 2D** ($X = [x_1, x_2]$):

$f(x_1, x_2) = 2x_1 + 3x_2$

- Gradiente: $\nabla f = [2, 3]$
- Significado: se você andar na direção $(x_1, x_2) = (2, 3)$, a função sobe mais rápido