# 🧭 Algoritmo A\* con Pygame

El **algoritmo A\*** (*A-star*) es un método de búsqueda de caminos usado en inteligencia artificial y videojuegos.  
Su función es **encontrar la ruta más corta** entre dos puntos evitando obstáculos, combinando **eficiencia** (velocidad) y **precisión** (distancia mínima).

Este programa lo visualiza de forma interactiva usando **Pygame**, permitiendo al usuario definir inicio, fin y paredes sobre una cuadrícula.

---

## 🎮 Librerías e inicialización

```python
import pygame

pygame.init()
icono = pygame.image.load("nucles.png")
pygame.display.set_icon(icono)
```

Inicializa **Pygame**, configura la ventana y carga un ícono personalizado.

---

## 🎨 Definición de colores

Cada color representa un estado del nodo:

| Estado | Color | Significado |
| --- | --- | --- |
| Blanco | ⚪ | Nodo vacío |
| Negro | ⬛ | Pared u obstáculo |
| Naranja | 🟧 | Punto de inicio |
| Púrpura | 🟪 | Punto final |
| Azul | 🟦 | Nodo abierto |
| Gris | ⚙️ | Nodo cerrado |
| Verde | 🟩 | Camino óptimo |

---

## 🧩 Clase `Nodo`

Representa cada casilla de la cuadrícula con su posición, color y costos A\* (`g`, `h`, `f`).

### Métodos principales:

-   `hacer_inicio()`, `hacer_fin()`, `hacer_pared()`: cambian el tipo de nodo.
    
-   `dibujar()`: renderiza el nodo en pantalla.
    
-   `evaluar_v()`: calcula vecinos válidos considerando obstáculos.
    

---

## 🧮 Heurística `h()`

```python
def h(p1, p2):
    X1, Y1 = p1.get_pos()
    X2, Y2 = p2.get_pos()
    return (abs(X1 - X2) + abs(Y1 - Y2)) * 10
```

Usa la **distancia de Manhattan** para estimar qué tan lejos está un nodo del objetivo.

---

## 🛣️ Función `camino()`

Reconstruye el trayecto más corto una vez hallado el destino, coloreando los nodos en verde 🟩.

---

## 🤖 Algoritmo `a_estrella()`

Implementa el corazón del A\*:

1.  Inicia desde el punto inicial.
    
2.  Calcula el costo total `f = g + h` para cada vecino.
    
3.  Explora el camino más prometedor.
    
4.  Finaliza cuando se llega al destino o no hay salida.
    

---

## 🧱 Funciones auxiliares

-   `crear_grid()`: genera la cuadrícula.
    
-   `dibujar_grid()`: pinta las líneas divisorias.
    
-   `dibujar()`: actualiza toda la ventana.
    
-   `obtener_click_pos()`: convierte coordenadas del mouse a celdas.
    

---

## 🕹️ Bucle principal `main()`

Controla la interacción con el usuario:

| Acción | Tecla / Click | Resultado |
| --- | --- | --- |
| 🖱️ Click izquierdo | Crear inicio, fin o pared |  |
| 🖱️ Click derecho | Borrar nodo |  |
| ⏎ Enter | Ejecutar el algoritmo A\* |  |
| ⎋ Escape | Reiniciar la cuadrícula |  |

---

## 🚀 Ejecución

```python
main(VENTANA, ANCHO_VENTANA)
```

Ejecuta la simulación y muestra el proceso de búsqueda en tiempo real.

---

## 💡 Conclusión

Este proyecto ofrece una forma visual e interactiva de comprender cómo el algoritmo **A\*** encuentra rutas óptimas en un entorno con obstáculos, ideal para aprender sobre **IA, heurísticas y pathfinding**.

---