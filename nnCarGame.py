import pygame
import random
import math
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from PIL import Image

# Inicializar pygame
pygame.init()

# Inicialización de algunas variables
img = 0  # Variable usada para grabar frames
size = width, height = 1600, 900  # Tamaño de la ventana de pygame

# Colores
white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)
black = (0, 0, 0)
gray = pygame.Color('gray12')
Color_line = (255, 0, 0)

# Inicialización de parámetros del juego
generation = 1
mutationRate = 90
FPS = 30
game_speed = 1  # Velocidad del juego (1 = normal, 2 = x2)
selectedCars = []
selected = 0
lines = True  # Si es True, se muestran las líneas de los sensores del carro
player = True  # Si es True, se muestra el carro del jugador
display_info = True  # Si es True, se muestra la información en pantalla
frames = 0
maxSpeed = 10
number_track = 1

# Carga de imágenes de los carros
white_small_car = pygame.image.load('Images/Sprites/white_small.png')
white_big_car = pygame.image.load('Images/Sprites/white_big.png')
green_small_car = pygame.image.load('Images/Sprites/green_small.png')
green_big_car = pygame.image.load('Images/Sprites/green_big.png')

# Carga de imágenes de fondo
bg = pygame.image.load('bg7.png')
bg4 = pygame.image.load('bg4.png')


# Función para calcular la distancia entre dos puntos
def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# Función para rotar un punto alrededor de un origen dado
def rotation(origin, tempPoint, angle):
    ox, oy = origin
    px, py = tempPoint

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


# Función para mover un punto en una dirección dada
def move(tempPoint, angle, unit):
    x = tempPoint[0]
    y = tempPoint[1]
    rad = math.radians(-angle % 360)

    x += unit * math.sin(rad)
    y += unit * math.cos(rad)

    return x, y


# Función sigmoide utilizada como función de activación en la red neuronal
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Función para mutar un gen (peso) en un hijo basado en un padre
def mutateOneWeightGene(parent1, child1):
    sizeNN = len(child1.sizes)

    # Copia los pesos del padre al hijo
    for i in range(sizeNN - 1):
        for j in range(child1.sizes[i + 1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]

    # Copia los sesgos del padre al hijo
    for i in range(sizeNN - 1):
        for j in range(child1.sizes[i + 1]):
            child1.biases[i][j] = parent1.biases[i][j]

    genomeWeights = []  # Lista que contiene todos los pesos, más fácil de modificar

    # Copia los pesos en una lista
    for i in range(sizeNN - 1):
        for j in range(child1.sizes[i] * child1.sizes[i + 1]):
            genomeWeights.append(child1.weights[i].item(j))

    # Modifica un gen aleatorio por una cantidad aleatoria
    r1 = random.randint(0, len(genomeWeights) - 1)
    genomeWeights[r1] = genomeWeights[r1] * random.uniform(0.8, 1.2)

    # Vuelve a copiar los pesos en el hijo desde la lista modificada
    count = 0
    for i in range(sizeNN - 1):
        for j in range(child1.sizes[i + 1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = genomeWeights[count]
                count += 1
    return


# Función para mutar un gen (sesgo) en un hijo basado en un padre
def mutateOneBiasesGene(parent1, child1):
    sizeNN = len(child1.sizes)

    # Copia los pesos del padre al hijo
    for i in range(sizeNN - 1):
        for j in range(child1.sizes[i + 1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]

    # Copia los sesgos del padre al hijo
    for i in range(sizeNN - 1):
        for j in range(child1.sizes[i + 1]):
            child1.biases[i][j] = parent1.biases[i][j]

    genomeBiases = []  # Lista que contiene todos los sesgos

    # Copia los sesgos en una lista
    for i in range(sizeNN - 1):
        for j in range(child1.sizes[i + 1]):
            genomeBiases.append(child1.biases[i].item(j))

    # Modifica un gen aleatorio por una cantidad aleatoria
    r1 = random.randint(0, len(genomeBiases) - 1)
    genomeBiases[r1] = genomeBiases[r1] * random.uniform(0.8, 1.2)

    # Vuelve a copiar los sesgos en el hijo desde la lista modificada
    count = 0
    for i in range(sizeNN - 1):
        for j in range(child1.sizes[i + 1]):
            child1.biases[i][j] = genomeBiases[count]
            count += 1
    return


# Función de cruce uniforme de pesos entre dos padres para crear dos hijos
def uniformCrossOverWeights(parent1, parent2, child1, child2):
    sizeNN = len(child1.sizes)

    # Copia los pesos del padre 1 al hijo 1 y del padre 2 al hijo 2
    for i in range(sizeNN - 1):
        for j in range(child1.sizes[i + 1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]

    for i in range(sizeNN - 1):
        for j in range(child1.sizes[i + 1]):
            for k in range(child1.sizes[i]):
                child2.weights[i][j][k] = parent2.weights[i][j][k]

    # Copia los sesgos del padre 1 al hijo 1 y del padre 2 al hijo 2
    for i in range(sizeNN - 1):
        for j in range(child2.sizes[i + 1]):
            child1.biases[i][j] = parent1.biases[i][j]

    for i in range(sizeNN - 1):
        for j in range(child2.sizes[i + 1]):
            child2.biases[i][j] = parent2.biases[i][j]

    genome1 = []  # Lista que contiene todos los pesos del hijo 1
    genome2 = []  # Lista que contiene todos los pesos del hijo 2

    # Copia los pesos en listas
    for i in range(sizeNN - 1):
        for j in range(child1.sizes[i] * child1.sizes[i + 1]):
            genome1.append(child1.weights[i].item(j))

    for i in range(sizeNN - 1):
        for j in range(child2.sizes[i] * child2.sizes[i + 1]):
            genome2.append(child2.weights[i].item(j))

    # Realiza el cruce uniforme de los pesos
    alter = True
    for i in range(len(genome1)):
        if alter:
            aux = genome1[i]
            genome1[i] = genome2[i]
            genome2[i] = aux
            alter = False
        else:
            alter = True

    # Vuelve a copiar los pesos en los hijos desde las listas modificadas
    count = 0
    for i in range(sizeNN - 1):
        for j in range(child1.sizes[i + 1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = genome1[count]
                count += 1

    count = 0
    for i in range(sizeNN - 1):
        for j in range(child2.sizes[i + 1]):
            for k in range(child2.sizes[i]):
                child2.weights[i][j][k] = genome2[count]
                count += 1
    return


# Función de cruce uniforme de sesgos entre dos padres para crear dos hijos
def uniformCrossOverBiases(parent1, parent2, child1, child2):
    sizeNN = len(parent1.sizes)

    # Copia los pesos del padre 1 al hijo 1 y del padre 2 al hijo 2
    for i in range(sizeNN - 1):
        for j in range(child1.sizes[i + 1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]

    for i in range(sizeNN - 1):
        for j in range(child1.sizes[i + 1]):
            for k in range(child1.sizes[i]):
                child2.weights[i][j][k] = parent2.weights[i][j][k]

    # Copia los sesgos del padre 1 al hijo 1 y del padre 2 al hijo 2
    for i in range(sizeNN - 1):
        for j in range(child2.sizes[i + 1]):
            child1.biases[i][j] = parent1.biases[i][j]

    for i in range(sizeNN - 1):
        for j in range(child2.sizes[i + 1]):
            child2.biases[i][j] = parent2.biases[i][j]

    genome1 = []  # Lista que contiene todos los sesgos del hijo 1
    genome2 = []  # Lista que contiene todos los sesgos del hijo 2

    # Copia los sesgos en listas
    for i in range(sizeNN - 1):
        for j in range(child1.sizes[i + 1]):
            genome1.append(child1.biases[i].item(j))

    for i in range(sizeNN - 1):
        for j in range(child2.sizes[i + 1]):
            genome2.append(child2.biases[i].item(j))

    # Realiza el cruce uniforme de los sesgos
    alter = True
    for i in range(len(genome1)):
        if alter:
            aux = genome1[i]
            genome1[i] = genome2[i]
            genome2[i] = aux
            alter = False
        else:
            alter = True

    # Vuelve a copiar los sesgos en los hijos desde las listas modificadas
    count = 0
    for i in range(sizeNN - 1):
        for j in range(child1.sizes[i + 1]):
            child1.biases[i][j] = genome1[count]
            count += 1

    count = 0
    for i in range(sizeNN - 1):
        for j in range(child2.sizes[i + 1]):
            child2.biases[i][j] = genome2[count]
            count += 1
    return


# Función para generar un mapa aleatorio
def generateRandomMap(screen):
    SCREEN = screen

    # Parámetros del mapa
    GREEN = (0, 255, 128)
    WINDOW_HEIGHT = 730
    WINDOW_WIDTH = 1460  # Estos parámetros son para la cuadrícula, no para la ventana de pygame
    blockSize = 146  # Tamaño de cada bloque de la cuadrícula
    rows, cols = (int(WINDOW_WIDTH / blockSize), int(WINDOW_HEIGHT / blockSize))
    maze = Maze(rows, cols, 0, 0)

    trackLength = 1
    moveX = 70
    moveY = 85

    # Elige una celda de inicio
    startX, startY = 0, 3
    currentCell = maze.cell_at(startX, startY)

    # Carga de imágenes de las pistas
    straight1 = pygame.image.load('Images/TracksMapGen/Straight1.png')
    straight1Rect = straight1.get_rect()

    straight2 = pygame.image.load('Images/TracksMapGen/Straight2.png')
    straight2Rect = straight2.get_rect()

    curve1 = pygame.image.load('Images/TracksMapGen/Curve1.png')
    curve1Rect = curve1.get_rect()

    curve2 = pygame.image.load('Images/TracksMapGen/Curve2.png')
    curve2Rect = curve2.get_rect()

    curve3 = pygame.image.load('Images/TracksMapGen/Curve3.png')
    curve3Rect = curve3.get_rect()

    curve4 = pygame.image.load('Images/TracksMapGen/Curve4.png')
    curve4Rect = curve4.get_rect()

    straight1Top = pygame.image.load('Images/TracksMapGen/Straight1Top.png')
    straight1RectTop = straight1Top.get_rect()

    straight2Top = pygame.image.load('Images/TracksMapGen/Straight2Top.png')
    straight2RectTop = straight2Top.get_rect()

    curve1Top = pygame.image.load('Images/TracksMapGen/Curve1Top.png')
    curve1RectTop = curve1Top.get_rect()

    curve2Top = pygame.image.load('Images/TracksMapGen/Curve2Top.png')
    curve2RectTop = curve2Top.get_rect()

    curve3Top = pygame.image.load('Images/TracksMapGen/Curve3Top.png')
    curve3RectTop = curve3Top.get_rect()

    curve4Top = pygame.image.load('Images/TracksMapGen/Curve4Top.png')
    curve4RectTop = curve4Top.get_rect()

    initialTop = pygame.image.load('Images/TracksMapGen/Initial.png')
    initialRectTop = initialTop.get_rect()

    bg = pygame.image.load('Images/TracksMapGen/Background.png')

    while True:
        # Si la celda actual tiene vecinos no visitados
        if len(maze.find_valid_neighbours(currentCell)) > 0:
            # Si la celda actual está en la posición inicial
            if currentCell.x == 0 and currentCell.y == 3:
                oldCell = currentCell
                currentCell = maze.cell_at(oldCell.x, oldCell.y - 1)
                currentCell.color = GREEN
                oldCell.knock_down_wall(currentCell, "N")
                trackLength += 1  # Aumenta la longitud del recorrido
            else:
                # Elige una dirección aleatoria para moverse a una celda vecina no visitada
                random_unvisited_direction = random.choice(maze.find_valid_neighbours(currentCell))[0]
                oldCell = currentCell
                if random_unvisited_direction == "N":  # Moverse hacia el norte
                    currentCell = maze.cell_at(oldCell.x, oldCell.y - 1)
                elif random_unvisited_direction == "S":  # Moverse hacia el sur
                    currentCell = maze.cell_at(oldCell.x, oldCell.y + 1)
                elif random_unvisited_direction == "E":  # Moverse hacia el este
                    currentCell = maze.cell_at(oldCell.x + 1, oldCell.y)
                elif random_unvisited_direction == "W":  # Moverse hacia el oeste
                    currentCell = maze.cell_at(oldCell.x - 1, oldCell.y)

                oldCell.knock_down_wall(currentCell, random_unvisited_direction)
                trackLength += 1
        else:
            # Si se ha regresado a la posición inicial y la longitud del recorrido es suficiente
            if currentCell.x == 0 and currentCell.y == 4 and trackLength > 40:
                SCREEN.fill((0, 0, 0))
                currentCell.knock_down_wall(maze.cell_at(0, 3), "N")

                # Pintar el mapa en pantalla
                for x in range(0, WINDOW_WIDTH, blockSize):
                    for y in range(0, WINDOW_HEIGHT, blockSize):
                        currentCell = maze.cell_at(int(x / blockSize), int(y / blockSize))
                        currentCell.color = (0, 0, 1, 255)

                # Colocar las pistas en la pantalla
                for x in range(0, WINDOW_WIDTH, blockSize):
                    for y in range(0, WINDOW_HEIGHT, blockSize):
                        currentCell = maze.cell_at(int(x / blockSize), int(y / blockSize))

                        if not currentCell.walls["N"] and not currentCell.walls["S"]:
                            SCREEN.blit(straight2, straight2Rect.move(x + moveX, y + moveY))
                        elif not currentCell.walls["E"] and not currentCell.walls["W"]:
                            SCREEN.blit(straight1, straight1Rect.move(x + moveX, y + moveY))
                        elif not currentCell.walls["N"] and not currentCell.walls["W"]:
                            SCREEN.blit(curve3, curve3Rect.move(x + moveX, y + moveY))
                        elif not currentCell.walls["W"] and not currentCell.walls["S"]:
                            SCREEN.blit(curve2, curve2Rect.move(x + moveX, y + moveY))
                        elif not currentCell.walls["S"] and not currentCell.walls["E"]:
                            SCREEN.blit(curve1, curve1Rect.move(x + moveX, y + moveY))
                        elif not currentCell.walls["E"] and not currentCell.walls["N"]:
                            SCREEN.blit(curve4, curve4Rect.move(x + moveX, y + moveY))

                # Guardar el mapa generado y hacer el fondo transparente
                pygame.image.save(SCREEN, "randomGeneratedTrackBack.png")
                img = Image.open("randomGeneratedTrackBack.png")
                img = img.convert("RGBA")
                pixData = img.load()
                for y in range(img.size[1]):
                    for x in range(img.size[0]):
                        if pixData[x, y] == (0, 0, 0, 255) or pixData[x, y] == (0, 0, 1, 255):
                            pixData[x, y] = (0, 0, 0, 0)
                img.save("randomGeneratedTrackBack.png")

                # Colocar la capa superior de las pistas
                SCREEN.blit(bg, (0, 0))
                for x in range(0, WINDOW_WIDTH, blockSize):
                    for y in range(0, WINDOW_HEIGHT, blockSize):
                        if x == 0 and y == 3 * blockSize:
                            SCREEN.blit(initialTop, initialRectTop.move(x - 20 + moveX, y + moveY))
                        else:
                            currentCell = maze.cell_at(int(x / blockSize), int(y / blockSize))
                            if not currentCell.walls["N"] and not currentCell.walls["S"]:
                                SCREEN.blit(straight2Top, straight2RectTop.move(x - 20 + moveX, y + moveY))
                            elif not currentCell.walls["E"] and not currentCell.walls["W"]:
                                SCREEN.blit(straight1Top, straight1RectTop.move(x + moveX, y - 20 + moveY))
                            elif not currentCell.walls["N"] and not currentCell.walls["W"]:
                                SCREEN.blit(curve3Top, curve3RectTop.move(x - 15 + moveX, y - 15 + moveY))
                            elif not currentCell.walls["W"] and not currentCell.walls["S"]:
                                SCREEN.blit(curve2Top, curve2RectTop.move(x - 15 + moveX, y - 15 + moveY))
                            elif not currentCell.walls["E"] and not currentCell.walls["N"]:
                                SCREEN.blit(curve4Top, curve4RectTop.move(x - 15 + moveX, y - 15 + moveY))
                            elif not currentCell.walls["S"] and not currentCell.walls["E"]:
                                SCREEN.blit(curve1Top, curve1RectTop.move(x - 15 + moveX, y - 15 + moveY))

                # Guardar el mapa completo
                pygame.image.save(SCREEN, "randomGeneratedTrackFront.png")

                break

            else:
                # Si no es lo suficientemente largo, reiniciar el proceso
                trackLength = 0
                for x in range(0, WINDOW_WIDTH, blockSize):
                    for y in range(0, WINDOW_HEIGHT, blockSize):
                        maze.cell_at(int(x / blockSize), int(y / blockSize)).walls["N"] = True
                        maze.cell_at(int(x / blockSize), int(y / blockSize)).walls["S"] = True
                        maze.cell_at(int(x / blockSize), int(y / blockSize)).walls["E"] = True
                        maze.cell_at(int(x / blockSize), int(y / blockSize)).walls["W"] = True
                        maze.cell_at(int(x / blockSize), int(y / blockSize)).color = 0, 0, 0

                # Forzar celdas ocupadas
                maze.cell_at(3, 3).walls["N"] = False
                maze.cell_at(4, 3).walls["N"] = False
                maze.cell_at(5, 3).walls["N"] = False
                maze.cell_at(6, 3).walls["N"] = False

                currentCell = maze.cell_at(startX, startY)
    return


# Clase que representa una celda en el mapa
class Cell:
    # Un muro separa un par de celdas en las direcciones N-S o W-E
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

    def __init__(self, x, y):
        self.x, self.y = x, y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}
        self.color = 0, 0, 0
        self.track = ""

    # Verifica si una celda tiene todos sus muros intactos
    def has_all_walls(self):
        return all(self.walls.values())

    # Derriba un muro entre la celda actual y otra
    def knock_down_wall(self, other, wall):
        self.walls[wall] = False
        other.walls[Cell.wall_pairs[wall]] = False


# Clase que representa un laberinto como una cuadrícula de celdas
class Maze:
    def __init__(self, nx, ny, ix=0, iy=0):
        self.nx, self.ny = nx, ny
        self.ix, self.iy = ix, iy
        self.maze_map = [[Cell(x, y) for y in range(ny)] for x in range(nx)]

    # Obtiene la celda en una posición específica
    def cell_at(self, x, y):
        return self.maze_map[x][y]

    # Encuentra los vecinos válidos (no visitados) de una celda
    def find_valid_neighbours(self, cell):
        delta = [('W', (-1, 0)),
                 ('E', (1, 0)),
                 ('S', (0, 1)),
                 ('N', (0, -1))]
        neighbours = []
        for direction, (dx, dy) in delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.cell_at(x2, y2)
                if neighbour.has_all_walls():
                    neighbours.append((direction, neighbour))
        return neighbours


# Clase que representa un carro en el juego
class Car:
    def __init__(self, sizes):
        self.score = 0
        self.num_layers = len(sizes)  # Número de capas en la red neuronal
        self.sizes = sizes  # Lista con el número de neuronas por capa
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # Sesgos
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]  # Pesos
        # c1, c2, c3, c4, c5 son cinco puntos 2D donde el carro podría colisionar, actualizados en cada frame
        self.c1 = 0, 0
        self.c2 = 0, 0
        self.c3 = 0, 0
        self.c4 = 0, 0
        self.c5 = 0, 0
        # d1, d2, d3, d4, d5 son las distancias desde el carro a esos puntos, actualizadas en cada frame y usadas como
        # entradas para la red neuronal
        self.d1 = 0
        self.d2 = 0
        self.d3 = 0
        self.d4 = 0
        self.d5 = 0
        self.yaReste = False
        # Las entradas y salidas de la red neuronal deben estar en formato numpy array
        self.inp = np.array([[self.d1], [self.d2], [self.d3], [self.d4], [self.d5]])
        self.outp = np.array([[0], [0], [0], [0]])
        # Booleano para alternar la visualización de las líneas de distancia
        self.showLinesBool = False
        # Ubicación inicial del carro
        self.x = 120
        self.y = 480
        self.center = self.x, self.y
        # Altura y anchura del carro
        self.height = 35
        self.width = 17
        # Estos son los cuatro vértices del carro, usando objetos de tipo polígono en lugar de rectángulo
        self.d = self.x - (self.width / 2), self.y - (self.height / 2)
        self.c = self.x + self.width - (self.width / 2), self.y - (self.height / 2)
        self.b = self.x + self.width - (self.width / 2), self.y + self.height - (self.height / 2)
        self.a = self.x - (self.width / 2), self.y + self.height - (self.height / 2)
        # Velocidad, aceleración y dirección del carro
        self.velocity = 0
        self.acceleration = 0
        self.angle = 180
        # Booleano que se vuelve True cuando el carro colisiona
        self.collided = False
        # Color e imagen del carro
        self.color = white
        self.car_image = white_small_car
        self.distance_traveled = 0  # Distancia recorrida
        self.time_taken = 0  # Tiempo tomado
        self.finished = False  # Indica si el coche terminó la pista
        self.start_time = pygame.time.get_ticks()  # Tiempo de inicio

    # Establece la aceleración del carro
    def set_accel(self, accel):
        self.acceleration = accel

    # Rota el carro
    def rotate(self, rot):
        self.angle += rot
        if self.angle > 360:
            self.angle = 0
        if self.angle < 0:
            self.angle = 360 + self.angle

    # Actualiza la posición y otros parámetros del carro
    def update(self):
        self.score += self.velocity
        self.time_taken = pygame.time.get_ticks() - self.start_time  # Actualiza el tiempo tomado
        if self.acceleration != 0:
            self.velocity += self.acceleration
            if self.velocity > maxSpeed:
                self.velocity = maxSpeed
            elif self.velocity < 0:
                self.velocity = 0
        else:
            self.velocity *= 0.92

        self.x, self.y = move((self.x, self.y), self.angle, self.velocity)
        self.center = self.x, self.y

        self.d = self.x - (self.width / 2), self.y - (self.height / 2)
        self.c = self.x + self.width - (self.width / 2), self.y - (self.height / 2)
        self.b = self.x + self.width - (self.width / 2), self.y + self.height - (self.height / 2)
        self.a = self.x - (self.width / 2), self.y + self.height - (self.height / 2)

        self.a = rotation((self.x, self.y), self.a, math.radians(self.angle))
        self.b = rotation((self.x, self.y), self.b, math.radians(self.angle))
        self.c = rotation((self.x, self.y), self.c, math.radians(self.angle))
        self.d = rotation((self.x, self.y), self.d, math.radians(self.angle))

        # Actualización de los puntos de colisión
        self.c1 = move((self.x, self.y), self.angle, 10)
        while bg4.get_at((int(self.c1[0]), int(self.c1[1]))).a != 0:
            self.c1 = move((self.c1[0], self.c1[1]), self.angle, 10)
        while bg4.get_at((int(self.c1[0]), int(self.c1[1]))).a == 0:
            self.c1 = move((self.c1[0], self.c1[1]), self.angle, -1)

        self.c2 = move((self.x, self.y), self.angle + 45, 10)
        while bg4.get_at((int(self.c2[0]), int(self.c2[1]))).a != 0:
            self.c2 = move((self.c2[0], self.c2[1]), self.angle + 45, 10)
        while bg4.get_at((int(self.c2[0]), int(self.c2[1]))).a == 0:
            self.c2 = move((self.c2[0], self.c2[1]), self.angle + 45, -1)

        self.c3 = move((self.x, self.y), self.angle - 45, 10)
        while bg4.get_at((int(self.c3[0]), int(self.c3[1]))).a != 0:
            self.c3 = move((self.c3[0], self.c3[1]), self.angle - 45, 10)
        while bg4.get_at((int(self.c3[0]), int(self.c3[1]))).a == 0:
            self.c3 = move((self.c3[0], self.c3[1]), self.angle - 45, -1)

        self.c4 = move((self.x, self.y), self.angle + 90, 10)
        while bg4.get_at((int(self.c4[0]), int(self.c4[1]))).a != 0:
            self.c4 = move((self.c4[0], self.c4[1]), self.angle + 90, 10)
        while bg4.get_at((int(self.c4[0]), int(self.c4[1]))).a == 0:
            self.c4 = move((self.c4[0], self.c4[1]), self.angle + 90, -1)

        self.c5 = move((self.x, self.y), self.angle - 90, 10)
        while bg4.get_at((int(self.c5[0]), int(self.c5[1]))).a != 0:
            self.c5 = move((self.c5[0], self.c5[1]), self.angle - 90, 10)
        while bg4.get_at((int(self.c5[0]), int(self.c5[1]))).a == 0:
            self.c5 = move((self.c5[0], self.c5[1]), self.angle - 90, -1)

        # Calcular las distancias de los puntos de colisión al centro del carro
        self.d1 = int(calculateDistance(self.center[0], self.center[1], self.c1[0], self.c1[1]))
        self.d2 = int(calculateDistance(self.center[0], self.center[1], self.c2[0], self.c2[1]))
        self.d3 = int(calculateDistance(self.center[0], self.center[1], self.c3[0], self.c3[1]))
        self.d4 = int(calculateDistance(self.center[0], self.center[1], self.c4[0], self.c4[1]))
        self.d5 = int(calculateDistance(self.center[0], self.center[1], self.c5[0], self.c5[1]))

    # Dibuja el carro en la pantalla
    def draw(self, display):
        rotated_image = pygame.transform.rotate(self.car_image, -self.angle - 180)
        rect_rotated_image = rotated_image.get_rect()
        rect_rotated_image.center = self.x, self.y
        gameDisplay.blit(rotated_image, rect_rotated_image)

        center = self.x, self.y
        if self.showLinesBool:
            pygame.draw.line(gameDisplay, Color_line, (self.x, self.y), self.c1, 2)
            pygame.draw.line(gameDisplay, Color_line, (self.x, self.y), self.c2, 2)
            pygame.draw.line(gameDisplay, Color_line, (self.x, self.y), self.c3, 2)
            pygame.draw.line(gameDisplay, Color_line, (self.x, self.y), self.c4, 2)
            pygame.draw.line(gameDisplay, Color_line, (self.x, self.y), self.c5, 2)

    # Alterna la visualización de las líneas de distancia
    def showLines(self):
        self.showLinesBool = not self.showLinesBool

    # Alimenta la red neuronal con las entradas y obtiene la salida
    def feedforward(self):
        self.inp = np.array([[self.d1], [self.d2], [self.d3], [self.d4], [self.d5], [self.velocity]])
        for b, w in zip(self.biases, self.weights):
            self.inp = sigmoid(np.dot(w, self.inp) + b)
        self.outp = self.inp
        return self.outp

    # Verifica si el carro colisiona
    def collision(self):
        if (bg4.get_at((int(self.a[0]), int(self.a[1]))).a == 0) or (
                bg4.get_at((int(self.b[0]), int(self.b[1]))).a == 0) or (
                bg4.get_at((int(self.c[0]), int(self.c[1]))).a == 0) or (
                bg4.get_at((int(self.d[0]), int(self.d[1]))).a == 0):
            return True
        else:
            return False

    # Resetea la posición del carro
    def resetPosition(self):
        self.x = 120
        self.y = 480
        self.angle = 180
        return

    # Realiza una acción según la salida de la red neuronal
    def takeAction(self):
        if self.outp.item(0) > 0.5:  # Acelerar
            self.set_accel(0.2)
        else:
            self.set_accel(0)
        if self.outp.item(1) > 0.5:  # Frenar
            self.set_accel(-0.2)
        if self.outp.item(2) > 0.5:  # Girar a la derecha
            self.rotate(-5)
        if self.outp.item(3) > 0.5:  # Girar a la izquierda
            self.rotate(5)
        return


# Inicialización de la lista de carros controlados por redes neuronales
nnCars = []
num_of_nnCars = 200  # Número de carros controlados por redes neuronales
alive = num_of_nnCars  # Número de carros que no han colisionado
collidedCars = []  # Lista de carros que han colisionado

# Textos mostrados en la ventana de pygame
infoX = 1365
infoY = 600
font = pygame.font.Font('freesansbold.ttf', 18)
text1 = font.render('0..9 - Cambiar Mutación', True, white)
text2 = font.render('LMB - Seleccionar/Deseleccionar', True, white)
text3 = font.render('RMB - Eliminar', True, white)
text4 = font.render('L - Mostrar/Ocultar Líneas', True, white)
text5 = font.render('R - Reiniciar', True, white)
text6 = font.render('B - Cruzar', True, white)
text7 = font.render('C - Limpiar', True, white)
text8 = font.render('N - Siguiente Pista', True, white)
text9 = font.render('A - Alternar Jugador', True, white)
text10 = font.render('D - Alternar Información', True, white)
text11 = font.render('M - Cruzar y Siguiente Pista', True, white)
text1Rect = text1.get_rect().move(infoX, infoY)
text2Rect = text2.get_rect().move(infoX, infoY + text1Rect.height)
text3Rect = text3.get_rect().move(infoX, infoY + 2 * text1Rect.height)
text4Rect = text4.get_rect().move(infoX, infoY + 3 * text1Rect.height)
text5Rect = text5.get_rect().move(infoX, infoY + 4 * text1Rect.height)
text6Rect = text6.get_rect().move(infoX, infoY + 5 * text1Rect.height)
text7Rect = text7.get_rect().move(infoX, infoY + 6 * text1Rect.height)
text8Rect = text8.get_rect().move(infoX, infoY + 7 * text1Rect.height)
text9Rect = text9.get_rect().move(infoX, infoY + 8 * text1Rect.height)
text10Rect = text10.get_rect().move(infoX, infoY + 9 * text1Rect.height)
text11Rect = text11.get_rect().move(infoX, infoY + 10 * text1Rect.height)


# Función para mostrar los textos en pantalla
def displayTexts():
    infoTextX = 20
    infoTextY = 600
    infoText1 = font.render('Gen ' + str(generation), True, white)
    infoText2 = font.render('carros: ' + str(num_of_nnCars), True, white)
    infoText3 = font.render('Vivos: ' + str(alive), True, white)
    infoText4 = font.render('Seleccionados: ' + str(selected), True, white)
    if lines:
        infoText5 = font.render('Líneas ON', True, white)
    else:
        infoText5 = font.render('Líneas OFF', True, white)
    if player:
        infoText6 = font.render('Jugador ON', True, white)
    else:
        infoText6 = font.render('Jugador OFF', True, white)
    infoText9 = font.render('FPS: 30', True, white)
    infoText1Rect = infoText1.get_rect().move(infoTextX, infoTextY)
    infoText2Rect = infoText2.get_rect().move(infoTextX, infoTextY + infoText1Rect.height)
    infoText3Rect = infoText3.get_rect().move(infoTextX, infoTextY + 2 * infoText1Rect.height)
    infoText4Rect = infoText4.get_rect().move(infoTextX, infoTextY + 3 * infoText1Rect.height)
    infoText5Rect = infoText5.get_rect().move(infoTextX, infoTextY + 4 * infoText1Rect.height)
    infoText6Rect = infoText6.get_rect().move(infoTextX, infoTextY + 5 * infoText1Rect.height)
    infoText9Rect = infoText9.get_rect().move(infoTextX, infoTextY + 6 * infoText1Rect.height)

    gameDisplay.blit(text1, text1Rect)
    gameDisplay.blit(text2, text2Rect)
    gameDisplay.blit(text3, text3Rect)
    gameDisplay.blit(text4, text4Rect)
    gameDisplay.blit(text5, text5Rect)
    gameDisplay.blit(text6, text6Rect)
    gameDisplay.blit(text7, text7Rect)
    gameDisplay.blit(text8, text8Rect)
    gameDisplay.blit(text9, text9Rect)
    gameDisplay.blit(text10, text10Rect)
    gameDisplay.blit(text11, text11Rect)

    gameDisplay.blit(infoText1, infoText1Rect)
    gameDisplay.blit(infoText2, infoText2Rect)
    gameDisplay.blit(infoText3, infoText3Rect)
    gameDisplay.blit(infoText4, infoText4Rect)
    gameDisplay.blit(infoText5, infoText5Rect)
    gameDisplay.blit(infoText6, infoText6Rect)
    gameDisplay.blit(infoText9, infoText9Rect)
    return


# Creación de la pantalla de juego
gameDisplay = pygame.display.set_mode(size)
clock = pygame.time.Clock()

# Inicialización de la red neuronal del carro
inputLayer = 6
hiddenLayer = 6
outputLayer = 4
car = Car([inputLayer, hiddenLayer, outputLayer])
auxcar = Car([inputLayer, hiddenLayer, outputLayer])

# Creación de los carros controlados por redes neuronales
for i in range(num_of_nnCars):
    nnCars.append(Car([inputLayer, hiddenLayer, outputLayer]))


# Función para redibujar la ventana del juego en cada frame
def redrawGameWindow():
    global alive
    global frames
    global img

    frames += 1

    gameD = gameDisplay.blit(bg, (0, 0))

    # Actualizar y dibujar carros controlados por redes neuronales
    for nncar in nnCars:
        if not nncar.collided:
            nncar.update()

        if nncar.collision():  # Verificar si algún carro ha colisionado
            nncar.collided = True  # Si colisionó, cambiar el estado a colisionado
            if not nncar.yaReste:
                alive -= 1
                nncar.yaReste = True
        else:  # Si no colisionó, alimentar la red neuronal y realizar una acción
            nncar.feedforward()
            nncar.takeAction()
        nncar.draw(gameDisplay)

    # Actualizar y dibujar el carro del jugador
    if player:
        car.update()
        if car.collision():
            car.resetPosition()
            car.update()
        car.draw(gameDisplay)

    # Mostrar información en pantalla
    if display_info:
        displayTexts()

    pygame.display.update()  # Actualiza la pantalla

# Definir puntos de control en el circuito (ejemplo: lista de coordenadas)
checkpoints = [(120, 480), (200, 400), (300, 300), (400, 200), (500, 100)]  # Ejemplo de puntos de control

# Función para verificar si el carro ha pasado un checkpoint
def passed_checkpoint(car, checkpoints):
    # Distancia mínima para considerar que ha pasado un checkpoint
    min_dist_to_checkpoint = 50  # Ajusta este valor según el tamaño del circuito
    for checkpoint in checkpoints:
        dist_to_checkpoint = calculateDistance(car.x, car.y, checkpoint[0], checkpoint[1])
        if dist_to_checkpoint < min_dist_to_checkpoint:
            car.checkpoint_index += 1  # Avanza al siguiente checkpoint
            return True  # Si pasa el checkpoint
    return False  # Si no lo ha pasado

# Función para calcular la distancia acumulada si el carro avanza hacia adelante
def calculate_progressive_distance(car):
    # Guardamos la última posición conocida del carro
    if not hasattr(car, 'last_position'):
        car.last_position = (car.x, car.y)  # Inicializa la posición en el primer frame
    
    # Calculamos la distancia movida desde la última posición
    distance_moved = calculateDistance(car.x, car.y, car.last_position[0], car.last_position[1])
    
    # Si el carro ha avanzado hacia adelante, actualizamos la última posición
    if distance_moved > 0:  # Solo contar si la distancia es positiva (avance hacia adelante)
        car.last_position = (car.x, car.y)  # Actualizamos la última posición
        car.score += distance_moved  # Sumamos la distancia al score (distancia progresiva)
    
    return car.score  # Retornamos la distancia acumulada (score)


# Función para calcular la fitness basada en el avance hacia los checkpoints
def calculate_fitness_with_checkpoints(car):
    if not hasattr(car, 'checkpoint_index'):
        car.checkpoint_index = 0  # Inicializar el índice del checkpoint actual
    
    # Verificar si el carro ha pasado un checkpoint
    if passed_checkpoint(car, checkpoints):
        car.checkpoint_index += 1  # Si pasa un checkpoint, avanza al siguiente
    
    # Calcular la fitness basada en el número de checkpoints alcanzados y la distancia progresiva
    return car.checkpoint_index * 100 + calculate_progressive_distance(car)  # Mayor peso a los checkpoints

# Penalización para carros que no avanzan
def penalize_no_progress(car, no_progress_frames=100):
    if not hasattr(car, 'last_progress_frame'):
        car.last_progress_frame = 0

    if calculate_progressive_distance(car) == 0:
        car.last_progress_frame += 1
    else:
        car.last_progress_frame = 0  # Reinicia si el carro avanza

    if car.last_progress_frame > no_progress_frames:
        return -100  # Penalización por no avanzar
    return 0

# Función para seleccionar los mejores carros considerando checkpoints y penalizaciones
def select_best_cars(nnCars):
    sorted_cars = sorted(nnCars, key=lambda car: calculate_fitness_with_checkpoints(car) + penalize_no_progress(car), reverse=True)
    
    # Seleccionar los dos mejores carros
    best_cars = sorted_cars[:2]
    
    return best_cars



# Inicialización de parámetros del juego
generation = 1
mutationRate = 90
FPS = 30  # Tasa base de frames por segundo (normal)
game_speed = 1  # Multiplicador de velocidad del juego (1 = normal, 2 = x2, etc.)

# Bucle principal del juego
while True:
    for event in pygame.event.get():  # Verificar eventos
        if event.type == pygame.QUIT:
            pygame.quit()  # Salir del juego
            quit()

        if event.type == pygame.KEYDOWN:  # Si el usuario presiona una tecla
            if event.key == ord("l"):  # Alternar visualización de líneas
                car.showLines()
                lines = not lines
            if event.key == ord("c"):  # Limpiar carros colisionados
                for nncar in nnCars:
                    if nncar.collided:
                        nnCars.remove(nncar)
                        if not nncar.yaReste:
                            alive -= 1
            if event.key == ord("a"):  # Alternar visualización del carro del jugador
                player = not player
            if event.key == ord("d"):  # Alternar visualización de información
                display_info = not display_info
            if event.key == ord("n"):  # Generar nueva pista aleatoria
                number_track = 2
                for nncar in nnCars:
                    nncar.velocity = 0
                    nncar.acceleration = 0
                    nncar.x = 140
                    nncar.y = 610
                    nncar.angle = 180
                    nncar.collided = False
                generateRandomMap(gameDisplay)
                bg = pygame.image.load('randomGeneratedTrackFront.png')
                bg4 = pygame.image.load('randomGeneratedTrackBack.png')

            # Ajuste de velocidad del juego
            if event.key == ord("x"):  # Tecla 'x' para aumentar la velocidad del juego (modo x2)
                game_speed = 2  # Aumenta la velocidad del juego
            if event.key == ord("z"):  # Tecla 'z' para volver a la velocidad normal
                game_speed = 1  # Vuelve a la velocidad normal

            if event.key == ord("b"):  # Cruzar nuevos carros a partir de los seleccionados automáticamente
                # Seleccionar los dos mejores carros basados en la distancia progresiva y los checkpoints
                best_cars = select_best_cars(nnCars)  # Utiliza la nueva fitness function con checkpoints y penalizaciones

                # Cruzar los dos mejores carros
                for nncar in nnCars:
                    nncar.score = 0

                alive = num_of_nnCars
                generation += 1
                nnCars.clear()

                for i in range(num_of_nnCars):
                    nnCars.append(Car([inputLayer, hiddenLayer, outputLayer]))

                # Aplicar cruce uniforme entre los dos mejores carros
                for i in range(0, num_of_nnCars - 2, 2):
                    uniformCrossOverWeights(best_cars[0], best_cars[1], nnCars[i], nnCars[i + 1])
                    uniformCrossOverBiases(best_cars[0], best_cars[1], nnCars[i], nnCars[i + 1])

                # Asignar los mejores carros a la nueva generación
                nnCars[num_of_nnCars - 2] = best_cars[0]
                nnCars[num_of_nnCars - 1] = best_cars[1]

                nnCars[num_of_nnCars - 2].car_image = green_small_car
                nnCars[num_of_nnCars - 1].car_image = green_small_car

                nnCars[num_of_nnCars - 2].resetPosition()
                nnCars[num_of_nnCars - 1].resetPosition()

                nnCars[num_of_nnCars - 2].collided = False
                nnCars[num_of_nnCars - 1].collided = False

                for i in range(num_of_nnCars - 2):
                    for j in range(mutationRate):
                        mutateOneWeightGene(nnCars[i], auxcar)
                        mutateOneWeightGene(auxcar, nnCars[i])
                        mutateOneBiasesGene(nnCars[i], auxcar)
                        mutateOneBiasesGene(auxcar, nnCars[i])

                # Reposicionar los carros en el inicio para la siguiente generación
                for nncar in nnCars:
                    nncar.x = 140
                    nncar.y = 610

                # Limpiar la lista de carros seleccionados
                selectedCars.clear()

            if event.key == ord("m"):  # Cruzar nuevos carros y generar nueva pista aleatoria
                if len(selectedCars) == 2:
                    for nncar in nnCars:
                        nncar.score = 0

                    alive = num_of_nnCars
                    generation += 1
                    selected = 0
                    nnCars.clear()

                    for i in range(num_of_nnCars):
                        nnCars.append(Car([inputLayer, hiddenLayer, outputLayer]))

                    for i in range(0, num_of_nnCars - 2, 2):
                        uniformCrossOverWeights(selectedCars[0], selectedCars[1], nnCars[i], nnCars[i + 1])
                        uniformCrossOverBiases(selectedCars[0], selectedCars[1], nnCars[i], nnCars[i + 1])

                    nnCars[num_of_nnCars - 2] = selectedCars[0]
                    nnCars[num_of_nnCars - 1] = selectedCars[1]

                    nnCars[num_of_nnCars - 2].car_image = green_small_car
                    nnCars[num_of_nnCars - 1].car_image = green_small_car

                    nnCars[num_of_nnCars - 2].resetPosition()
                    nnCars[num_of_nnCars - 1].resetPosition()

                    nnCars[num_of_nnCars - 2].collided = False
                    nnCars[num_of_nnCars - 1].collided = False

                    for i in range(num_of_nnCars - 2):
                        for j in range(mutationRate):
                            mutateOneWeightGene(nnCars[i], auxcar)
                            mutateOneWeightGene(auxcar, nnCars[i])
                            mutateOneBiasesGene(nnCars[i], auxcar)
                            mutateOneBiasesGene(auxcar, nnCars[i])

                    for nncar in nnCars:
                        nncar.x = 140
                        nncar.y = 610

                    selectedCars.clear()

                    number_track = 2
                    for nncar in nnCars:
                        nncar.velocity = 0
                        nncar.acceleration = 0
                        nncar.x = 140
                        nncar.y = 610
                        nncar.angle = 180
                        nncar.collided = False
                    generateRandomMap(gameDisplay)
                    bg = pygame.image.load('randomGeneratedTrackFront.png')
                    bg4 = pygame.image.load('randomGeneratedTrackBack.png')
            if event.key == ord("r"):  # Reiniciar la simulación
                generation = 1
                alive = num_of_nnCars
                nnCars.clear()
                selectedCars.clear()
                for i in range(num_of_nnCars):
                    nnCars.append(Car([inputLayer, hiddenLayer, outputLayer]))
                for nncar in nnCars:
                    if number_track == 1:
                        nncar.x = 120
                        nncar.y = 480
                    elif number_track == 2:
                        nncar.x = 100
                        nncar.y = 300
            if event.key == ord("0"):  # Cambiar tasa de mutación
                mutationRate = 0
            if event.key == ord("1"):
                mutationRate = 10
            if event.key == ord("2"):
                mutationRate = 20
            if event.key == ord("3"):
                mutationRate = 30
            if event.key == ord("4"):
                mutationRate = 40
            if event.key == ord("5"):
                mutationRate = 50
            if event.key == ord("6"):
                mutationRate = 60
            if event.key == ord("7"):
                mutationRate = 70
            if event.key == ord("8"):
                mutationRate = 80
            if event.key == ord("9"):
                mutationRate = 90

        if event.type == pygame.MOUSEBUTTONDOWN:
            # Detectar clics del ratón
            mouses = pygame.mouse.get_pressed(num_buttons=3)
            if mouses[0]:  # Clic izquierdo
                pos = pygame.mouse.get_pos()
                point = Point(pos[0], pos[1])
                for nncar in nnCars:
                    polygon = Polygon([nncar.a, nncar.b, nncar.c, nncar.d])
                    if polygon.contains(point):
                        if nncar in selectedCars:
                            selectedCars.remove(nncar)
                            selected -= 1
                            if nncar.car_image == white_big_car:
                                nncar.car_image = white_small_car
                            if nncar.car_image == green_big_car:
                                nncar.car_image = green_small_car
                            if nncar.collided:
                                nncar.velocity = 0
                                nncar.acceleration = 0
                            nncar.update()
                        else:
                            if len(selectedCars) < 2:
                                selectedCars.append(nncar)
                                selected += 1
                                if nncar.car_image == white_small_car:
                                    nncar.car_image = white_big_car
                                if nncar.car_image == green_small_car:
                                    nncar.car_image = green_big_car
                                if nncar.collided:
                                    nncar.velocity = 0
                                    nncar.acceleration = 0
                                nncar.update()
                        break

            if mouses[2]:  # Clic derecho
                pos = pygame.mouse.get_pos()
                point = Point(pos[0], pos[1])
                for nncar in nnCars:
                    polygon = Polygon([nncar.a, nncar.b, nncar.c, nncar.d])
                    if polygon.contains(point):
                        if nncar not in selectedCars:
                            nnCars.remove(nncar)
                            alive -= 1
                        break

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        car.rotate(-5)
    if keys[pygame.K_RIGHT]:
        car.rotate(5)
    if keys[pygame.K_UP]:
        car.set_accel(0.2)
    else:
        car.set_accel(0)
    if keys[pygame.K_DOWN]:
        car.set_accel(-0.2)

    redrawGameWindow()

    # Ajustar la velocidad del juego utilizando la tasa de FPS multiplicada por game_speed
    clock.tick(FPS * game_speed)  # Si game_speed es 2, corre a x2 la velocidad normal
