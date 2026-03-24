# pap_air

## Informe técnico del repositorio

Este repositorio no implementa una plataforma completa de "gestión de aeropuertos" en el sentido habitual del término. El código fuente que contiene hoy es, en realidad, una práctica académica o experimental en **C++ con CUDA** orientada al **procesamiento de un dataset de vuelos** usando CPU para la carga de datos y GPU para una reducción simple.

El programa toma un fichero CSV con información de vuelos, extrae tres columnas relacionadas con retrasos, deja al usuario elegir cuál analizar y después calcula en la GPU el **máximo** o el **mínimo** de esa columna mediante operaciones atómicas.

En otras palabras, el objetivo técnico del proyecto es este:

- leer un conjunto de datos de vuelos desde CPU,
- convertir una parte de ese dataset a una estructura simple en memoria,
- copiar los datos relevantes a la GPU,
- ejecutar un kernel CUDA muy básico,
- obtener un resultado agregado final.

## Qué hay realmente en el repositorio

El repositorio está organizado alrededor de un proyecto de Visual Studio llamado `PL1_CUDA`. Su contenido útil se divide en varios grupos:

### 1. Código fuente real

- `PL1_CUDA/src/main.cu`
- `PL1_CUDA/src/csv_reader.h`
- `PL1_CUDA/src/csv_reader.cpp`
- `PL1_CUDA/src/kernels.cuh`
- `PL1_CUDA/src/kernels.cu`

Aquí está toda la lógica funcional del programa.

### 2. Configuración de compilación

- `PL1_CUDA/PL1_CUDA.sln`
- `PL1_CUDA/PL1_CUDA.vcxproj`
- `PL1_CUDA/PL1_CUDA.vcxproj.user`

Estos archivos describen cómo Visual Studio organiza y compila el proyecto, qué configuraciones existen y qué integración CUDA utiliza.

### 3. Metadatos y artefactos del entorno

- `.vs/`
- `PL1_CUDA/.vs/`
- `PL1_CUDA/x64/Debug/`

Estos elementos no contienen la lógica principal del sistema. Son ficheros generados por Visual Studio, por la compilación o por la ejecución previa del proyecto. Su presencia indica que el repositorio ha sido trabajado desde un entorno Windows con Visual Studio y CUDA, y que incluso se han subido binarios y salidas de build al control de versiones.

### 4. Documentación y licencia

- `README.md`
- `LICENSE`
- `.gitignore`

## Objetivo funcional del proyecto

El programa busca responder una operación muy concreta sobre un dataset de vuelos:

1. cargar registros desde un CSV,
2. quedarse solo con tres variables de retraso,
3. pedir al usuario qué variable quiere estudiar,
4. pedir si desea calcular un máximo o un mínimo,
5. lanzar una reducción simple en GPU,
6. mostrar el resultado.

Por tanto, el objetivo no es gestionar reservas, terminales, puertas de embarque o aeropuertos como entidades completas. El objetivo real es **experimentar con paralelismo GPU sobre datos tabulares de aviación**.

## Visión general de cómo funciona

La ejecución del programa sigue este flujo:

1. `main.cu` inicia el programa y fija la ruta del dataset.
2. `main.cu` llama a `loadCSV(...)`, definida en `csv_reader.cpp`.
3. `loadCSV(...)` abre el archivo, recorre línea a línea el CSV y construye un `std::vector<FlightData>`.
4. De vuelta en `main.cu`, el usuario selecciona qué columna desea analizar:
   - `DEP_DELAY`
   - `ARR_DELAY`
   - `WEATHER_DELAY`
5. El programa transforma el vector de estructuras en un `std::vector<int>` con una sola columna.
6. Se reserva memoria en GPU con `cudaMalloc`.
7. Los datos se copian de CPU a GPU mediante `cudaMemcpy`.
8. Se inicializa un acumulador global en GPU con `INT_MIN` o `INT_MAX`, según se quiera calcular máximo o mínimo.
9. Se lanza el kernel `reductionSimple`.
10. Cada hilo CUDA procesa un elemento y aplica una operación atómica global sobre el resultado.
11. El resultado se copia de vuelta a CPU y se imprime.
12. La memoria de GPU se libera.

Ese flujo ya deja ver la separación principal del proyecto:

- **CPU**: lectura del fichero, parseo del CSV, selección interactiva y preparación de datos.
- **GPU**: cálculo paralelo del máximo o mínimo.

## Análisis detallado de los archivos fuente

## `PL1_CUDA/src/csv_reader.h`

Este fichero define la interfaz mínima del módulo de lectura del CSV.

### Qué contiene

- una estructura `FlightData`,
- la declaración de la función `loadCSV`.

### Qué representa `FlightData`

La estructura contiene tres campos enteros:

- `dep_delay`
- `arr_delay`
- `weather_delay`

Eso significa que, de todo el dataset original, el programa solo conserva tres medidas de retraso. Todo lo demás se descarta.

### Por qué es importante

Este header desacopla la lectura del CSV del resto del programa. `main.cu` no necesita conocer cómo se abre el fichero ni cómo se parsean las columnas; solo necesita pedir un `std::vector<FlightData>`.

Es una separación simple, pero correcta: el contrato del módulo queda reducido a "dame una colección de vuelos con estos tres retrasos".

## `PL1_CUDA/src/csv_reader.cpp`

Este fichero implementa la función `loadCSV(...)`. Es la fase de adquisición y transformación de datos en CPU.

### Objetivo de `loadCSV`

Su trabajo consiste en:

- abrir el archivo CSV,
- leerlo línea a línea,
- extraer ciertas columnas,
- convertir sus valores a enteros,
- almacenar el resultado en memoria como una colección de `FlightData`.

### Flujo interno

#### 1. Inicialización del vector de salida

La función crea:

```cpp
std::vector<FlightData> data;
```

Ese vector será el resultado final que se devolverá a `main.cu`.

#### 2. Apertura del fichero

Se crea un `std::ifstream` con el nombre recibido por parámetro. Si el archivo no se puede abrir, la función:

- imprime un error por `stderr`,
- devuelve el vector vacío.

Este comportamiento es importante porque marca la primera condición de fallo del programa: si el CSV no existe o no está en la ruta esperada, todo el procesamiento termina antes de empezar.

#### 3. Lectura de la cabecera

La primera línea se consume con `std::getline(file, line);`. Esa línea se asume como cabecera y no se procesa como dato.

El programa, además, la imprime por consola. Eso sirve como depuración, no como parte de la lógica analítica.

#### 4. Recorrido línea a línea

Cada iteración del bucle principal lee una línea completa del fichero y la trata como un registro de vuelo.

Se lleva un contador `line_count` que permite mostrar progreso cada 100000 líneas. Eso sugiere que el dataset esperado puede ser grande y que el autor quería visibilidad durante la carga.

#### 5. Parseo de columnas

Cada línea se mete en un `std::stringstream`, y después se trocea por comas con:

```cpp
std::getline(ss, value, ',')
```

Esto implica una decisión técnica importante: el CSV se trata como un formato simple separado por comas, sin un parser especializado.

#### 6. Limpieza básica de comillas

Si el campo empieza o termina con comillas, estas se eliminan manualmente.

Esto intenta cubrir casos básicos como:

```text
"12.0"
```

pero no resuelve casos complejos de CSV real, por ejemplo:

- comas dentro de campos entrecomillados,
- comillas escapadas,
- formatos más complejos del estándar CSV.

#### 7. Conversión numérica

Cada valor se convierte con:

```cpp
std::stof(value)
```

y después se trunca a entero mediante `static_cast<int>`.

Esta decisión tiene varias consecuencias:

- si el dato original es decimal, se pierde precisión,
- si el campo no es numérico, la conversión lanza excepción,
- las excepciones se capturan y el valor se ignora.

#### 8. Selección de columnas

El código usa índices de columna duros:

- columna 10 -> `dep_delay`
- columna 12 -> `arr_delay`
- columna 13 -> `weather_delay`

Esto hace que el programa dependa completamente del orden exacto del dataset esperado. Si cambia el CSV, aunque el nombre de las columnas siga siendo el mismo, el resultado puede quedar mal sin que el programa lo detecte.

#### 9. Política de validez

La variable `valid` pasa a `true` si se ha podido leer al menos uno de los tres campos.

Eso significa que una fila se almacena incluso si no tiene los tres valores válidos. En ese caso, los campos no leídos quedan con el valor inicial `0`.

Este detalle es muy importante porque afecta al significado del resultado:

- una fila parcialmente inválida puede introducir ceros artificiales,
- esos ceros pueden influir en el mínimo o el máximo final,
- el programa no distingue entre "valor real 0" y "dato faltante reemplazado por 0".

### Qué consigue este módulo

Transforma un CSV general en una estructura compacta y manejable para el resto del pipeline.

### Qué no resuelve

No valida el esquema del CSV, no verifica nombres de columnas, no maneja bien CSV complejos y no distingue bien entre valores ausentes y valores cero.

## `PL1_CUDA/src/kernels.cuh`

Este archivo declara el kernel CUDA:

```cpp
__global__ void reductionSimple(int* data, int* result, int n, bool isMax);
```

### Qué papel cumple

Es la interfaz del módulo GPU. Sirve para que `main.cu` conozca la firma del kernel sin necesitar su implementación.

### Qué expresa la firma

- `data`: array de entrada en GPU,
- `result`: puntero a un único entero global donde se acumula el resultado,
- `n`: tamaño del array,
- `isMax`: selector entre máximo y mínimo.

Este diseño deja claro que la reducción no usa memoria compartida, ni reducción por bloques, ni árboles paralelos. Usa un único acumulador global.

## `PL1_CUDA/src/kernels.cu`

Aquí está la implementación del cálculo paralelo.

### Qué hace el kernel `reductionSimple`

Cada hilo CUDA:

1. calcula su índice global,
2. comprueba si está dentro del rango,
3. lee un elemento del array,
4. aplica una operación atómica sobre un resultado global.

### Índice global

El índice se calcula como:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

Ese patrón es el estándar en CUDA para mapear hilos a posiciones lineales de un vector.

### Control de límites

La condición:

```cpp
if (idx >= n) return;
```

evita que hilos sobrantes accedan fuera del array. Esto es necesario porque el número total de hilos suele redondearse hacia arriba para cubrir todo el vector.

### Operación principal

El hilo lee:

```cpp
int value = data[idx];
```

y luego:

- si `isMax` es verdadero, ejecuta `atomicMax(result, value)`,
- si no, ejecuta `atomicMin(result, value)`.

### Qué implica este enfoque

Es una solución correcta a nivel funcional para una práctica simple, pero no es la forma más eficiente de hacer una reducción en GPU.

La razón es que todos los hilos compiten por actualizar la misma posición de memoria global. Eso genera contención y limita el escalado.

Desde el punto de vista didáctico, sin embargo, el enfoque tiene sentido porque:

- es fácil de entender,
- tiene poco código,
- deja clara la idea de paralelismo básico sobre datos.

## `PL1_CUDA/src/main.cu`

Este es el archivo central del programa. Orquesta todo el flujo de ejecución.

## Estructura conceptual de `main`

`main` está dividido de forma bastante clara en fases:

1. presentación del programa,
2. carga del CSV,
3. selección de columna,
4. selección de tipo de reducción,
5. reserva y copia de memoria GPU,
6. configuración de ejecución,
7. recogida del resultado,
8. liberación de recursos.

### 1. Inicio del programa

El programa imprime un mensaje de cabecera:

```cpp
=== PL1 CUDA - Fase 03 (Variante 3.1 Simple) ===
```

Esto da contexto académico: parece corresponder a una práctica o fase concreta de una asignatura o laboratorio.

### 2. Ruta del dataset

La ruta se fija como:

```cpp
std::string path = "data/Airline_dataset.csv";
```

Eso revela una dependencia crítica: el programa espera un fichero concreto en una ruta relativa concreta.

En el estado actual del repositorio, ese archivo no aparece dentro del árbol observado. Por tanto:

- el proyecto está incompleto como paquete autocontenido,
- o el dataset se gestiona fuera del repositorio,
- o el fichero fue omitido del control de versiones.

### 3. Llamada al lector CSV

`main` llama a:

```cpp
std::vector<FlightData> data = loadCSV(path);
```

Aquí se produce la transición entre el módulo principal y el módulo de lectura.

Después se comprueba:

- cuántos datos se han cargado,
- si el vector está vacío.

Si no hay datos, el programa termina con error.

### 4. Selección de columna por parte del usuario

El usuario puede elegir entre:

- `DEP_DELAY`
- `ARR_DELAY`
- `WEATHER_DELAY`

Después, el programa recorre todos los `FlightData` y extrae la columna elegida a un nuevo vector `h_data`.

Esto es relevante porque la GPU no recibe estructuras complejas; recibe un vector lineal de enteros. Es una transformación de datos desde un formato orientado a registros a un formato orientado a columna.

### 5. Selección del tipo de reducción

El usuario elige:

- `1. Max`
- `2. Min`

Luego se genera el booleano:

```cpp
bool isMax = (type == 1);
```

Aquí hay un detalle funcional importante: cualquier opción distinta de `1` implica `false`, por lo que el programa la tratará como mínimo. No hay una validación estricta equivalente a la de la selección de columna.

### 6. Preparación de memoria GPU

Se definen dos punteros:

- `d_data`
- `d_result`

Después:

- se reserva memoria para el array de entrada,
- se reserva memoria para el resultado global,
- se copian los datos del host al device,
- se inicializa el acumulador con un valor centinela.

La inicialización:

- `INT_MIN` si se busca máximo,
- `INT_MAX` si se busca mínimo,

es la decisión correcta para que la primera comparación atómica funcione.

### 7. Configuración del lanzamiento

El programa usa:

```cpp
int threads = 256;
int blocks = (n + threads - 1) / threads;
```

Esto es el patrón estándar para cubrir un vector lineal con bloques de tamaño fijo.

### 8. Lanzamiento del kernel

La llamada:

```cpp
reductionSimple<<<blocks, threads>>>(d_data, d_result, n, isMax);
```

es el punto donde la lógica preparada en CPU pasa a ejecutarse en GPU.

Después se hace:

```cpp
cudaDeviceSynchronize();
```

para asegurar que el cálculo ha terminado antes de recuperar el resultado.

### 9. Recuperación del valor final

El programa copia un único entero desde GPU a CPU y lo imprime por pantalla. Ese valor es el mínimo o el máximo de la columna elegida.

### 10. Limpieza

Se liberan ambas zonas de memoria con `cudaFree`.

Con esto termina la vida del programa.

## Cómo interactúan entre sí los módulos

La interacción entre archivos es sencilla y lineal:

### `main.cu` depende de `csv_reader.h`

Porque necesita:

- la definición de `FlightData`,
- la declaración de `loadCSV`.

### `main.cu` depende de `kernels.cuh`

Porque necesita conocer la firma del kernel CUDA que va a lanzar.

### `csv_reader.cpp` implementa el contrato definido en `csv_reader.h`

Este módulo se encarga únicamente de la lectura y transformación del dataset.

### `kernels.cu` implementa el contrato definido en `kernels.cuh`

Este módulo encapsula la parte GPU del programa.

### Flujo de datos completo

El flujo de datos puede resumirse así:

```text
CSV en disco
-> lectura en CPU
-> vector<FlightData>
-> vector<int> de una columna concreta
-> memoria GPU
-> kernel de reducción
-> entero resultado
-> salida por consola
```

Esa es la interacción esencial del repositorio.

## Análisis de la configuración del proyecto

## `PL1_CUDA/PL1_CUDA.sln`

La solución de Visual Studio define un único proyecto:

- `PL1_CUDA`

También define dos configuraciones:

- `Debug|x64`
- `Release|x64`

Esto confirma que el proyecto está pensado para compilación de 64 bits en Windows.

## `PL1_CUDA/PL1_CUDA.vcxproj`

Este archivo es muy importante porque revela el entorno de compilación esperado.

### Lo que se puede deducir de él

- usa `PlatformToolset v142`,
- está pensado para Visual Studio 2019,
- integra `CUDA 11.8`,
- compila como aplicación de consola,
- enlaza con `cudart_static.lib`,
- incluye dos archivos `.cu` y un `.cpp`,
- declara un recurso de datos esperado: `data/Airline_dataset.csv`.

### Qué implica esto

El proyecto no está pensado como biblioteca, ni como servicio, ni como interfaz gráfica. Está diseñado como un ejecutable de consola para pruebas o prácticas.

Además, al utilizar `cudart_static.lib`, el runtime CUDA se enlaza de forma estática en la configuración definida por el proyecto.

## `PL1_CUDA/PL1_CUDA.vcxproj.user`

Solo activa `ShowAllFiles`. No contiene lógica funcional. Es una preferencia local del entorno de desarrollo.

## `.gitignore`

El `.gitignore` solo ignora ciertos artefactos intermedios de CUDA:

- `*.i`
- `*.ii`
- `*.gpu`
- `*.ptx`
- `*.cubin`
- `*.fatbin`

### Qué significa esto

No se están ignorando muchos archivos que normalmente no deberían versionarse, como:

- `.vs/`
- `x64/Debug/`
- bases de datos del IDE,
- logs,
- ejecutables,
- objetos compilados.

Esto explica por qué el repositorio contiene numerosos ficheros generados. No forman parte de la lógica del programa, pero sí forman parte del estado actual del repositorio.

## Sobre los artefactos generados que aparecen versionados

La carpeta `PL1_CUDA/x64/Debug/` contiene:

- ejecutable,
- librerías generadas,
- objetos,
- cachés,
- dependencias,
- logs.

Esto no añade comportamiento nuevo a la aplicación. Solo refleja una compilación previa.

De igual forma, las carpetas `.vs/` almacenan información local de Visual Studio, como preferencias de espacio de trabajo o bases de datos de exploración.

Desde un punto de vista de ingeniería del software, estos archivos:

- no deberían ser la base del análisis funcional,
- no son la fuente de verdad del sistema,
- complican la limpieza del repositorio,
- y dificultan distinguir código real de salidas de entorno.

## Qué hace el programa exactamente

En términos operativos, el programa hace lo siguiente:

1. intenta abrir un dataset de vuelos en formato CSV;
2. toma únicamente tres columnas de retrasos;
3. convierte cada fila válida a una estructura `FlightData`;
4. deja al usuario elegir una de esas tres medidas;
5. crea un vector lineal con esa métrica;
6. envía el vector a la GPU;
7. calcula un mínimo o un máximo usando un kernel simple;
8. devuelve e imprime el resultado final.

## Cómo lo hace exactamente

Lo hace combinando dos estrategias:

### En CPU

- gestiona entrada/salida,
- abre el fichero,
- recorre texto línea a línea,
- parsea columnas,
- transforma los datos,
- recoge decisiones del usuario,
- prepara los buffers de entrada.

### En GPU

- lanza muchos hilos,
- asigna un elemento del vector a cada hilo,
- usa operaciones atómicas sobre una variable global,
- obtiene una reducción final sencilla.

El método es simple y didáctico, no especialmente optimizado.

## Con qué objetivo está construido

El objetivo visible del repositorio parece ser académico y pedagógico:

- practicar integración de **C++ + CUDA**,
- familiarizarse con la gestión de memoria entre host y device,
- aplicar un kernel elemental sobre datos reales o semirreales,
- trabajar con operaciones de reducción sobre un dataset de aviación.

No parece un producto final ni una aplicación de producción. Le faltan varios elementos que sí serían esperables en software maduro:

- validación robusta del input,
- manejo estructurado de errores CUDA,
- parser CSV fiable,
- pruebas automatizadas,
- documentación de ejecución completa,
- separación más clara entre datos faltantes y datos válidos,
- optimizaciones reales de reducción en GPU.

## Fortalezas del diseño actual

- La arquitectura es muy fácil de entender.
- La separación entre lectura del CSV y cálculo en GPU es clara.
- El flujo de datos está bien delimitado.
- El kernel es corto y didáctico.
- La estructura `FlightData` simplifica el manejo del dataset.
- El uso de `INT_MIN` y `INT_MAX` para inicializar el acumulador es correcto.

## Debilidades y limitaciones técnicas

### 1. Dependencia de un dataset no presente en el árbol observado

El código espera `data/Airline_dataset.csv`, pero ese fichero no aparece en el repositorio inspeccionado. Sin él, la ejecución real queda bloqueada.

### 2. Parseo CSV frágil

Usar `stringstream` con separación por comas no cubre correctamente todo el estándar CSV.

### 3. Columnas fijadas por posición

La lógica depende de los índices 10, 12 y 13. Si el orden del dataset cambia, el programa puede producir resultados incorrectos sin detectarlo.

### 4. Pérdida de información por truncado

`std::stof` seguido de `static_cast<int>` elimina la parte decimal.

### 5. Tratamiento ambiguo de datos faltantes

Las filas parcialmente válidas pueden terminar con valores `0` en algunos campos, contaminando el análisis.

### 6. Validación incompleta de la interacción del usuario

La columna sí se valida explícitamente, pero el tipo de reducción no queda validado con la misma rigorosidad.

### 7. Falta de comprobación de errores CUDA

No se comprueban retornos de:

- `cudaMalloc`
- `cudaMemcpy`
- lanzamiento del kernel
- `cudaDeviceSynchronize`

Esto dificulta depuración y robustez.

### 8. Reducción GPU poco escalable

El uso de un único acumulador global con atómicas simplifica el código, pero introduce contención.

### 9. Repositorio poco limpio

Hay artefactos de compilación y metadatos del IDE versionados, lo que mezcla implementación con salidas de entorno.

## Lectura global del comportamiento del programa

Si se observa el repositorio como un sistema completo, su comportamiento se puede resumir así:

- **entrada**: un CSV de vuelos;
- **transformación en CPU**: reducción del dataset a tres columnas de retrasos;
- **selección por usuario**: elección de métrica y operación;
- **cálculo en GPU**: mínimo o máximo usando atómicas;
- **salida**: un único entero mostrado por consola.

Toda la arquitectura está construida alrededor de esa tubería.

## Qué significa cada parte dentro del objetivo total

- `csv_reader.*` existe para traducir texto bruto a datos manejables.
- `FlightData` existe para representar solo los atributos que interesan al cálculo.
- `main.cu` existe para coordinar la aplicación y conectar CPU con GPU.
- `kernels.*` existe para encapsular la operación paralela.
- `vcxproj` y `sln` existen para hacer compilable el proyecto en Visual Studio con CUDA.
- `.gitignore` y los artefactos generados muestran el estado de mantenimiento del repositorio más que la lógica funcional.

## Conclusión

Este repositorio implementa una práctica de procesamiento de datos de vuelos con CUDA. Su función principal es leer un CSV de aerolíneas, extraer tres métricas de retraso y calcular en GPU el máximo o el mínimo de la columna elegida por el usuario.

Lo hace mediante una arquitectura sencilla:

- lectura y preparación de datos en CPU,
- traslado a memoria GPU,
- kernel de reducción simple basado en operaciones atómicas,
- devolución del resultado final a consola.

Su objetivo es claramente didáctico: demostrar una tubería básica de procesamiento heterogéneo CPU/GPU sobre un dataset tabular.

El repositorio, sin embargo, también refleja varias limitaciones importantes:

- dataset ausente en el árbol inspeccionado,
- parseo CSV poco robusto,
- escasa validación,
- ausencia de control de errores CUDA,
- presencia de artefactos generados versionados.

Aun con esas limitaciones, el código sí deja ver con claridad qué se quería construir: una demostración simple y comprensible de cómo leer datos en C++, transformarlos y aplicar una reducción elemental en CUDA.
