# pap_air

## Estado actual del proyecto

Este repositorio contiene una practica academica de **CUDA C/C++** basada en el
**US Airline Dataset**. En el estado actual del codigo, la parte realmente
implementada es la **Fase 0**:

- lectura del CSV desde disco;
- limpieza basica y validacion de cabecera;
- carga del dataset en memoria del host usando una estructura por columnas;
- calculo de estadisticas de calidad de datos;
- deteccion del hardware CUDA disponible;
- interfaz de consola completa y navegable para las Fases 1-4.

Las **Fases 1, 2, 3 y 4** ya tienen su flujo de menu preparado, pero todavia no
ejecutan su logica CUDA definitiva desde `main.cu`.

---

## Estructura del proyecto

El proyecto principal es `PL1_CUDA` y los archivos mas importantes son:

- `PL1_CUDA/src/main.cu`
  - coordina el flujo general de la aplicacion;
  - mantiene el estado global en memoria;
  - llama al lector CSV;
  - consulta la GPU;
  - gestiona el menu principal y los submenus.
- `PL1_CUDA/src/csv_reader.h`
  - declara las estructuras de datos de la Fase 0;
  - declara las funciones de lectura, limpieza y validacion.
- `PL1_CUDA/src/csv_reader.cpp`
  - implementa la carga del CSV;
  - limpia texto y numeros;
  - genera estadisticas de faltantes y categorias unicas.
- `PL1_CUDA/src/cli_utils.h`
  - declara enums y funciones auxiliares de la interfaz por consola.
- `PL1_CUDA/src/cli_utils.cpp`
  - implementa menus, lectura segura de opciones y pausas entre pantallas.
- `PL1_CUDA/src/kernels.cuh`
  - declara el kernel `reductionSimple`.
- `PL1_CUDA/src/kernels.cu`
  - implementa una reduccion simple en GPU con atomicas.
- `PL1_CUDA/cuda.local.props.example`
  - plantilla de configuracion local para la version de CUDA de cada miembro.

Tambien existe un CSV de ejemplo en:

- `PL1_CUDA/src/data/Airline_dataset.csv`

---

## Configuracion local de Visual Studio y CUDA

El proyecto esta pensado para abrirse en **Visual Studio 2022** y permitir que
cada desarrollador use su propia version de CUDA sin reescribir el
`PL1_CUDA.vcxproj`.

### Preparacion inicial en cada equipo

1. Abrir `PL1_CUDA.sln` con Visual Studio 2022.
2. Intentar compilar directamente si la variable de entorno `CUDA_PATH` ya
   apunta a la instalacion correcta de CUDA.
3. Si Visual Studio no detecta bien la version, copiar
   `PL1_CUDA/cuda.local.props.example` a `PL1_CUDA/cuda.local.props`.
4. Editar `cuda.local.props` y ajustar:
   - `CudaBuildCustomizationVersion`
   - `CudaToolkitCustomDir`
5. Comprobar que ambas propiedades apuntan a la misma instalacion de CUDA.
6. Si Visual Studio intenta reutilizar un binario antiguo, hacer `Clean
   Solution`, borrar `PL1_CUDA/x64/` si hace falta y volver a compilar.

### Ejemplo de configuracion local

```xml
<Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <PropertyGroup>
    <CudaBuildCustomizationVersion>12.6</CudaBuildCustomizationVersion>
    <CudaToolkitCustomDir>$(CUDA_PATH_V12_6)</CudaToolkitCustomDir>
  </PropertyGroup>
</Project>
```

Notas importantes:

- `PL1_CUDA/cuda.local.props` esta ignorado por git y es opcional.
- Si `CUDA_PATH` esta bien definido, el proyecto intenta deducir
  automaticamente la version de CUDA desde esa variable.
- Cada miembro del equipo puede tener valores distintos en ese archivo.
- El proyecto fallara con un mensaje claro solo si no puede deducir CUDA o si
  la version indicada no existe en `BuildCustomizations`.
- El directorio de trabajo del depurador se fija a `$(ProjectDir)` para que las
  rutas relativas del CSV se resuelvan siempre desde `PL1_CUDA/`.

---

## Que hace hoy la aplicacion

Cuando se ejecuta el programa, el flujo actual es este:

1. `main()` imprime un banner inicial.
2. `main()` llama a `queryGpuInfo(...)`.
3. Si hay una GPU CUDA accesible, se muestran sus propiedades basicas.
4. `main()` llama a `promptAndLoadDataset(...)`.
5. `promptAndLoadDataset(...)` usa `promptDatasetPath(...)` para pedir la ruta
   del CSV y proponer `src/data/Airline_dataset.csv` como ruta por defecto si
   existe.
6. Cuando el usuario elige una ruta, `loadDatasetIntoState(...)` llama a
   `loadDataset(...)`.
7. `loadDataset(...)` abre el CSV, valida la cabecera, recorre el fichero fila
   a fila y construye un `DatasetColumns`.
8. Al terminar la carga, `main()` muestra el resumen de Fase 0.
9. El programa entra en un menu persistente con las opciones de las cuatro
   fases, recarga del CSV, estado y salida.
10. Si el usuario entra en una fase, la CLI recoge los parametros y muestra un
    mensaje de "fase pendiente de implementar".

En otras palabras:

- la **CPU** ya gestiona lectura, limpieza, validacion, estado y menus;
- la **GPU** hoy solo se consulta para detectar hardware y sugerir una futura
  configuracion de lanzamiento;
- el kernel de reduccion existe en el repositorio, pero no esta conectado al
  flujo principal actual.

---

## Modelo de datos actual

La estructura central de la Fase 0 es `DatasetColumns`, definida en
`csv_reader.h`.

En lugar de guardar una estructura por fila, el proyecto guarda una **columna
por vector**, lo cual facilita mucho las fases CUDA posteriores.

### Columnas almacenadas

- `depDelay`
- `arrDelay`
- `weatherDelay`
- `depTime`
- `arrTime`
- `tailNum`
- `originSeqId`
- `destSeqId`
- `originAirport`
- `destAirport`

### Reglas de limpieza

- Si un campo numerico falta o no puede convertirse, se guarda como `NAN`.
- Si un texto falta, se guarda como `""`.
- Si un identificador entero falta, se guarda `-1`.
- Todas las columnas deben quedar alineadas por indice.

Esto significa que el indice `i` representa siempre la misma fila logica en
todas las columnas.

---

## Como funciona el lector CSV

La funcion central es:

```cpp
CsvLoadResult loadDataset(const std::string& filename);
```

### Flujo interno de `loadDataset`

1. Intenta abrir el fichero.
2. Lee la cabecera.
3. Divide la cabecera en tokens con `splitCsvLineSimple(...)`.
4. Limpia cada token con `cleanQuotedToken(...)`.
5. Valida que las columnas esperadas esten donde deben con `validateHeader(...)`.
6. Recorre el resto del fichero linea a linea.
7. Para cada fila:
   - divide la linea en tokens;
   - descarta la fila si tiene menos columnas de las esperadas;
   - limpia campos de texto;
   - convierte numericos con `parseFloatOrNan(...)`;
   - convierte IDs enteros con `parseIntFromFloatToken(...)`;
   - actualiza contadores de faltantes;
   - almacena la fila en `DatasetColumns`.
8. Al final:
   - calcula el numero de aeropuertos unicos;
   - verifica que todas las columnas tengan el mismo tamano;
   - devuelve un `CsvLoadResult`.

### Funciones auxiliares del lector

- `splitCsvLineSimple(...)`
  - parser CSV simple sin librerias externas;
  - soporta comillas basicas, comas dentro de campos quoted y campos vacios.
- `cleanQuotedToken(...)`
  - recorta espacios y elimina comillas envolventes.
- `parseFloatOrNan(...)`
  - convierte un token a `float`;
  - devuelve `NAN` si el dato no existe o no es valido.
- `parseIntFromFloatToken(...)`
  - convierte tokens como `1129806.0` a entero truncado;
  - usa `-1` como centinela si el dato falta.
- `validateHeader(...)`
  - comprueba que el fichero cargado coincide con el formato esperado.

### Estructuras de apoyo del lector

- `CsvLoadStats`
  - guarda filas leidas, filas almacenadas, filas descartadas y faltantes.
- `CsvLoadResult`
  - agrupa el dataset, las estadisticas, la ruta procesada y un mensaje de
    error si algo falla.

---

## Como funciona la interfaz de consola

La capa de consola esta en `cli_utils.*`.

### Objetivo de `cli_utils`

Separar del `main` todo lo relacionado con:

- impresion de menus;
- lectura segura de opciones;
- validacion de enteros;
- seleccion de ruta del CSV;
- cancelacion con `X`;
- pausas entre pantallas.

### Funciones mas importantes

- `printApplicationBanner()`
  - muestra el encabezado inicial del programa.
- `printMainMenu()`
  - imprime el menu principal.
- `promptDatasetPath(...)`
  - pide la ruta del CSV;
  - si detecta `src/data/Airline_dataset.csv`, la ofrece como valor por
    defecto.
- `readMainMenuOption()`
  - convierte la entrada del usuario en un `MainMenuOption`.
- `readSignedInt(...)`
  - valida enteros firmados para umbrales.
- `readBoundedIntOption(...)`
  - valida opciones numericas dentro de un rango cerrado.
- `waitForEnter()`
  - pausa la ejecucion para que el usuario pueda leer la pantalla.

### Opciones actuales del menu

- `1` Fase 01 - Retraso en salida
- `2` Fase 02 - Retraso en llegada
- `3` Fase 03 - Reduccion de retraso
- `4` Fase 04 - Histograma de aeropuertos
- `R` Recargar CSV
- `I` Ver estado de la aplicacion
- `X` Salir

### Submenus actuales

Cada fase ya tiene su propia pantalla de entrada:

- Fase 01
  - pide el umbral firmado para `DEP_DELAY`.
- Fase 02
  - pide el umbral firmado para `ARR_DELAY`.
- Fase 03
  - pide columna y tipo de reduccion.
- Fase 04
  - pide origen/destino y umbral minimo.

Por ahora, tras recoger esos parametros, la aplicacion:

- resume la configuracion elegida;
- muestra la configuracion de lanzamiento sugerida si ya hay dataset y GPU;
- avisa de que la logica CUDA de esa fase aun esta pendiente.

---

## Como funciona `main.cu`

`main.cu` es el orquestador del proyecto.

### Tipos importantes definidos en `main.cu`

- `LaunchConfig`
  - guarda `blocks` y `threadsPerBlock`.
- `AppState`
  - guarda:
    - ruta del dataset activa;
    - estado de carga del CSV;
    - `CsvLoadResult`;
    - estado CUDA;
    - propiedades del dispositivo detectado.

### Funciones principales de `main.cu`

- `buildDatasetCandidates(...)`
  - construye la lista de rutas candidatas del CSV;
  - prioriza la ultima ruta usada y la ruta local `src/data/Airline_dataset.csv`.
- `queryGpuInfo(...)`
  - consulta si hay GPU CUDA y rellena `cudaDeviceProp`.
- `computeLaunchConfig(...)`
  - calcula una configuracion sugerida:
    - hasta 256 hilos por bloque;
    - numero de bloques suficiente para cubrir el dataset.
- `printLoadSummary(...)`
  - imprime el resumen de limpieza del CSV.
- `printGpuSummary(...)`
  - imprime el resumen del hardware CUDA.
- `printApplicationState(...)`
  - combina estado del dataset y estado CUDA.
- `loadDatasetIntoState(...)`
  - llama a `loadDataset(...)` y actualiza `AppState`.
- `promptAndLoadDataset(...)`
  - controla el ciclo completo de preguntar ruta e intentar cargar.
- `printSuggestedLaunchConfigIfAvailable(...)`
  - muestra una configuracion futura de lanzamiento para el dataset cargado.
- `runPhase1Shell(...)`
- `runPhase2Shell(...)`
- `runPhase3Shell(...)`
- `runPhase4Shell(...)`
  - son los submenus actuales de las fases futuras.

### Flujo de llamadas real

```text
main()
-> printApplicationBanner()
-> queryGpuInfo()
-> printGpuSummary()
-> promptAndLoadDataset()
   -> promptDatasetPath()
   -> loadDatasetIntoState()
      -> loadDataset()
         -> splitCsvLineSimple()
         -> cleanQuotedToken()
         -> validateHeader()
         -> parseFloatOrNan()
         -> parseIntFromFloatToken()
-> waitForEnter()
-> bucle del menu principal
   -> readMainMenuOption()
   -> runPhase1Shell() / runPhase2Shell() / runPhase3Shell() / runPhase4Shell()
   -> printApplicationState()
   -> promptAndLoadDataset() al recargar
```

---

## El kernel que existe ahora mismo

Aunque el flujo principal actual no lo llama, el repositorio mantiene un kernel
en `kernels.cu`:

```cpp
__global__ void reductionSimple(int* data, int* result, int n, bool isMax);
```

### Que hace `reductionSimple`

Cada hilo:

1. calcula su indice global 1D;
2. comprueba si ese indice esta dentro de rango;
3. lee un valor del vector `data`;
4. aplica:
   - `atomicMax(result, value)` si se busca maximo;
   - `atomicMin(result, value)` si se busca minimo.

### Para que sirve hoy

Sirve como referencia del trabajo CUDA ya existente en el repositorio y como
base de lo que en su momento fue una reduccion simple.

### Que no hace hoy

- no se lanza desde el `main` actual;
- no participa en la Fase 0;
- no implementa aun el flujo completo exigido por la practica para Fase 03.

---

## Que se procesa en CPU y que se procesara en GPU

### Hoy en CPU

- lectura del CSV;
- limpieza de datos;
- validacion de cabecera;
- almacenamiento por columnas;
- conteo de faltantes;
- conteo de aeropuertos unicos;
- menus por consola;
- captura de parametros del usuario;
- consulta del hardware CUDA.

### Hoy en GPU

- nada dentro del flujo principal actual;
- solo existe el kernel `reductionSimple` como implementacion aislada.

### Futuro previsto segun la estructura actual

- Fase 01
  - filtrado por umbral sobre `DEP_DELAY`.
- Fase 02
  - deteccion por `ARR_DELAY` y `TAIL_NUM` con memoria constante y atomicas.
- Fase 03
  - reducciones de maximo/minimo sobre columnas de retraso.
- Fase 04
  - histograma por aeropuerto, previsiblemente basado en `SEQ_ID`.

---

## Como defender el proyecto en su estado actual

Si hay que explicar el codigo hoy, la idea clave es esta:

- el proyecto ya tiene una **base de datos limpia y reutilizable en host**;
- ya tiene una **CLI completa** para el flujo de la practica;
- ya tiene un **estado global coherente** en `AppState`;
- ya conoce el **hardware CUDA disponible**;
- ya conserva un **kernel simple** de reduccion como referencia;
- pero todavia falta conectar e implementar la logica real de las Fases 1-4.

La defensa actual debe centrarse en:

- por que se usa almacenamiento por columnas;
- por que los faltantes se guardan como `NAN`;
- por que los IDs enteros usan `-1` como centinela;
- por que la CLI esta separada en `cli_utils`;
- por que se consulta el hardware antes de fijar una configuracion de lanzamiento;
- y por que el kernel existente aun no forma parte del flujo principal.

---

## Limitaciones actuales

En el estado actual del proyecto:

- las Fases 1-4 aun no ejecutan su computo CUDA real;
- no hay comprobacion sistematica de errores CUDA en cada llamada futura porque
  esas llamadas aun no se han integrado;
- el build depende de que cada maquina tenga `CUDA_PATH` bien definido o, en su
  defecto, un `cuda.local.props` correcto;
- el parser CSV es sencillo y deliberadamente limitado al dataset de la practica;
- la aplicacion depende de que la cabecera del CSV coincida con el formato
  esperado;
- no hay tests automaticos integrados en este entorno.

---

## Siguiente paso natural

El siguiente paso tecnico coherente es implementar la **Fase 01** sobre esta
base ya preparada:

- reutilizando `DatasetColumns`;
- construyendo un buffer entero a partir de `DEP_DELAY`;
- usando `LaunchConfig` calculado segun hardware;
- y conectando desde `runPhase1Shell(...)` el kernel correspondiente.
