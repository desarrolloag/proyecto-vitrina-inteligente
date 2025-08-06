# Proyecto Vitrina Inteligente con Raspberry Pi y Coral TPU 🤖️

Este proyecto utiliza una Raspberry Pi, un acelerador Coral de Google y visión por computadora para detectar la atención de las personas hacia una vitrina o exhibición. Cuando se detecta una "atención cualificada", el sistema registra un "impacto" y puede activar una salida física, como un LED.

![placeholder](https://i.imgur.com/roQvi7U.jpg)

## Índice

* [Hardware Requerido](#-hardware-requerido)
* [Configuración del Software](#-configuración-del-software)
* [Conexión del Hardware (GPIO)](#-conexión-del-hardware-gpio)
* [Estructura del Repositorio](#-estructura-del-repositorio)
* [Ejecutando el Proyecto](#-ejecutando-el-proyecto)
* [Solución de Problemas](#-solución-de-problemas)

---

### Hardware Requerido

Para montar este proyecto, necesitarás los siguientes componentes:

* **Procesamiento:**
    * [Raspberry Pi 4](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/) (se recomienda el modelo de 4GB de RAM o superior).
    * [Acelerador USB Coral Edge TPU](https://coral.ai/products/accelerator/).
* **Periféricos:**
    * Cámara USB genérica (compatible con Linux/UVC).
    * Tarjeta MicroSD de 16 GB o más (Clase 10).
    * Fuente de alimentación USB-C de calidad para la Raspberry Pi (5V, 3A).
* **Componentes de Salida (Opcional):**
    * 1x LED del color que prefieras.
    * 1x Resistencia de ~330Ω (Ohms).
    * Protoboard y cables Jumper (macho-hembra).

---

### ⚙️ Configuración del Software

Sigue estos pasos para preparar tu Raspberry Pi desde cero.

#### **Paso 1: Instalar el Sistema Operativo**

La forma más sencilla es usar la imagen de sistema operativo que ya incluye los drivers de Coral.

1.  Descarga e instala la herramienta **Raspberry Pi Imager** en tu computadora.
2.  Descarga la imagen del sistema operativo **AIY Maker Kit**: `aiy-maker-kit-2022-05-18.img.xz` (puedes buscarla en la web o usar la que ya tienes).
3.  En Raspberry Pi Imager, selecciona "Choose OS" -> "Use custom" y elige el archivo `.img.xz` que descargaste.
4.  Selecciona tu tarjeta MicroSD y haz clic en "WRITE".

#### **Paso 2: Configuración Inicial del Sistema**

1.  Inserta la MicroSD en tu Raspberry Pi, conecta el monitor, teclado, mouse, la cámara USB y el acelerador Coral USB. Enciéndela.
2.  Completa el asistente de configuración inicial de Raspberry Pi OS (configurar idioma, contraseña, red WiFi, etc.).
3.  Abre una terminal y actualiza el sistema. Es una buena práctica.
    ```bash
    sudo apt-get update
    sudo apt-get upgrade -y
    ```

#### **Paso 3: Clonar este Repositorio**

Abre una terminal y clona este repositorio para obtener todos los archivos del proyecto.
```bash
git clone [https://github.com/TU_USUARIO/NOMBRE_DEL_REPOSITORIO.git](https://github.com/TU_USUARIO/NOMBRE_DEL_REPOSITORIO.git)
cd NOMBRE_DEL_REPOSITORIO
```
*(Recuerda cambiar la URL por la de tu propio repositorio)*

#### **Paso 4: Instalar Dependencias**

El script necesita algunas librerías de Python. La imagen de AIY Maker Kit ya incluye la mayoría, pero nos aseguraremos de tener todo.
```bash
pip3 install -r requirements.txt
```

---

### 🔌 Conexión del Hardware (GPIO)

Si deseas usar la salida del LED, conéctalo según el siguiente diagrama. El script usa el **pin GPIO 17**.

* **Pin GPIO 17** de la Raspberry Pi -> **Resistencia de 330Ω**.
* **Resistencia** -> **Pata larga (ánodo)** del LED.
* **Pata corta (cátodo)** del LED -> **Pin GND (Tierra)** de la Raspberry Pi.

![GPIO Diagram](https://i.imgur.com/J8B1s2W.png)

---

### 📂 Estructura del Repositorio

* **`E9_VITRINA_INTELIGENTE.py`**: El script principal de Python que ejecuta toda la lógica.
* **`models/`**: Esta carpeta debe contener los dos modelos de IA compilados para el Edge TPU.
    * `ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite` (Detector de objetos).
    * `ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite` (Detector de rostros).
* **`requirements.txt`**: Lista de librerías de Python necesarias para el proyecto.
* **`evidencia/`**: Carpeta creada automáticamente por el script para guardar las fotos de los "impactos".
* **`.gitignore`**: Archivo para que Git ignore carpetas y archivos innecesarios.
* **`README.md`**: Este mismo archivo de instrucciones.

---

### Ejecutando el Proyecto

#### **Paso 1: ¡Calibrar! (Muy Importante)**

Antes de ejecutar, abre el archivo `E9_VITRINA_INTELIGENTE.py` con un editor de texto (como Thonny, Geany o nano) y ajusta los parámetros en la sección `--- ¡MODIFICA Y CALIBRA AQUÍ! ---`.

* `FENCE_X_MIN_REL`, `FENCE_Y_MIN_REL`, etc.: Definen el **rectángulo azul (zona de interés)**. Ajústalo para que enmarque tu vitrina. Los valores van de 0.0 a 1.0.
* `POSITIVE_IMPACT_SECONDS`: El **tiempo en segundos** que una persona debe mantener la atención para que cuente como un "impacto".
* `MIN_PERSON_AREA`: El **tamaño mínimo del cuadro verde/rojo** de una persona para ser considerada. Aumenta este valor si quieres que solo cuenten las personas más cercanas.
* `MAX_FACE_ASPECT_RATIO`: Filtro de **orientación del rostro**. El valor por defecto (1.4) funciona bien para detectar rostros que miran hacia adelante.

#### **Paso 2: Ejecutar el Script**

Con todo conectado y calibrado, abre una terminal, navega a la carpeta del proyecto y ejecuta:
```bash
python3 E9_VITRINA_INTELIGENTE.py
```

El programa se abrirá en pantalla completa. Para cerrarlo, presiona la tecla **'q'**.

---

### 🤔 Solución de Problemas

* **Error "Camera not found" o la pantalla se queda en negro:**
    * Asegúrate de que la cámara USB esté bien conectada.
    * Reinicia la Raspberry Pi.
    * En la terminal, escribe `lsusb` para ver si la cámara es detectada por el sistema.
    * Si tienes varias cámaras, podrías necesitar cambiar `cv2.VideoCapture(0)` a `cv2.VideoCapture(1)` en el script.
* **El LED no enciende:**
    * Verifica 100% el cableado. Asegúrate de usar el pin **GPIO 17** y un pin **GND**.
    * Comprueba que el LED no esté al revés (pata larga con el pin, pata corta a tierra).
* **Bajo rendimiento (video lento):**
    * Asegúrate de que el **Acelerador Coral USB esté conectado firmemente**. La luz del acelerador debe encenderse.
    * Ejecuta `lsusb` en la terminal. Deberías ver un dispositivo de "Google, Inc.". Si no aparece, el driver podría no estar cargado.
