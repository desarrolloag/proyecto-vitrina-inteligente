"""
Proyecto Vitrina Inteligente v5.2 (Final, Estable y Optimizado)
- Corregido error NameError de proc_w.
- Rendimiento mejorado ejecutando la detección de rostros de forma intermitente.
- Filtro de Proximidad y Orientación por Proporción.
- Salida digital GPIO para actuar en el mundo físico.
"""

# --- IMPORTACIONES Y DEPENDENCIAS ---
import os, time, datetime, cv2, tkinter as tk, numpy as np
import RPi.GPIO as GPIO 
from aiymakerkit import vision, utils
from pycoral.adapters.detect import BBox
import models

# --- CONFIGURACIÓN GENERAL ---
BLUE, GREEN, RED, WHITE, YELLOW = (255,0,0), (0,255,0), (0,0,255), (255,255,255), (0,255,255)

# --- CONFIGURACIÓN DE PINES GPIO ---
LED_PIN = 17
GPIO.setwarnings(False); GPIO.setmode(GPIO.BCM); GPIO.setup(LED_PIN, GPIO.OUT); GPIO.output(LED_PIN, GPIO.LOW)

# --- ¡MODIFICA Y CALIBRA AQUÍ! ---
FENCE_X_MIN_REL, FENCE_Y_MIN_REL = 0.2, 0.0
FENCE_X_MAX_REL, FENCE_Y_MAX_REL = 0.8, 1.0
POSITIVE_IMPACT_SECONDS = 10.0
MIN_PERSON_AREA = 65000#35000 
MAX_FACE_ASPECT_RATIO = 1.4 

# --- PARÁMETROS DE OPTIMIZACIÓN ---
PROCESSING_WIDTH = 640
GRACE_PERIOD_SECONDS = 2.0
PROCESS_FACES_EVERY_N_FRAMES = 5

# --- FUNCIONES DE AYUDA ---
def path(name):
    root = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(root, name)

def scale_bbox(bbox, scale_x, scale_y):
    xmin, ymin, xmax, ymax = bbox
    return BBox(xmin=int(xmin * scale_x), ymin=int(ymin * scale_y),
                xmax=int(xmax * scale_x), ymax=int(ymax * scale_y))

# --- Carga de DOS modelos de IA ---
OBJECT_DETECTION_MODEL_PATH = path('ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')
object_detector = vision.Detector(OBJECT_DETECTION_MODEL_PATH)
labels = utils.read_labels_from_metadata(OBJECT_DETECTION_MODEL_PATH)
FACE_DETECTION_MODEL_PATH = path('ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite')
face_detector = vision.Detector(FACE_DETECTION_MODEL_PATH)

# --- Configuración de la Pantalla Completa ---
try:
    root = tk.Tk()
    root.withdraw()
    screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
except tk.TclError: screen_width, screen_height = 1280, 720
WINDOW_NAME = 'Dashboard - Vitrina Inteligente (Optimizado)'
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# --- Inicio de la Aplicación ---
cap = cv2.VideoCapture(0)
if not cap.isOpened(): exit("Error Crítico: No se pudo abrir la cámara.")
os.makedirs('evidencia', exist_ok=True)

start_time, last_frame_had_attention, impact_counted_for_this_person, positive_impact_count, disappearance_time = None, False, False, 0, None
frame_counter = 0
last_known_persons_data = [] 
last_known_attention_count = 0

try:
    while True:
        ret, frame_original = cap.read()
        if not ret: break
        frame_counter += 1

        original_height, original_width, _ = frame_original.shape
        aspect_ratio = original_height / original_width
        processing_height = int(PROCESSING_WIDTH * aspect_ratio)
        frame_procesado = cv2.resize(frame_original, (PROCESSING_WIDTH, processing_height))
        scale_x, scale_y = original_width / PROCESSING_WIDTH, original_height / processing_height

        # La detección de rostros (pesada) solo se ejecuta cada N frames.
        if frame_counter % PROCESS_FACES_EVERY_N_FRAMES == 0:
            last_known_persons_data = [] # Reiniciar la lista de resultados conocidos
            last_known_attention_count = 0
            
            objects = object_detector.get_objects(frame_procesado, threshold=0.5)
            for person_obj in objects:
                if 'person' in labels.get(person_obj.id, ''):
                    is_qualified_attention = False # Asumir que no hay atención para esta persona
                    
                    if person_obj.bbox.area >= MIN_PERSON_AREA:
                        # --- LÍNEA CORREGIDA AÑADIDA AQUÍ ---
                        proc_h, proc_w, _ = frame_procesado.shape
                        fence_box_proc = BBox(*[int(c * s) for c, s in zip([FENCE_X_MIN_REL, FENCE_Y_MIN_REL, FENCE_X_MAX_REL, FENCE_Y_MAX_REL], [proc_w, proc_h, proc_w, proc_h])])
                        is_inside_fence = (BBox.intersect(person_obj.bbox, fence_box_proc).area / person_obj.bbox.area) > 0.3
                        
                        if is_inside_fence:
                            (xmin, ymin, xmax, ymax) = person_obj.bbox
                            person_image = frame_procesado[ymin:ymax, xmin:xmax]
                            if person_image.size > 0:
                                faces = face_detector.get_objects(person_image, threshold=0.4)
                                if faces:
                                    face_bbox = faces[0].bbox
                                    if face_bbox.width > 0:
                                        aspect_ratio = face_bbox.height / face_bbox.width
                                        if aspect_ratio < MAX_FACE_ASPECT_RATIO:
                                            is_qualified_attention = True
                    
                    last_known_persons_data.append({'bbox': person_obj.bbox, 'attention': is_qualified_attention})
                    if is_qualified_attention:
                        last_known_attention_count += 1
        
        # Dibujamos en cada frame usando la información guardada.
        person_count_in_frame = len(last_known_persons_data)
        attention_count_in_frame = last_known_attention_count
        
        proc_h, proc_w, _ = frame_procesado.shape
        fence_box_proc = BBox(*[int(c * s) for c, s in zip([FENCE_X_MIN_REL, FENCE_Y_MIN_REL, FENCE_X_MAX_REL, FENCE_Y_MAX_REL], [proc_w, proc_h, proc_w, proc_h])])
        scaled_fence_box = scale_bbox(fence_box_proc, scale_x, scale_y)
        vision.draw_rect(frame_original, scaled_fence_box, BLUE, 3)

        for person_data in last_known_persons_data:
            color = RED if person_data['attention'] else GREEN
            scaled_person_box = scale_bbox(person_data['bbox'], scale_x, scale_y)
            vision.draw_rect(frame_original, scaled_person_box, color)

        active_attention_in_frame = attention_count_in_frame > 0

        if active_attention_in_frame:
            GPIO.output(LED_PIN, GPIO.HIGH)
        else:
            GPIO.output(LED_PIN, GPIO.LOW)

        if active_attention_in_frame:
            disappearance_time = None
            if not last_frame_had_attention:
                start_time = time.time()
                impact_counted_for_this_person = False
        else:
            if last_frame_had_attention and disappearance_time is None:
                disappearance_time = time.time()
            if disappearance_time is not None and (time.time() - disappearance_time) > GRACE_PERIOD_SECONDS:
                start_time = None
                disappearance_time = None
        last_frame_had_attention = active_attention_in_frame
        
        scale = min(screen_width / original_width, screen_height / original_height)
        new_width, new_height = int(original_width * scale), int(original_height * scale)
        resized_frame = cv2.resize(frame_original, (new_width, new_height))
        
        fullscreen_canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        x_offset, y_offset = (screen_width - new_width) // 2, (screen_height - new_height) // 2
        fullscreen_canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame

        dwell_time = 0
        if start_time is not None:
            dwell_time = time.time() - start_time
            cv2.putText(fullscreen_canvas, "ATENCION ACTIVA!!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, YELLOW, 3)
            timer_text = "Tiempo de Atencion: {:.1f}s".format(dwell_time)
            cv2.putText(fullscreen_canvas, timer_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

        impact_text = f"Impactos Reales: {positive_impact_count}"
        text_size = cv2.getTextSize(impact_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        cv2.putText(fullscreen_canvas, impact_text, (screen_width - text_size[0] - 20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, YELLOW, 3)

        person_count_text = f"Personas en Cuadro: {person_count_in_frame}"
        text_size_person = cv2.getTextSize(person_count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cv2.putText(fullscreen_canvas, person_count_text, (screen_width - text_size_person[0] - 20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)

        if start_time is not None and dwell_time > POSITIVE_IMPACT_SECONDS and not impact_counted_for_this_person:
            if attention_count_in_frame > 0:
                positive_impact_count += attention_count_in_frame
                impact_counted_for_this_person = True
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"evidencia/atencion_activa_{timestamp}_x{attention_count_in_frame}.jpg"
                cv2.imwrite(filename, fullscreen_canvas)
                print(f"¡Impacto positivo ({attention_count_in_frame} personas)! Evidencia guardada en: {filename}")

        cv2.imshow(WINDOW_NAME, fullscreen_canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    print("Cerrando aplicación...")
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()