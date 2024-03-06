import cv2

# Ouvrir la vidéo
video_capture = cv2.VideoCapture('Trial.avi')

# Initialiser le détecteur de mouvement
object_detector = cv2.createBackgroundSubtractorMOG2()

# Récupérer les propriétés de la vidéo
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Définir le codec et le créateur de vidéo
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('Trial-Mov.avi', fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,23,3)
    # Appliquer le détecteur de mouvement
    mask = object_detector.apply(thresh)

    # Trouver les contours des objets en mouvement
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dessiner des rectangles autour des objets en mouvement
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Écrire la frame avec les rectangles encadrant les objets en mouvement
    output_video.write(frame)

    # Afficher la vidéo avec les rectangles en temps réel
    cv2.imshow('Motion Detection', frame)

    # Sortir de la boucle si la touche 'q' est pressée
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Libérer les ressources
video_capture.release()
output_video.release()
cv2.destroyAllWindows()
