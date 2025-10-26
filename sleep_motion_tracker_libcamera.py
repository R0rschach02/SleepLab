# sleep_motion_tracker_video.py
# Schlafbewegungserkennung mit OpenCV (BackgroundSubtractorMOG2) für Videodateien

import cv2
import numpy as np
import datetime
import csv
import os
import sys

# --- KONFIGURATION ---
VIDEO_FILE = "schlaftest.mp4"  # Pfad zur zu analysierenden Videodatei
LOG_FILE = "motion_log.csv"
MIN_CONTOUR_AREA = 1000
LOG_INTERVAL = 30  # Alle 30 Frames einen Log-Eintrag erstellen
SHOW_PREVIEW = True  # Auf False setzen für headless Betrieb
# --- ENDE KONFIGURATION ---

def setup_video_capture(video_path):
    """Öffnet die Videodatei zum Lesen."""
    if not os.path.exists(video_path):
        print(f"FEHLER: Videodatei '{video_path}' nicht gefunden!")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"FEHLER: Videodatei '{video_path}' konnte nicht geöffnet werden!")
        return None
    
    # Video-Informationen ausgeben
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"Videodatei geladen: {video_path}")
    print(f"  Auflösung: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Frames: {frame_count}")
    print(f"  Dauer: {duration:.2f} Sekunden ({duration/60:.2f} Minuten)")
    
    return cap

def setup_logger(log_file):
    """Stellt sicher, dass die Log-Datei existiert und schreibt den Header."""
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame_number", "timestamp", "is_motion_detected", "max_contour_area"])
        print(f"Log-Datei '{log_file}' erstellt.")

def log_motion(frame_number, timestamp, is_detected, area):
    """Schreibt den Bewegungsstatus in die Log-Datei."""
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([frame_number, timestamp, is_detected, area])

def process_frame(frame, fgbg):
    """Verarbeitet einen einzelnen Frame zur Bewegungserkennung."""
    
    # 1. Konvertierung zu Graustufen 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Glätten (Blurring) zur Reduzierung von Rauschen
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # 3. Hintergrundsubtraktion
    fgmask = fgbg.apply(gray)
    
    # 4. Morphologische Operationen (Rauschen entfernen)
    fgmask = cv2.erode(fgmask, None, iterations=2)
    fgmask = cv2.dilate(fgmask, None, iterations=2)
    
    # 5. Konturen finden
    contours, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_detected = False
    max_area = 0
    
    # Iteriere über alle gefundenen Konturen
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 6. Konturen filtern: Ignoriere Konturen, die kleiner als die Mindestfläche sind
        if area > MIN_CONTOUR_AREA:
            motion_detected = True
            if area > max_area:
                max_area = area
                
            # Zeichne eine Bounding Box um die Bewegung (nur für visuelles Feedback)
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Bewegung", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, motion_detected, max_area

def main():
    # Überprüfe, ob eine Videodatei als Argument übergeben wurde
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    else:
        video_file = VIDEO_FILE
    
    # Initialisiere Logger und Video
    setup_logger(LOG_FILE)
    
    cap = setup_video_capture(video_file)
    if cap is None:
        print("Video konnte nicht geladen werden. Programm wird beendet.")
        return

    # Initialisiere den Hintergrund-Subtraktor MOG2
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    print("\nStarte Bewegungsanalyse...")
    if SHOW_PREVIEW:
        print("Drücke 'q' im Fenster zum Abbrechen, 'p' zum Pausieren, Leertaste zum Fortsetzen.")
    
    frame_number = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    paused = False

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    print("\nEnde der Videodatei erreicht.")
                    break
                
                frame_number += 1
                
                # Berechne den Timestamp basierend auf Frame und FPS
                timestamp_seconds = frame_number / fps if fps > 0 else frame_number
                timestamp = str(datetime.timedelta(seconds=int(timestamp_seconds)))
                
                # Verarbeite den Frame
                processed_frame, motion_detected, max_area = process_frame(frame, fgbg)
                
                # Logging-Logik
                if frame_number % LOG_INTERVAL == 0:
                    log_motion(frame_number, timestamp, motion_detected, max_area)
                    
                    # Fortschrittsanzeige
                    progress = (frame_number / total_frames) * 100
                    print(f"Fortschritt: {progress:.1f}% (Frame {frame_number}/{total_frames}) - "
                          f"Zeit: {timestamp} - Bewegung: {'Ja' if motion_detected else 'Nein'}")
                
                # Zeige den verarbeiteten Frame an
                if SHOW_PREVIEW:
                    # Füge Informationen zum Frame hinzu
                    info_text = f"Frame: {frame_number}/{total_frames} | Zeit: {timestamp}"
                    status_text = f"Bewegung: {'ERKANNT' if motion_detected else 'Keine'}"
                    
                    cv2.putText(processed_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(processed_frame, status_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                               (0, 255, 0) if motion_detected else (255, 255, 255), 2)
                    
                    display_frame = cv2.resize(processed_frame, (960, 540))
                    cv2.imshow('Sleep Motion Tracker - Video Analysis', display_frame)
            
            # Tastatureingaben verarbeiten
            if SHOW_PREVIEW:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nAnalyse durch Benutzer abgebrochen.")
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("\nVideo pausiert." if paused else "\nVideo fortgesetzt.")
                elif key == ord(' '):
                    paused = False

    except KeyboardInterrupt:
        print("\nProgramm durch Benutzer beendet.")
    finally:
        # Aufräumen
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nAnalyse beendet. Ergebnisse in '{LOG_FILE}'.")
        print(f"Insgesamt {frame_number} Frames analysiert.")

if __name__ == "__main__":
    main()