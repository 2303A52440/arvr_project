import math
import os
import sys
import time

# Ensure UTF-8 output if supported, fallback to ASCII
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def clear_screen():
    if os.name == 'nt':
        os.system('cls')
    else:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

def run_sphere_rotation_terminal(duration=0, speed=50.0):
    """
    Simulates Unity's MonoBehaviour Lifecycle and 3D Transform.Rotate()
    directly in the terminal using ASCII 3D ray-surface projection.
    """
    width = 48
    height = 18
    scale_x = 1.4  # Matches Assignment 2 Step 12 (elongated sphere X=1.5, Y=1.0)
    scale_y = 1.0
    scale_z = 1.0

    rot_y = 0.0  # Euler angle Y in degrees
    start_time = time.time()
    last_frame_time = start_time

    # Pre-generate sphere surface points with latitude and longitude lines for visible rotation
    num_lat = 10
    num_lon = 20
    points = []
    for i in range(num_lat):
        lat = math.pi * (-0.5 + float(i + 1) / (num_lat + 1))
        for j in range(num_lon):
            lon = 2 * math.pi * float(j) / num_lon
            x = math.cos(lat) * math.sin(lon) * scale_x
            y = math.sin(lat) * scale_y
            z = math.cos(lat) * math.cos(lon) * scale_z
            # Marker stripes so rotation is plainly visible
            is_marker = (j % 5 == 0) or (i == num_lat // 2)
            points.append((x, y, z, is_marker))

    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            if duration > 0 and elapsed >= duration:
                break

            dt = current_time - last_frame_time
            if dt <= 0:
                dt = 0.016
            last_frame_time = current_time

            # Unity's Update() execution: transform.Rotate(0f, rotationSpeed * Time.deltaTime, 0f)
            rot_y = (rot_y + speed * dt) % 360.0
            rad_y = math.radians(rot_y)
            cos_y = math.cos(rad_y)
            sin_y = math.sin(rad_y)

            # Frame buffer
            buffer = [[" " for _ in range(width)] for _ in range(height)]
            z_buffer = [[-999.0 for _ in range(width)] for _ in range(height)]

            # Rotate points around Y axis and project to 2D
            for x, y, z, is_marker in points:
                rx = x * cos_y + z * sin_y
                ry = y
                rz = -x * sin_y + z * cos_y

                cam_dist = 2.6
                proj_z = rz + cam_dist

                screen_x = int(width / 2 + (rx / proj_z) * (width * 0.85))
                screen_y = int(height / 2 - (ry / proj_z) * (height * 0.85))

                if 0 <= screen_x < width and 0 <= screen_y < height:
                    if rz > z_buffer[screen_y][screen_x]:
                        z_buffer[screen_y][screen_x] = rz
                        if is_marker:
                            char = "@" if rz > 0 else "o"
                        else:
                            char = "#" if rz > 0.4 else ("*" if rz > 0 else ".")
                        buffer[screen_y][screen_x] = char

            # Build terminal output (Standard ASCII)
            lines = []
            lines.append("+" + "-" * 62 + "+")
            lines.append("|  UNITY ENGINE (TERMINAL PLAY MODE) - ASSIGNMENT 2 ROTATION   |")
            lines.append("+" + "-" * 62 + "+")
            lines.append("| Script: SphereRotation.cs  -> Update(): transform.Rotate()   |")
            lines.append("+--------------------------------------------------------------+")
            lines.append("| [INSPECTOR: RotatingSphere]                                  |")
            lines.append(f"|  Transform.Position: ( 0.00,  0.00,  0.00)                   |")
            lines.append(f"|  Transform.Rotation: (X: 0.00 deg, Y: {rot_y:6.1f} deg, Z: 0.00 deg) ↻   |")
            lines.append(f"|  Transform.Scale:    ( 1.50,  1.00,  1.00) (Elongated)       |")
            lines.append(f"|  SphereRotation.rotationSpeed: {speed:5.1f} deg/sec                  |")
            lines.append(f"|  Time.deltaTime: {dt:6.4f}s | FPS: {1.0/max(dt,0.001):4.0f}                        |")
            lines.append("+--------------------------------------------------------------+")
            lines.append("| [3D SCENE VIEW RENDERER]                                     |")
            for row in buffer:
                line_str = "".join(row)
                lines.append(f"|  {line_str.ljust(58)}  |")
            lines.append("+" + "-" * 62 + "+")
            lines.append(" [Press Ctrl+C to Exit Play Mode]")

            clear_screen()
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()

            time.sleep(0.04)  # ~25 FPS

    except KeyboardInterrupt:
        pass

    print("\n[Unity Play Mode Stopped] Program terminated cleanly.")

if __name__ == "__main__":
    speed = 50.0
    duration = 0  # 0 means continuous live rotation until Ctrl+C
    if len(sys.argv) > 1:
        try:
            speed = float(sys.argv[1])
        except ValueError:
            pass
    if len(sys.argv) > 2:
        try:
            duration = float(sys.argv[2])
        except ValueError:
            pass
    run_sphere_rotation_terminal(duration=duration, speed=speed)
