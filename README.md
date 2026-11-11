# 🥽 SR University — Augmented & Virtual Reality Lab Suite (23CS201PE403)

[![SR University](https://img.shields.io/badge/SR%20University-B.Tech%20CSE%20%2F%20CSAI-0284C7.svg)](https://sru.edu.in)
[![Course: 23CS201PE403](https://img.shields.io/badge/Course-23CS201PE403%20AR%2FVR-purple.svg)](https://sru.edu.in)
[![Unity 6 / WebGL](https://img.shields.io/badge/Platform-Unity%206%20%7C%20Three.js%20WebGL-646CFF.svg)](https://unity.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-amber.svg)](https://opensource.org/licenses/MIT)

> **School of Computer Science & Artificial Intelligence, SR University, Warangal**  
> **Candidate**: Seetharam Ravula (`2303A52440`)  
> **Course**: Augmented and Virtual Reality Lab (`23CS201PE403`)  
> **Regulation**: R23 | **Academic Year**: 2026 – 2027  

---

## 🌐 Quick Access: Direct Links (Localhost 3000)

Once the local server is running (`python -m http.server 3000`), open each assignment directly in your browser:

| Assignment | Title | Direct Localhost Link | Interactive Features |
| :---: | :--- | :--- | :--- |
| **02** | **Sphere Rotation & Treasure Hunt** | **[`http://localhost:3000/#assignment2`](http://localhost:3000/#assignment2)** | 3D wireframe rotating sphere, live Inspector `Transform.Rotation (X, Y, Z)`, speed slider, axis toggles. |
| **03** | **OpenXR & XR Device Simulator** | **[`http://localhost:3000/#assignment3`](http://localhost:3000/#assignment3)** | Headset-free VR scene, mouse-look HMD rotation (`Yaw/Pitch`), 6-DOF controller tracking telemetry. |
| **04** | **Continuous VR Locomotion** | **[`http://localhost:3000/#assignment4`](http://localhost:3000/#assignment4)** | Continuous Move Provider, WASD / on-screen thumbstick, collider wall hit detection, variable speed. |
| **06** | **AR Foundation & ARCore Tap-to-Place** | **[`http://localhost:3000/#assignment6`](http://localhost:3000/#assignment6)** | Planar surface detection, screen tap raycasting (`PlaneWithinPolygon`), 10cm cube pose anchoring. |
| **09** | **VR World Space UI & Gaze Menus** | **[`http://localhost:3000/#assignment9`](http://localhost:3000/#assignment9)** | 3D Museum gallery, floating World Space Canvas, laser ray pointer, 1.0s gaze dwell countdown timer. |

---

## 💻 Terminal Execution (Runs Without Unity GUI)

You can run each assignment individually or all together directly in PowerShell / Command Prompt:

```powershell
# Run Assignment 2 (3D Sphere Rotation)
python run_all_arvr_assignments.py 2

# Run Assignment 3 (OpenXR VR Scene Simulator)
python run_all_arvr_assignments.py 3

# Run Assignment 4 (Continuous VR Movement)
python run_all_arvr_assignments.py 4

# Run Assignment 6 (Mobile AR Foundation Tap-to-Place)
python run_all_arvr_assignments.py 6

# Run Assignment 9 (World Space UI & Gaze Menus)
python run_all_arvr_assignments.py 9

# Run ALL 5 Assignments sequentially (Full Automated Validation Suite)
python run_all_arvr_assignments.py all
```

---

## 📁 Repository Structure

```text
.
├── index.html                   # Unified 3D WebGL Browser Execution Hub (Localhost:3000)
├── run_all_arvr_assignments.py  # Master Terminal Runner for all AR/VR assignments
├── run_sphere_terminal.py       # Standalone 3D ASCII Sphere Rotation Engine
├── Scripts/                     # Production Unity C# Scripts
│   ├── SphereRotation.cs        # Assignment 2: Continuous Transform.Rotate MonoBehaviour
│   ├── TapToPlace.cs            # Assignment 6: ARRaycastManager plane tap-to-place
│   ├── VRMuseumMenuManager.cs   # Assignment 9: World Space UI panel management
│   └── VRGazeDwellButton.cs     # Assignment 9: 1.0s Gaze Dwell circular radial selection
├── .github/workflows/pages.yml  # Automated GitHub Pages CI/CD pipeline
└── README.md                    # Comprehensive documentation and quick-access links
```

---

## 📄 License
This repository is licensed under the [MIT License](LICENSE).
Free for academic, research, and laboratory evaluation.
