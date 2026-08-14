#!/usr/bin/env python3
"""
Camera intrinsic calibration (fx, fy, cx, cy + distortion) via a standard
OpenCV checkerboard calibration. Defaults to the DSJ-3079-HE (DSJSource).

Print a checkerboard with CHECKERBOARD_COLS x CHECKERBOARD_ROWS *inner*
corners -- e.g. OpenCV's standard sample pattern (9x6 inner corners, 10x7
squares): https://github.com/opencv/opencv/blob/4.x/doc/pattern.png
Any checkerboard works as long as CHECKERBOARD_COLS/ROWS below match it.
The physical square size does NOT need to be measured: camera intrinsics
(fx/fy/cx/cy) come from cross-view projective geometry, not absolute scale
-- it would only matter for recovering real-world pose/distance, which this
script doesn't do.

Usage:
    python calibrate_camera.py

Hold the checkerboard in front of the camera. Once corners are drawn on it
(green overlay), press SPACE to capture that pose, then move/tilt the board
and capture again -- vary angle, distance, and position (corners in every
part of the frame, not just the center) for a well-conditioned calibration.
Press 'c' once you have at least MIN_CAPTURES to run calibration, or 'q' to
quit without calibrating.

Output: prints the camera matrix (fx, fy, cx, cy) and distortion
coefficients -- including the exact <arg> lines to paste into
ibvs_perching/launch/ibvs_udp.launch on the docker side -- and saves the
full result to calibration_result.yaml next to this script.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from sources.DSJSource import DSJSource

CHECKERBOARD_COLS = 9   # inner corners, not squares
CHECKERBOARD_ROWS = 6
MIN_CAPTURES = 15

OUTPUT_PATH = _HERE / "calibration_result.yaml"


def main():
    source = DSJSource()

    objp = np.zeros((CHECKERBOARD_ROWS * CHECKERBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_COLS, 0:CHECKERBOARD_ROWS].T.reshape(-1, 2)

    objpoints = []  # 3D points (arbitrary units -- see module docstring)
    imgpoints = []  # 2D points (pixels)
    frame_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print(f"[calibrate] Looking for a {CHECKERBOARD_COLS}x{CHECKERBOARD_ROWS} "
          f"checkerboard (inner corners). SPACE to capture, 'c' to calibrate "
          f"(need >= {MIN_CAPTURES}), 'q' to quit.")

    try:
        while True:
            ret, frame, _ = source.read()
            if not ret:
                print("[calibrate] Camera read failed, stopping.")
                break
            frame_size = frame.shape[:2][::-1]  # (w, h)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, (CHECKERBOARD_COLS, CHECKERBOARD_ROWS),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            display = frame.copy()
            corners_refined = None
            if found:
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )
                cv2.drawChessboardCorners(
                    display, (CHECKERBOARD_COLS, CHECKERBOARD_ROWS), corners_refined, found
                )

            cv2.putText(
                display, f"captures: {len(objpoints)}/{MIN_CAPTURES}+  "
                         f"{'BOARD FOUND (SPACE to capture)' if found else 'no board'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if found else (0, 0, 255), 2,
            )
            cv2.imshow("Calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and found:
                objpoints.append(objp.copy())
                imgpoints.append(corners_refined)
                print(f"[calibrate] Captured pose {len(objpoints)}")
            elif key == ord('c'):
                break
            elif key == ord('q'):
                objpoints = []
                break
    finally:
        source.release()
        cv2.destroyAllWindows()

    if len(objpoints) < MIN_CAPTURES:
        print(f"[calibrate] Only {len(objpoints)} captures (< {MIN_CAPTURES}), "
              f"not calibrating. Re-run and capture more poses.")
        return

    print(f"[calibrate] Calibrating from {len(objpoints)} captures...")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, frame_size, None, None
    )

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    print(f"\n[calibrate] RMS reprojection error: {rms:.4f} px "
          f"({'good' if rms < 0.5 else 'high -- consider recapturing with more/better poses'})")
    print(f"[calibrate] Resolution: {frame_size[0]}x{frame_size[1]}")
    print(f"[calibrate] fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    print(f"[calibrate] distortion coeffs (k1,k2,p1,p2,k3): {dist_coeffs.flatten().tolist()}")
    print(f"\n[calibrate] For ibvs_perching/launch/ibvs_udp.launch:")
    print(f'  <arg name="cx" default="{cx:.1f}"/>')
    print(f'  <arg name="cy" default="{cy:.1f}"/>')
    print(f'  <arg name="fx" default="{fx:.1f}"/>')
    print(f'  <arg name="fy" default="{fy:.1f}"/>')

    result = {
        "resolution": {"width": int(frame_size[0]), "height": int(frame_size[1])},
        "camera_matrix": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "distortion_coefficients": dist_coeffs.flatten().tolist(),
        "rms_reprojection_error_px": float(rms),
        "n_captures": len(objpoints),
    }
    with open(OUTPUT_PATH, "w") as f:
        yaml.dump(result, f, default_flow_style=False)
    print(f"\n[calibrate] Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
