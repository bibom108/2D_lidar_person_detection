import time
import numpy as np
import matplotlib.pyplot as plt
from rplidar import RPLidar

from dr_spaam.utils import utils as u
from dr_spaam.detector import Detector

np.int = int


# ---------- CONFIG ----------
PORT = "COM3"
NUM_PTS = 720          # 0.5° resolution over 360°
MAX_RANGE = 8.0        # small room limit
MIN_RANGE = 0.10
X_LIM = (-5, 5)
Y_LIM = (-5, 5)
# ----------------------------


def scan_to_fixed_beams(meas):
    ranges = np.full((NUM_PTS,), MAX_RANGE, dtype=np.float32)

    for q, ang_deg, dist_mm in meas:
        if dist_mm <= 0:
            continue

        r = dist_mm * 1e-3
        if r < MIN_RANGE or r > MAX_RANGE:
            continue

        a = ang_deg % 360.0
        idx = int(a / 360.0 * NUM_PTS)

        if r < ranges[idx]:
            ranges[idx] = r

    return ranges


def main():

    # ---- Load detector ----
    ckpt_file = "/home/phuc/2D_lidar_person_detection/dr_spaam/ckpt/ckpt_jrdb_ann_dr_spaam_e20.pth"
    detector = Detector(ckpt_file, model="DR-SPAAM", gpu=True, stride=1, panoramic_scan=True)
    detector.set_laser_fov(360)

    # ---- Init lidar ----
    lidar = RPLidar(PORT)
    lidar.start_motor()

    # ---- Prepare plot ----
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    scan_phi = np.linspace(0, 2 * np.pi, NUM_PTS, endpoint=False)

    try:
        for meas in lidar.iter_scans(max_buf_meas=3000):

            scan_r = scan_to_fixed_beams(meas)

            # convert to XY
            scan_x, scan_y = u.rphi_to_xy(scan_r, scan_phi)

            # clear plot
            ax.cla()
            ax.set_aspect("equal")
            ax.set_xlim(X_LIM)
            ax.set_ylim(Y_LIM)

            # plot scan
            ax.scatter(scan_x, scan_y, s=2, c="black")

            # run detector
            dets_xy, dets_cls, _ = detector(scan_r)

            # plot detections
            thresh = 0.3
            for xy, cls in zip(dets_xy, dets_cls):
                if cls < thresh:
                    continue
                c = plt.Circle((xy[0], xy[1]), 0.4,
                               color="yellow",
                               fill=False,
                               linewidth=2)
                ax.add_artist(c)

            plt.pause(0.01)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        lidar.stop_motor()
        lidar.stop()
        lidar.disconnect()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
