"""PC-side data logger for ESP32 EMG+IMU streaming."""
import argparse
import csv
from pathlib import Path

import serial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record EMG/IMU data over serial")
    parser.add_argument("--port", required=True, help="Serial port e.g. COM3 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate")
    parser.add_argument("--subject", required=True, help="Subject ID")
    parser.add_argument("--output", default="../data", help="Output folder for CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_file = output_path / f"subject_{args.subject}_raw.csv"

    ser = serial.Serial(args.port, args.baud)

    with csv_file.open("a", newline="") as f:
        writer = csv.writer(f)
        while True:
            line = ser.readline().decode().strip()
            if not line:
                continue
            row = line.split(",")
            # TODO: add real-time label acquisition (e.g., keyboard input)
            writer.writerow(row)
            f.flush()


if __name__ == "__main__":
    main()
