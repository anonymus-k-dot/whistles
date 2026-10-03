import os
import sys
import argparse
import urllib.request
import json
import csv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BASE_API = "https://storage.googleapis.com/storage/v1/b/noaa-passive-bioacoustic/o"
BASE_DOWNLOAD = "https://storage.googleapis.com/noaa-passive-bioacoustic"

def get_file_list_for_time_range(start_dt, end_dt, prefix="dclde/2022/dclde_2022_1705/audio/"):
    all_files = []
    page_token = None
    
    start_str = f"1705_{start_dt.strftime('%Y%m%d_%H%M%S')}"
    end_str = f"1705_{end_dt.strftime('%Y%m%d_%H%M%S')}"
    
    current_date = start_dt.date()
    end_date = end_dt.date()
    
    while current_date <= end_date:
        date_prefix = f"{prefix}1705_{current_date.strftime('%Y%m%d')}_"
        url = f"{BASE_API}?prefix={date_prefix}&maxResults=1000"
        
        while True:
            req_url = url + (f"&pageToken={page_token}" if page_token else "")
            req = urllib.request.urlopen(req_url)
            data = json.loads(req.read().decode())
            
            for item in data.get("items", []):
                fname = os.path.basename(item["name"])
                if fname.endswith(".flac"):
                    if fname >= start_str[:20] and fname <= end_str[:20] + "z":
                        all_files.append((item["name"], int(item.get("size", 0))))
            
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        
        current_date = datetime.fromordinal(current_date.toordinal() + 1).date()
        
    return all_files

def get_file_list_by_prefix(file_prefix, gcs_audio_dir="dclde/2022/dclde_2022_1705/audio/"):
    all_files = []
    page_token = None
    full_prefix = f"{gcs_audio_dir}{file_prefix}"
    url = f"{BASE_API}?prefix={full_prefix}&maxResults=1000"

    while True:
        req_url = url + (f"&pageToken={page_token}" if page_token else "")
        req = urllib.request.urlopen(req_url)
        data = json.loads(req.read().decode())
        for item in data.get("items", []):
            if item["name"].endswith(".flac"):
                all_files.append((item["name"], int(item.get("size", 0))))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return all_files

def download_file(item, output_dir, max_retries=3):
    remote_path, expected_size = item
    fname = os.path.basename(remote_path)
    local_path = os.path.join(output_dir, fname)
    
    # Check if already downloaded and matches size
    if os.path.exists(local_path):
        if os.path.getsize(local_path) == expected_size and expected_size > 0:
            return fname, "skipped", expected_size
    
    url = f"{BASE_DOWNLOAD}/{remote_path}"
    temp_path = local_path + ".tmp"
    
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "BioacousticResearch/1.0"})
            with urllib.request.urlopen(req, timeout=60) as response, open(temp_path, "wb") as out_file:
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
            
            # Check size if available
            if expected_size > 0 and os.path.getsize(temp_path) != expected_size:
                raise IOError(f"Size mismatch: got {os.path.getsize(temp_path)}, expected {expected_size}")

            if os.path.exists(local_path):
                os.remove(local_path)
            os.rename(temp_path, local_path)
            return fname, "downloaded", expected_size
        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            if attempt < max_retries:
                import time
                time.sleep(2 * attempt)
            else:
                return fname, f"error: {str(e)}", 0

def main():
    parser = argparse.ArgumentParser(description="Bulk Downloader for NOAA DCLDE 2022 Whale Audio Files")
    parser.add_argument("--species", type=str, default="",
                        help="Download all encounters for a species (e.g. 'false killer whale' or '33')")
    parser.add_argument("--encounter", type=str, default="",
                        help="Encounter ID from whale_encounters_1705.csv (e.g. 191)")
    parser.add_argument("--prefix", type=str, default="",
                        help="Audio filename prefix (e.g. 1705_20170929_22)")
    parser.add_argument("--output-dir", type=str, default="./fkw_audio",
                        help="Local folder where audio will be downloaded (default: ./fkw_audio)")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Maximum number of files to download (0 for all matching files)")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of parallel download threads (default: 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only list files and sizes without downloading")
    args = parser.parse_args()

    csv_file = "whale_encounters_1705.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run inspect_encounters.py first.")
        sys.exit(1)

    encounters = []
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            encounters.append(row)

    target_encounters = []
    if args.species:
        query = args.species.strip().lower()
        for row in encounters:
            if (query in row["common_name"].lower() or 
                query in row["scientific_name"].lower() or 
                query == row["species_id"] or 
                (query in ("fkw", "false killer whale") and row["species_id"] == "33")):
                target_encounters.append(row)
        if not target_encounters:
            print(f"No encounters found for species: '{args.species}'")
            return
    elif args.encounter:
        for row in encounters:
            if row["encounter_id"].strip() == args.encounter.strip():
                target_encounters.append(row)
                break
        if not target_encounters:
            print(f"Encounter #{args.encounter} not found.")
            return

    files = []
    if args.prefix:
        print("=" * 70)
        print(f"Mode: Download by filename prefix '{args.prefix}*'")
        print("=" * 70)
        files = get_file_list_by_prefix(args.prefix)
    elif target_encounters:
        print("=" * 70)
        print(f"Selected {len(target_encounters)} Encounter(s):")
        total_enc_min = sum(float(x["duration_minutes"]) for x in target_encounters)
        for x in target_encounters:
            print(f"  • Enc #{x['encounter_id']:<3} | {x['common_name']:<20} | {x['utc_start']} to {x['utc_end']} ({x['duration_minutes']} min)")
        print(f"Total Duration: {total_enc_min:.1f} minutes (~{total_enc_min/60:.1f} hours, ~{int(total_enc_min)} files)")
        print("=" * 70)
        
        print("\nQuerying NOAA GCS for exact audio file list...")
        seen = set()
        for x in target_encounters:
            s_dt = datetime.strptime(x["utc_start"], "%Y-%m-%d %H:%M:%S")
            e_dt = datetime.strptime(x["utc_end"], "%Y-%m-%d %H:%M:%S")
            enc_files = get_file_list_for_time_range(s_dt, e_dt)
            for item in enc_files:
                if item[0] not in seen:
                    seen.add(item[0])
                    files.append(item)
    else:
        # Default if no arguments passed: prompt user or default to false killer whale
        print("No filter specified. Use --species 33 (or --species 'false killer whale') or --encounter 191.")
        return

    if not files:
        print("No files found matching criteria.")
        return

    if args.max_files > 0:
        files = files[:args.max_files]

    total_bytes = sum(size for _, size in files)
    print(f"\nFound {len(files)} total FLAC files ({total_bytes / (1024**3):.2f} GB).")

    if args.dry_run:
        print("\n[DRY RUN] First 10 files:")
        for name, size in files[:10]:
            print(f"  {os.path.basename(name)} ({size / (1024**2):.1f} MB)")
        print(f"\nDry run complete. Remove --dry-run to start downloading.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nDestination folder: {os.path.abspath(args.output_dir)}")
    print(f"Downloading with {args.threads} parallel threads (skips already downloaded files)...\n")

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(download_file, item, args.output_dir): item for item in files}
        with tqdm(total=len(files), unit="file", desc="Downloading") as pbar:
            for future in as_completed(futures):
                fname, status, size = future.result()
                pbar.update(1)
                if status not in ("downloaded", "skipped"):
                    print(f"\nFailed: {fname} - {status}")

    print(f"\nDone! All files saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
