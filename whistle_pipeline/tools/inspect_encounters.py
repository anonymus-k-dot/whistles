import zipfile, xml.etree.ElementTree as ET, datetime, csv

def excel_to_dt(val):
    try:
        return datetime.datetime(1899, 12, 30) + datetime.timedelta(days=float(val))
    except:
        return None

def get_sheet_data(sheet_idx):
    with zipfile.ZipFile('DCLDE2020_DetectionData.xlsx') as z:
        sst = []
        if 'xl/sharedStrings.xml' in z.namelist():
            sst_root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in sst_root.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si'):
                text = ''.join(t.text for t in si.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t') if t.text)
                sst.append(text)
        sheet_root = ET.fromstring(z.read(f'xl/worksheets/sheet{sheet_idx}.xml'))
        rows = []
        for r in sheet_root.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheetData/{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row'):
            row_data = []
            for c in r.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c'):
                t = c.attrib.get('t')
                v = c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
                val = v.text if v is not None else ''
                if t == 's' and val.isdigit():
                    val = sst[int(val)]
                row_data.append(val)
            rows.append(row_data)
        return rows

species_raw = get_sheet_data(9)
species_map = {r[0].strip(): (r[1].strip(), r[2].strip() if len(r)>2 else '') for r in species_raw[1:] if r}

odonto = get_sheet_data(4)
whale_codes = {'46', '59', '33', '61', '65', '32', '31', '36', '49'}

out_records = []
for r in odonto[1:]:
    if len(r) > 12 and r[12].strip() in whale_codes:
        sp_id = r[12].strip()
        common_name, sci_name = species_map.get(sp_id, ('Unknown', ''))
        t_start = excel_to_dt(r[1])
        t_end = excel_to_dt(r[2])
        det_id = r[3]
        if t_start and t_end:
            dur_min = round((t_end - t_start).total_seconds() / 60.0, 1)
            ts_start_str = t_start.strftime("%Y%m%d_%H%M")
            ts_end_str = t_end.strftime("%Y%m%d_%H%M")
            out_records.append({
                'encounter_id': det_id,
                'species_id': sp_id,
                'common_name': common_name,
                'scientific_name': sci_name,
                'utc_start': t_start.strftime('%Y-%m-%d %H:%M:%S'),
                'utc_end': t_end.strftime('%Y-%m-%d %H:%M:%S'),
                'duration_minutes': dur_min,
                'file_start_prefix': f"1705_{ts_start_str}",
                'file_end_prefix': f"1705_{ts_end_str}",
                'gcs_folder': 'gs://noaa-passive-bioacoustic/dclde/2022/dclde_2022_1705/audio/'
            })

with open('whale_encounters_1705.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(out_records[0].keys()))
    writer.writeheader()
    writer.writerows(out_records)

print(f'Saved {len(out_records)} whale encounters to whale_encounters_1705.csv')
