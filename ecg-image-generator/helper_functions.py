import os, sys, argparse, yaml, math
import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from math import ceil 
import wfdb
import re

BIT_NAN_16 = -(2.**15)

def read_config_file(config_file):
    with open(config_file) as f:
        yamlObject = yaml.safe_load(f)
    return dict(yamlObject)

def find_records(folder, output_dir):
    header_files, recording_files = [], []
    for root, _, files in os.walk(folder):
        for file in sorted(files):
            base, ext = os.path.splitext(file)
            if ext in ['.mat', '.dat']:
                header_path = os.path.join(root, base + '.hea')
                if os.path.exists(header_path):
                    recording_files.append(os.path.relpath(os.path.join(root, file), folder))
                    header_files.append(os.path.relpath(header_path, folder))
    if not recording_files:
        raise Exception("No WFDB compatible ECG files found in the input directory.")
    for file in header_files:
        parent_dir = os.path.dirname(file)
        if parent_dir:
            os.makedirs(os.path.join(output_dir, parent_dir), exist_ok=True)
    return header_files, recording_files

def find_files(data_directory):
    header_files, recording_files = [], []
    for f in sorted(os.listdir(data_directory)):
        full_path = os.path.join(data_directory, f)
        root, ext = os.path.splitext(f)
        if ext in ['.mat', '.dat'] and not root.startswith('.'):
            header_file = os.path.join(data_directory, root + '.hea')
            if os.path.isfile(header_file):
                header_files.append(header_file)
                recording_files.append(full_path)
    return header_files, recording_files

def load_header(header_file):
    with open(header_file, 'r') as f:
        header = f.read()
    return re.sub(r'\(\d+\)', '', header)

def load_recording(recording_file, header=None,key='val'):
    rootname, extension = os.path.splitext(recording_file)
    try:
        if extension == '.dat':
            return wfdb.rdrecord(rootname).p_signal
        if extension == '.mat':
            return loadmat(recording_file)[key]
    except Exception as e:
        print(f"WARNING: Failed to process record '{rootname}'. Error: {e}. Skipping.")
        return None

def get_leads(header):
    leads = []
    try:
        lines = header.split('\n')
        num_leads = int(lines[0].split()[1])
        for i in range(1, min(num_leads + 1, len(lines))):
            if lines[i].strip():
                leads.append(lines[i].split()[-1])
    except (ValueError, IndexError):
        pass
    return tuple(leads)

def get_frequency(header):
    try:
        return float(header.split('\n')[0].split()[2].split('/')[0])
    except (ValueError, IndexError):
        return None

def get_adc_gains(header, leads):
    adc_gains = np.zeros(len(leads))
    lead_map = {lead: i for i, lead in enumerate(leads)}
    try:
        lines = header.split('\n')
        num_leads = int(lines[0].split()[1])
        for i in range(1, min(num_leads + 1, len(lines))):
            if lines[i].strip():
                parts = lines[i].split()
                lead_name = parts[-1]
                if lead_name in lead_map:
                    adc_gains[lead_map[lead_name]] = float(parts[2].split('/')[0])
    except (ValueError, IndexError):
        pass
    return adc_gains

def truncate_signal(signal,sampling_rate,length_in_secs):
    return signal[0:int(sampling_rate*length_in_secs)]

def create_signal_dictionary(signal,full_leads):
    return {lead: signal[k] for k, lead in enumerate(full_leads)}

def standardize_leads(full_leads):
    lead_map = {'AVR': 'aVR', 'AVL': 'aVL', 'AVF': 'aVF'}
    return [lead_map.get(l.upper(), l.upper()) for l in full_leads]

def read_leads(leads):
    lead_bbs, text_bbs, labels, startTimeStamps, endTimeStamps, plotted_pixels = [], [], [], [], [], []
    for lead_dict in leads:
        labels.append(lead_dict.get('lead_name', ''))
        startTimeStamps.append(lead_dict.get('start_sample', 0))
        endTimeStamps.append(lead_dict.get('end_sample', 0))
        plotted_pixels.append(lead_dict.get('plotted_pixels', []))
        if "lead_bounding_box" in lead_dict:
            parts = lead_dict["lead_bounding_box"]
            lead_bbs.append([[parts['0'][0], parts['0'][1]], [parts['1'][0], parts['1'][1]], [parts['2'][0], parts['2'][1]], [parts['3'][0], parts['3'][1]]])
        if "text_bounding_box" in lead_dict:
            parts = lead_dict["text_bounding_box"]
            text_bbs.append([[parts['0'][0], parts['0'][1]], [parts['1'][0], parts['1'][1]], [parts['2'][0], parts['2'][1]], [parts['3'][0], parts['3'][1]]])
    return np.array(lead_bbs), np.array(text_bbs), labels, startTimeStamps, endTimeStamps, plotted_pixels

def convert_bounding_boxes_to_dict(lead_bboxes, text_bboxes, labels, startTimeList=None, endTimeList=None, plotted_pixels_dict=None):
    leads_ds = []
    for i, label in enumerate(labels):
        current_lead_ds = {"lead_name": label}
        if i < len(lead_bboxes):
            current_lead_ds["lead_bounding_box"] = {str(j): [round(p[0]), round(p[1])] for j, p in enumerate(lead_bboxes[i])}
        if i < len(text_bboxes):
            current_lead_ds["text_bounding_box"] = {str(j): [round(p[0]), round(p[1])] for j, p in enumerate(text_bboxes[i])}
        if startTimeList: current_lead_ds["start_sample"] = startTimeList[i]
        if endTimeList: current_lead_ds["end_sample"] = endTimeList[i]
        if plotted_pixels_dict: current_lead_ds["plotted_pixels"] = plotted_pixels_dict[i]
        leads_ds.append(current_lead_ds)
    return leads_ds

def write_wfdb_file(ecg_frame, filename, rate, header_file, write_dir, full_mode, mask_unplotted_samples):
    # ... (this function is correct) ...
    full_header = load_header(header_file)
    full_leads = get_leads(full_header)
    header_name, _ = os.path.splitext(header_file)
    header = wfdb.rdheader(header_name)
    rhythm_key = 'full' + full_mode
    samples = len(ecg_frame.get(rhythm_key, ecg_frame.get(full_leads[0], []))) if full_leads else 0
    recording_data = np.full((samples, len(full_leads)), BIT_NAN_16, dtype=np.int16)
    for i, lead in enumerate(full_leads):
        signal = np.array(ecg_frame.get(lead, []), dtype=float)
        signal[np.isnan(signal)] = BIT_NAN_16 / header.adc_gain[i]
        recording_data[:len(signal), i] = (signal * header.adc_gain[i]).astype(np.int16)
    wfdb.wrsamp(record_name=os.path.splitext(filename)[0], fs=rate, units=header.units,
                sig_name=list(full_leads), p_signal=recording_data, fmt=header.fmt,
                adc_gain=header.adc_gain, baseline=header.baseline, write_dir=write_dir)

def get_lead_pixel_coordinate(leads):
    pixel_coordinates = dict()
    for lead_info in leads:
        leadName = lead_info["lead_name"]
        plotted_pixels = np.array(lead_info["plotted_pixels"])
        pixel_coordinates[leadName] = plotted_pixels
    return pixel_coordinates


def rotate_points(pixel_coordinates, origin, angle):
    """
    Rotates lists of points. This version is robust and handles empty lists.
    """
    rotates_pixel_coords = []
    angle = math.radians(angle)
    ox, oy = origin
    
    transformation = np.array([[math.cos(angle), -math.sin(angle)],
                               [math.sin(angle), math.cos(angle)]])
    origin_point = np.array([ox, oy])

    # The input `pixel_coordinates` is a list of lists (one list of points for each lead)
    for pixels_array in pixel_coordinates:
        # If the list of points for a lead is empty, don't try to rotate it. Just append an empty list to the results and continue.
        if len(pixels_array) == 0:
            rotates_pixel_coords.append([])
            continue
        
        # Original logic is now safe to run
        translated_points = np.array(pixels_array) - origin_point
        rotated_points = translated_points @ transformation.T + origin_point
        rotates_pixel_coords.append(np.round(rotated_points, 2).tolist())
        
    return rotates_pixel_coords

def rotate_bounding_box(box, origin, angle):
    # ... (this function is correct) ...
    angle = math.radians(angle)
    ox, oy = origin
    transformation = np.array([[math.cos(angle), -math.sin(angle)],
                               [math.sin(angle), math.cos(angle)]])
    origin_point = np.array([ox, oy])
    translated_box = np.array(box) - origin_point
    rotated_box = translated_box @ transformation.T + origin_point
    return rotated_box