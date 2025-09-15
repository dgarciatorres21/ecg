import os, sys, argparse
import json
import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from TemplateFiles.generate_template import generate_template
from math import ceil 
from helper_functions import get_adc_gains,get_frequency,get_leads,load_recording,load_header,find_files, truncate_signal, create_signal_dictionary, standardize_leads, write_wfdb_file
from ecg_plot import ecg_plot
import wfdb
from PIL import Image, ImageDraw, ImageFont
from random import randint
import random

def get_paper_ecg(input_file,header_file,output_directory, seed, add_dc_pulse,add_bw,show_grid, add_print, configs, mask_unplotted_samples = False, start_index = -1, store_configs=False, store_text_bbox=True,key='val',resolution=100,units='inches',papersize='',add_lead_names=True,pad_inches=1,template_file=os.path.join('TemplateFiles','TextFile1.txt'),font_type=os.path.join('Fonts','Times_New_Roman.ttf'),standard_colours=5,full_mode='II',bbox = False, generate_masks=False, mask_output_directory=None, columns=-1):

    # Extract a reduced-lead set from each pair of full-lead header and recording files.
    full_header_file = header_file
    full_recording_file = input_file
    full_header = load_header(full_header_file)
    full_leads = get_leads(full_header)
    num_full_leads = len(full_leads)

    # Update the header file
    full_lines = full_header.split('\n')

    # For the first line, update the number of leads.
    entries = full_lines[0].split()

    head, tail = os.path.split(full_header_file)

    output_header_file = os.path.join(output_directory, tail)
    with open(output_header_file, 'w') as f:
            f.write('\n'.join(full_lines))

    #Load the full-lead recording file, extract the lead data, and save the reduced-lead recording file.
    recording = load_recording(full_recording_file, full_header,key)

    if recording is None:
        # We return an empty list to signal that no images were generated for this file.
        print(f"Skipping processing for {full_recording_file} due to loading error.")
        return []

    # Get values from header
    rate = get_frequency(full_header)
    adc = get_adc_gains(full_header,full_leads)
    
    full_leads = standardize_leads(full_leads)
    
    # Read the number of columns from the config file, with a default of 4.
    columns = configs.get('columns', 4) if columns == -1 else columns

    # If a desired_order is specified in the config, use it. This is crucial for layout.
    if 'desired_order' in configs:
        desired_order_from_config = configs['desired_order']
        # Filter to only leads present in the recording, maintaining the desired order.
        full_leads = [lead for lead in desired_order_from_config if lead in full_leads]

    # This part remains mostly the same, just simplified
    if(len(full_leads)==12):
        gen_m = 12
        if full_mode not in full_leads:
            full_mode = full_leads[0]
    elif(len(full_leads)==2):
        full_mode = 'None'
        gen_m = 2
        columns = configs.get('columns', 1) if columns == -1 else columns # Adjust for 2-lead
    else:
        gen_m = len(full_leads)
        full_mode = 'None'

    template_name = 'custom_template.png'

    if(recording.shape[0] != num_full_leads):
        recording = np.transpose(recording)
    
    rhythm_strip_len_seconds = configs.get('rhythm_strip_len_seconds', 10.0)
    rhythm_strip_samples = int(rate * rhythm_strip_len_seconds)

    record_dict = create_signal_dictionary(recording,full_leads)
   
    gain_index = 0

    ecg_frame = []
    end_flag = False
    start = 0
    lead_length_in_seconds = configs['paper_len']/columns
    abs_lead_step = configs['abs_lead_step']
    
    segmented_ecg_data = {}

    if start_index != -1:
        start = start_index
        frame = {}
        
        segment_duration_samples = int(rate * lead_length_in_seconds)
        
        for i, key in enumerate(full_leads):
            row_index = i // columns
            data_start_sample = start + (row_index * segment_duration_samples)
            data_end_sample = data_start_sample + segment_duration_samples
            
            if len(record_dict[key]) >= data_end_sample:
                frame[key] = record_dict[key][data_start_sample:data_end_sample]
            else:
                frame[key] = record_dict[key][data_start_sample:] if len(record_dict[key]) > data_start_sample else np.array([])
                end_flag = True

        if full_mode != 'None' and full_mode in record_dict:
            rhythm_start = start
            rhythm_end = rhythm_start + rhythm_strip_samples
            if len(record_dict[full_mode]) >= rhythm_end:
                frame['full' + full_mode] = record_dict[full_mode][rhythm_start:rhythm_end]
            else:
                frame['full' + full_mode] = record_dict[full_mode][rhythm_start:]
                
        ecg_frame.append(frame)

    else:
        while(end_flag==False):
            frame = {}
            
            segment_duration_samples = int(rate * lead_length_in_seconds)
            max_rows = ceil(len(full_leads) / columns)
            total_duration_needed = start + (max_rows * segment_duration_samples)

            for key in full_leads:
                if len(record_dict[key]) < total_duration_needed:
                    end_flag = True
                    break
            if end_flag:
                continue

            for i, key in enumerate(full_leads):
                row_index = i // columns
                data_start_sample = start + (row_index * segment_duration_samples)
                data_end_sample = data_start_sample + segment_duration_samples
                frame[key] = record_dict[key][data_start_sample:data_end_sample]

            if full_mode != 'None' and full_mode in record_dict:
                rhythm_start = start
                rhythm_end = rhythm_start + rhythm_strip_samples
                if len(record_dict[full_mode]) >= rhythm_end:
                    frame['full' + full_mode] = record_dict[full_mode][rhythm_start:rhythm_end]
                else:
                    end_flag = True
                    continue

            ecg_frame.append(frame)
            start = start + int(rate*abs_lead_step)

    outfile_array = []
    
    name, ext = os.path.splitext(full_header_file)

    if len(ecg_frame) == 0:
        return outfile_array

    start = 0
    for i in range(len(ecg_frame)):
        dc = add_dc_pulse.rvs()
        bw = add_bw.rvs()
        grid = show_grid.rvs()
        print_txt = add_print.rvs()

        json_dict = {}
        json_dict['sampling_frequency'] = rate
        grid_colour = 'colour'
        if(bw):
            grid_colour = 'bw'

        rec_file = name + '-' + str(i)
        if ecg_frame[i] == {}:
            continue

        # This generates the realistic image (X) and saves it to 'output_directory'
        x_grid,y_grid = ecg_plot(ecg_frame[i], configs=configs, full_header_file=full_header_file, sample_rate = rate,columns=columns,rec_file_name = rec_file, output_dir = output_directory, resolution = resolution, pad_inches = pad_inches, lead_index=full_leads, full_mode = full_mode, show_lead_name=add_lead_names,show_dc_pulse=dc,papersize=papersize,show_grid=(grid),standard_colours=standard_colours,bbox=bbox, json_dict=json_dict, lead_length_in_seconds=lead_length_in_seconds, generate_mask_mode=False)
        
        if generate_masks:
            # If a specific mask directory is provided, use it.
            # Otherwise, fall back to the old behavior for safety.
            if mask_output_directory:
                mask_output_dir = mask_output_directory
            else:
                # Fallback behavior: create a 'masks' subfolder in the parent directory
                parent_directory = os.path.dirname(output_directory)
                mask_output_dir = os.path.join(parent_directory, 'masks')
            
            os.makedirs(mask_output_dir, exist_ok=True)
            
            # The mask should have the same base name as the realistic image
            mask_rec_file = os.path.join(mask_output_dir, os.path.basename(rec_file))
            
            # Call ecg_plot a second time, but in MASK MODE
            ecg_plot(
                ecg=ecg_frame[i], 
                configs=configs, 
                sample_rate=rate,
                columns=columns,
                rec_file_name=mask_rec_file,        
                output_dir=mask_output_dir,         
                resolution=resolution, 
                pad_inches=pad_inches, 
                lead_index=full_leads, 
                full_mode=full_mode,
                full_header_file=full_header_file,
                lead_length_in_seconds=lead_length_in_seconds,
                generate_mask_mode=True             # Key that activates mask mode
            )

        rec_head, rec_tail = os.path.split(rec_file)
        
        json_dict["x_grid"] = round(x_grid, 3)
        json_dict["y_grid"] = round(y_grid, 3)
        json_dict["resolution"] =resolution
        json_dict["pad_inches"] = pad_inches

        if store_configs == 2:
            json_dict["dc_pulse"] = bool(dc)
            json_dict["bw"] = bool(bw)
            json_dict["gridlines"] = bool(grid)
            json_dict["printed_text"] = bool(print_txt)
            json_dict["number_of_columns_in_image"] = columns
            json_dict["full_mode_lead"] =full_mode

        outfile = os.path.join(output_directory,rec_tail+'.png')

        json_object = json.dumps(json_dict, indent=4)

        if store_configs:
            with open(os.path.join(output_directory,rec_tail+'.json'), "w") as f:
                f.write(json_object)

        outfile_array.append(outfile)
        start  += int(rate*abs_lead_step)
        
    return outfile_array