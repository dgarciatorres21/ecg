import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import AutoMinorLocator
from math import ceil 
from PIL import Image

def ecg_plot(ecg, configs, sample_rate, columns, rec_file_name, output_dir, lead_index, full_mode, full_header_file, show_lead_name=True, show_grid=True, bbox=False, json_dict=dict(), lead_length_in_seconds=2.5, show_dc_pulse=True, standard_colours=5, generate_mask_mode=False, **kwargs):
    """
    Generates a realistic or mask ECG image based on parameters from the config file.
    All layout and color settings are pulled from the `configs` dictionary.
    """
    matplotlib.use("Agg")
    if not ecg: return 

    # --- CONFIGURATION LOADING ---
    # Load all layout and color parameters from the config dictionary with safe defaults.
    layout_params = configs.get('layout_parameters', {})
    color_configs = configs.get('colors', {})

    # Paper and Grid dimensions
    width = layout_params.get('width', 11.0)
    height = layout_params.get('height', 8.5)
    y_grid_size = layout_params.get('y_grid_size', 0.5)
    x_grid_size = layout_params.get('x_grid_size', 0.2)
    y_grid_inch = layout_params.get('y_grid_inch', 5/25.4)
    x_grid_inch = layout_params.get('x_grid_inch', 5/25.4)
    
    # Font and Line properties
    lead_fontsize = layout_params.get('lead_fontsize', 11)
    line_width = layout_params.get('line_width', 0.75)
    
    # Bounding Box parameters
    min_bbox_height_mv = layout_params.get('min_bbox_height_mv', 2.0)
    bbox_vertical_margin_mv = layout_params.get('bbox_vertical_margin_mv', 0.1)

    # --- LAYOUT CALCULATION ---
    secs = lead_length_in_seconds
    leads_count = len(lead_index)
    rows = int(ceil(leads_count / columns))
    if 'full' + full_mode in ecg: rows += 1
    
    row_height = (height * y_grid_size / y_grid_inch) / (rows + 2)
    x_max = width * x_grid_size / x_grid_inch
    x_gap = np.floor(((x_max - (columns * secs)) / 2) / 0.2) * 0.2
    y_min, y_max = 0, height * y_grid_size / y_grid_inch

    fig, ax = plt.subplots(figsize=(width, height), dpi=kwargs.get('resolution', 100))
    fig.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)
    
    # --- MASK MODE LOGIC ---
    if generate_mask_mode:
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        color_line = (1, 1, 1) # White
        show_grid = False
        show_lead_name = False
        show_dc_pulse = False
        bbox = False
    else:
        # Load colors from the config for realistic mode
        color_sets = color_configs.get('standard_sets', {})
        selected_color_set = color_sets.get(f'colour{standard_colours}', {})
        color_major = tuple(selected_color_set.get('major', (1,0,0))) # Default red
        color_minor = tuple(selected_color_set.get('minor', (0.996,0.8745,0.8588))) # Default pink
        color_line = (0,0,0) # Black

    ax.set_ylim(y_min, y_max); ax.set_xlim(0, x_max)
    ax.tick_params(axis='both', which='both', length=0)
    
    fig.canvas.draw()
    canvas_width_px, canvas_height_px = fig.canvas.get_width_height()
    
    if not generate_mask_mode:
        json_dict['width'], json_dict['height'] = canvas_width_px, canvas_height_px
    
    step = 1.0/sample_rate
    dc_offset_time = layout_params.get('dc_offset_length', 0.2)
    
    total_block_height = rows * row_height
    y_offset_start = total_block_height + row_height / 2
    
    signal_x_shift = dc_offset_time if show_dc_pulse else 0
    
    for i, leadName in enumerate(lead_index):
        if leadName not in ecg: continue
        
        signal_with_pulse = []

        if not generate_mask_mode:
            if "leads" not in json_dict: json_dict["leads"] = []
            current_lead_ds = {"lead_name": leadName}

        col_index = i % columns
        if col_index == 0: y_offset_start -= row_height
        
        y_offset = y_offset_start
        x_offset = col_index * secs
        
        signal_plot_start_x = x_offset + x_gap
        
        if show_dc_pulse and col_index == 0:
            pulse_time = np.arange(0, dc_offset_time, step)
            dc_pulse = np.ones_like(pulse_time); dc_pulse[:2] = 0; dc_pulse[-2:] = 0
            ax.plot(pulse_time + signal_plot_start_x, dc_pulse + y_offset, linewidth=1.5, color=color_line)
            signal_with_pulse.append(dc_pulse)
            
        if show_lead_name:
            ax.text(x_offset + x_gap + 0.1, y_offset - 0.7, leadName, fontsize=lead_fontsize)

        signal = ecg[leadName]
        signal_with_pulse.append(signal)
        signal_time = np.arange(0, len(signal) * step, step)
        ax.plot(signal_time + signal_plot_start_x + signal_x_shift, signal + y_offset, linewidth=line_width, color=color_line)
        
        if bbox:
            cell_x_min = x_offset + x_gap + signal_x_shift
            cell_x_max = cell_x_min + secs
            full_signal_for_lead = np.concatenate(signal_with_pulse)
            
            signal_min_raw = np.nanmin(full_signal_for_lead); signal_max_raw = np.nanmax(full_signal_for_lead)
            max_deviation = max(abs(signal_min_raw), abs(signal_max_raw))
            
            half_height = max_deviation + bbox_vertical_margin_mv
            final_height = max(2 * half_height, min_bbox_height_mv)
            
            final_y_min_data = y_offset - (final_height / 2)
            final_y_max_data = y_offset + (final_height / 2)
            
            plot_y_min, plot_y_max = ax.get_ylim()
            final_y_min_data_clamped = max(plot_y_min, final_y_min_data)
            final_y_max_data_clamped = min(plot_y_max, final_y_max_data)
            
            box_points_data = [[cell_x_min, final_y_min_data_clamped], [cell_x_max, final_y_max_data_clamped]]
            box_points_display = ax.transData.transform(box_points_data)
            x_min_px, y_min_display = box_points_display[0]
            x_max_px, y_max_display = box_points_display[1]
            
            y_min_image, y_max_image = canvas_height_px - y_max_display, canvas_height_px - y_min_display
            current_lead_ds["lead_bounding_box"] = {0: [int(y_min_image), int(x_min_px)], 1: [int(y_min_image), int(x_max_px)], 2: [int(y_max_image), int(x_max_px)], 3: [int(y_max_image), int(x_min_px)]}
        
        if not generate_mask_mode:
            json_dict["leads"].append(current_lead_ds)
        
        if columns > 1 and col_index < columns - 1:
            tick_x = x_offset + x_gap + signal_x_shift + len(signal)*step; tick_y_center = y_offset
            ax.plot([tick_x, tick_x], [tick_y_center - 0.4, tick_y_center + 0.4], linewidth=1.5, color=color_line)

    if 'full' + full_mode in ecg:
        if not generate_mask_mode:
            current_lead_ds = {"lead_name": full_mode}
        
        signal_with_pulse_rhythm = []
        rhythm_y_offset = y_offset_start - row_height
        if show_lead_name: ax.text(x_gap + 0.1, rhythm_y_offset - 0.7, full_mode, fontsize=lead_fontsize)
        
        if show_dc_pulse:
            pulse_time = np.arange(0, dc_offset_time, step)
            dc_pulse = np.ones_like(pulse_time); dc_pulse[:2] = 0; dc_pulse[-2:] = 0
            ax.plot(pulse_time + x_gap, dc_pulse + rhythm_y_offset, linewidth=1.5, color=color_line)
            signal_with_pulse_rhythm.append(dc_pulse)
        
        signal = ecg['full' + full_mode]
        signal_with_pulse_rhythm.append(signal)
        signal_time = np.arange(0, len(signal)*step, step)
        ax.plot(signal_time + x_gap + signal_x_shift, signal + rhythm_y_offset, linewidth=line_width, color=color_line)
        
        if bbox:
            cell_x_min = x_gap + (dc_offset_time if show_dc_pulse else 0)
            signal_duration = len(signal) * step
            cell_x_max = cell_x_min + signal_duration
            
            full_signal_for_rhythm = np.concatenate(signal_with_pulse_rhythm)
            
            signal_min_raw = np.nanmin(full_signal_for_rhythm); signal_max_raw = np.nanmax(full_signal_for_rhythm)
            max_deviation = max(abs(signal_min_raw), abs(signal_max_raw))

            half_height = max_deviation + bbox_vertical_margin_mv
            final_height = max(2 * half_height, min_bbox_height_mv)

            final_y_min_data = rhythm_y_offset - (final_height / 2)
            final_y_max_data = rhythm_y_offset + (final_height / 2)

            plot_y_min, plot_y_max = ax.get_ylim()
            final_y_min_data_clamped = max(plot_y_min, final_y_min_data)
            final_y_max_data_clamped = min(plot_y_max, final_y_max_data)

            box_points_data = [[cell_x_min, final_y_min_data_clamped], [cell_x_max, final_y_max_data_clamped]]
            box_points_display = ax.transData.transform(box_points_data)
            x_min_px, y_min_display = box_points_display[0]
            x_max_px, y_max_display = box_points_display[1]
            
            y_min_image, y_max_image = canvas_height_px - y_max_display, canvas_height_px - y_min_display
            current_lead_ds["lead_bounding_box"] = {0: [int(y_min_image), int(x_min_px)], 1: [int(y_min_image), int(x_max_px)], 2: [int(y_max_image), int(x_max_px)], 3: [int(y_max_image), int(x_min_px)]}
        
        if not generate_mask_mode:
            json_dict["leads"].append(current_lead_ds)
            
    if not generate_mask_mode:
        ax.text(2, 0.5, '25mm/s', fontsize=lead_fontsize)
        ax.text(4, 0.5, '10mm/mV', fontsize=lead_fontsize)
    
    if show_grid:
        ax.set_xticks(np.arange(0, x_max, x_grid_size))
        ax.set_yticks(np.arange(y_min, y_max, y_grid_size))
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which='major', linestyle='-', linewidth=layout_params.get('grid_line_width', 0.5), color=color_major)
        ax.grid(which='minor', linestyle=':', linewidth=0.4, color=color_minor)

    head, tail = os.path.split(rec_file_name)
    plt.savefig(os.path.join(output_dir, os.path.basename(tail) + '.png'), dpi=kwargs.get('resolution', 100))
    plt.close(fig)

    return 1.0, 1.0