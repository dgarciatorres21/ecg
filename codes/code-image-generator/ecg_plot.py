import os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import AutoMinorLocator
from math import ceil 
from PIL import Image

standard_values = { 
    'y_grid_size': 0.5, 'x_grid_size': 0.2, 'y_grid_inch': 5/25.4, 'x_grid_inch': 5/25.4, 
    'grid_line_width': 0.5, 'lead_name_offset': 0.5, 'lead_fontsize': 11, 'x_gap': 1, 
    'y_gap': 0.5, 'display_factor': 1, 'line_width': 0.75, 'row_height': 8, 
    'dc_offset_length': 0.2, 'lead_length': 2.5, 'width': 11, 'height': 8.5,
    'min_bbox_height_mv': 2.0,
    'bbox_vertical_margin_mv': 0.1
}
standard_major_colors = {'colour5': (1,0,0)}
standard_minor_colors = {'colour5': (0.996,0.8745,0.8588)}

def ecg_plot(ecg, configs, sample_rate, columns, rec_file_name, output_dir, lead_index, full_mode, full_header_file, show_lead_name=True, show_grid=True, bbox=False, json_dict=dict(), lead_length_in_seconds=2.5, show_dc_pulse=True, standard_colours=5, **kwargs):
    
    matplotlib.use("Agg")
    if not ecg: return 

    secs = lead_length_in_seconds
    leads_count = len(lead_index)
    rows = int(ceil(leads_count / columns))
    if 'full'+full_mode in ecg: rows += 1
    
    width, height = standard_values['width'], standard_values['height']
    y_grid_size, x_grid_size = standard_values['y_grid_size'], standard_values['x_grid_size']
    y_grid_inch, x_grid_inch = standard_values['y_grid_inch'], standard_values['x_grid_inch']
    row_height = (height * y_grid_size / y_grid_inch) / (rows + 2)
    x_max = width * x_grid_size / x_grid_inch
    x_gap = np.floor(((x_max - (columns * secs)) / 2) / 0.2) * 0.2
    y_min, y_max = 0, height * y_grid_size / y_grid_inch

    fig, ax = plt.subplots(figsize=(width, height), dpi=kwargs.get('resolution', 100))
    fig.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)
    
    color_major, color_minor, color_line = standard_major_colors[f'colour{standard_colours}'], standard_minor_colors[f'colour{standard_colours}'], (0,0,0)

    ax.set_ylim(y_min, y_max); ax.set_xlim(0, x_max)
    ax.tick_params(axis='both', which='both', length=0)
    
    fig.canvas.draw()
    canvas_width_px, canvas_height_px = fig.canvas.get_width_height()
    json_dict['width'], json_dict['height'] = canvas_width_px, canvas_height_px
    
    step = 1.0/sample_rate
    dc_offset_time = standard_values['dc_offset_length']
    
    total_block_height = rows * row_height
    y_offset_start = total_block_height + row_height / 2
    
    signal_x_shift = dc_offset_time if show_dc_pulse else 0
    
    for i, leadName in enumerate(lead_index):
        if leadName not in ecg: continue
        if "leads" not in json_dict: json_dict["leads"] = []
        current_lead_ds = {"lead_name": leadName}

        col_index = i % columns
        if col_index == 0: y_offset_start -= row_height
        
        y_offset = y_offset_start
        x_offset = col_index * secs
        
        signal_plot_start_x = x_offset + x_gap
        
        signal_with_pulse = []
        if show_dc_pulse and col_index == 0:
            pulse_time = np.arange(0, dc_offset_time, step)
            dc_pulse = np.ones_like(pulse_time); dc_pulse[:2] = 0; dc_pulse[-2:] = 0
            ax.plot(pulse_time + signal_plot_start_x, dc_pulse + y_offset, linewidth=1.5, color=color_line)
            signal_with_pulse.append(dc_pulse)
            
        if show_lead_name:
            ax.text(x_offset + x_gap + 0.1, y_offset - 0.7, leadName, fontsize=standard_values['lead_fontsize'])

        signal = ecg[leadName]
        signal_with_pulse.append(signal)
        signal_time = np.arange(0, len(signal) * step, step)
        ax.plot(signal_time + signal_plot_start_x + signal_x_shift, signal + y_offset, linewidth=0.75, color=color_line)
        
        if bbox:
            cell_x_min = x_offset + x_gap + signal_x_shift
            cell_x_max = cell_x_min + secs

            full_signal_for_lead = np.concatenate(signal_with_pulse)
            signal_min_raw = np.nanmin(full_signal_for_lead); signal_max_raw = np.nanmax(full_signal_for_lead)
            
            max_deviation = max(abs(signal_min_raw), abs(signal_max_raw))
            margin = standard_values.get('bbox_vertical_margin_mv', 0.1)
            half_height = max_deviation + margin
            min_height = standard_values.get('min_bbox_height_mv', 1.0)
            final_height = max(2 * half_height, min_height)
            
            final_y_min_data = y_offset - (final_height / 2)
            final_y_max_data = y_offset + (final_height / 2)
            
            # --- START OF FIX ---
            # Get the plot's valid Y-axis limits from the matplotlib plot.
            plot_y_min, plot_y_max = ax.get_ylim()
            
            # Clamp the calculated box coordinates to stay within the plot's boundaries.
            # This prevents the box from growing too large due to extreme signal values.
            final_y_min_data_clamped = max(plot_y_min, final_y_min_data)
            final_y_max_data_clamped = min(plot_y_max, final_y_max_data)

            # Use the new clamped coordinates to define the bounding box.
            box_points_data = [[cell_x_min, final_y_min_data_clamped], [cell_x_max, final_y_max_data_clamped]]
            # --- END OF FIX ---

            box_points_display = ax.transData.transform(box_points_data); x_min_px, y_min_display = box_points_display[0]; x_max_px, y_max_display = box_points_display[1]
            y_min_image, y_max_image = canvas_height_px - y_max_display, canvas_height_px - y_min_display
            current_lead_ds["lead_bounding_box"] = {0: [int(y_min_image), int(x_min_px)], 1: [int(y_min_image), int(x_max_px)], 2: [int(y_max_image), int(x_max_px)], 3: [int(y_max_image), int(x_min_px)]}
        
        json_dict["leads"].append(current_lead_ds)
        
        if columns > 1 and col_index < columns - 1:
            tick_x = x_offset + x_gap + signal_x_shift + len(signal)*step; tick_y_center = y_offset
            ax.plot([tick_x, tick_x], [tick_y_center - 0.4, tick_y_center + 0.4], linewidth=1.5, color=color_line)

    if 'full'+full_mode in ecg:
        current_lead_ds = {"lead_name": full_mode}
        rhythm_y_offset = y_offset_start - row_height
        if show_lead_name: ax.text(x_gap + 0.1, rhythm_y_offset - 0.7, full_mode, fontsize=standard_values['lead_fontsize'])
        
        signal_with_pulse_rhythm = []
        if show_dc_pulse:
            pulse_time = np.arange(0, dc_offset_time, step)
            dc_pulse = np.ones_like(pulse_time); dc_pulse[:2] = 0; dc_pulse[-2:] = 0
            ax.plot(pulse_time + x_gap, dc_pulse + rhythm_y_offset, linewidth=1.5, color=color_line)
            signal_with_pulse_rhythm.append(dc_pulse)
        
        signal = ecg['full'+full_mode]
        signal_with_pulse_rhythm.append(signal)
        signal_time = np.arange(0, len(signal)*step, step)
        ax.plot(signal_time + x_gap + dc_offset_time, signal + rhythm_y_offset, linewidth=0.75, color=color_line)
        
        if bbox:
            cell_x_min = x_gap + (dc_offset_time if show_dc_pulse else 0)
            signal_duration = len(signal) * step
            cell_x_max = cell_x_min + signal_duration

            full_signal_for_rhythm = np.concatenate(signal_with_pulse_rhythm)
            signal_min_raw = np.nanmin(full_signal_for_rhythm); signal_max_raw = np.nanmax(full_signal_for_rhythm)
            max_deviation = max(abs(signal_min_raw), abs(signal_max_raw))
            margin = standard_values.get('bbox_vertical_margin_mv', 0.1)
            half_height = max_deviation + margin
            min_height = standard_values.get('min_bbox_height_mv', 1.0)
            final_height = max(2 * half_height, min_height)
            final_y_min_data = rhythm_y_offset - (final_height / 2)
            final_y_max_data = rhythm_y_offset + (final_height / 2)
            
            # --- CLAMPING THE BOX COORDINATES (already correct for rhythm strip) ---
            plot_y_min, plot_y_max = ax.get_ylim()
            final_y_min_data_clamped = max(plot_y_min, final_y_min_data)
            final_y_max_data_clamped = min(plot_y_max, final_y_max_data)

            box_points_data = [[cell_x_min, final_y_min_data_clamped], [cell_x_max, final_y_max_data_clamped]]
            box_points_display = ax.transData.transform(box_points_data); x_min_px, y_min_display = box_points_display[0]; x_max_px, y_max_display = box_points_display[1]
            y_min_image, y_max_image = canvas_height_px - y_max_display, canvas_height_px - y_min_display
            current_lead_ds["lead_bounding_box"] = {0: [int(y_min_image), int(x_min_px)], 1: [int(y_min_image), int(x_max_px)], 2: [int(y_max_image), int(x_max_px)], 3: [int(y_max_image), int(x_min_px)]}
        
        json_dict["leads"].append(current_lead_ds)
        
    ax.text(2, 0.5, '25mm/s', fontsize=standard_values['lead_fontsize']); ax.text(4, 0.5, '10mm/mV', fontsize=standard_values['lead_fontsize'])
    
    if show_grid:
        ax.set_xticks(np.arange(0, x_max, x_grid_size)); ax.set_yticks(np.arange(y_min, y_max, y_grid_size))
        ax.minorticks_on(); ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which='major', linestyle='-', linewidth=0.5, color=color_major)
        ax.grid(which='minor', linestyle=':', linewidth=0.4, color=color_minor)

    head, tail = os.path.split(rec_file_name)
    plt.savefig(os.path.join(output_dir, os.path.basename(tail) + '.png'), dpi=kwargs.get('resolution', 100))
    plt.close(fig)

    return 1.0, 1.0