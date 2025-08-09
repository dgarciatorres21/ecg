import os, sys, argparse
import random
import csv
from helper_functions import find_records
from gen_ecg_image_from_data import run_single_file
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings("ignore")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', type=str, required=True)
    parser.add_argument('-o', '--output_directory', type=str, required=True)
    parser.add_argument('-se', '--seed', type=int, required=False, default = -1)
    parser.add_argument('--num_leads',type=str,default='twelve')
    parser.add_argument('--max_num_images',type=int,default = -1)
    parser.add_argument('--config_file', type=str, default='config.yaml')
    
    parser.add_argument('-r','--resolution',type=int,required=False,default = 200)
    parser.add_argument('--pad_inches',type=int,required=False,default=0)
    parser.add_argument('-ph','--print_header', action="store_true",default=False)
    parser.add_argument('--num_columns',type=int,default = -1)
    parser.add_argument('--full_mode', type=str,default='II')
    parser.add_argument('--mask_unplotted_samples', action="store_true", default=False)
    parser.add_argument('--add_qr_code', action="store_true", default=False)

    parser.add_argument('-l', '--link', type=str, required=False,default='')
    parser.add_argument('-n','--num_words',type=int,required=False,default=5)
    parser.add_argument('--x_offset',dest='x_offset',type=int,default = 30)
    parser.add_argument('--y_offset',dest='y_offset',type=int,default = 30)
    parser.add_argument('--hws',dest='handwriting_size_factor',type=float,default = 0.2)
    
    parser.add_argument('-ca','--crease_angle',type=int,default=90)
    parser.add_argument('-nv','--num_creases_vertically',type=int,default=10)
    parser.add_argument('-nh','--num_creases_horizontally',type=int,default=10)

    parser.add_argument('-rot','--rotate',type=int,default=0)
    parser.add_argument('-noise','--noise',type=int,default=50)
    parser.add_argument('-c','--crop',type=float,default=0.01)
    parser.add_argument('-t','--temperature',type=int,default=40000)

    parser.add_argument('--random_resolution',action="store_true",default=False)
    parser.add_argument('--random_padding',action="store_true",default=False)
    parser.add_argument('--random_grid_color',action="store_true",default=False)
    parser.add_argument('--standard_grid_color', type=int, default=5)
    parser.add_argument('--calibration_pulse',type=float,default=1)
    parser.add_argument('--random_grid_present',type=float,default=1)
    parser.add_argument('--random_print_header',type=float,default=0)
    parser.add_argument('--random_bw',type=float,default=0)
    parser.add_argument('--remove_lead_names',action="store_false",default=True)
    parser.add_argument('--lead_name_bbox',action="store_true",default=False)
    parser.add_argument('--store_config', type=int, nargs='?', const=1, default=0)

    parser.add_argument('--deterministic_offset',action="store_true",default=False)
    parser.add_argument('--deterministic_num_words',action="store_true",default=False)
    parser.add_argument('--deterministic_hw_size',action="store_true",default=False)

    parser.add_argument('--deterministic_angle',action="store_true",default=False)
    parser.add_argument('--deterministic_vertical',action="store_true",default=False)
    parser.add_argument('--deterministic_horizontal',action="store_true",default=False)

    parser.add_argument('--deterministic_rot',action="store_true",default=False)
    parser.add_argument('--deterministic_noise',action="store_true",default=False)
    parser.add_argument('--deterministic_crop',action="store_true",default=False)
    parser.add_argument('--deterministic_temp',action="store_true",default=False)

    parser.add_argument('--fully_random',action='store_true',default=False)
    parser.add_argument('--hw_text',action='store_true',default=False)
    parser.add_argument('--wrinkles',action='store_true',default=False)
    parser.add_argument('--augment',action='store_true',default=False)
    parser.add_argument('--lead_bbox',action='store_true',default=False)

    return parser

def run(args):
        random.seed(args.seed)

        if os.path.isabs(args.input_directory) == False:
            args.input_directory = os.path.normpath(os.path.join(os.getcwd(), args.input_directory))
        if os.path.isabs(args.output_directory) == False:
            original_output_dir = os.path.normpath(os.path.join(os.getcwd(), args.output_directory))
        else:
            original_output_dir = args.output_directory
        
        # if os.path.exists(args.input_directory) == False or os.path.isdir(args.input_directory) == False:
        #     raise Exception("The input directory does not exist, Please re-check the input arguments!")

        if os.path.exists(original_output_dir) == False:
            os.makedirs(original_output_dir)

        i = 0
        # full_header_files, full_recording_files = find_records(args.input_directory, original_output_dir)
        # --- NEW, SMARTER CODE BLOCK ---
        # Initialize the lists that will hold file paths
        full_header_files = []
        full_recording_files = []

        # Check if the provided input path is a directory
        if os.path.isdir(args.input_directory):
            # This is the ORIGINAL behavior. If given a directory, find all records inside.
            print("Input is a directory, finding all records...")
            full_header_files, full_recording_files = find_records(args.input_directory, original_output_dir)

        # Check if the provided input path is a single .dat file
        elif os.path.isfile(args.input_directory) and args.input_directory.endswith('.dat'):
            # This is the NEW behavior for our wrapper script.
            print(f"Input is a single file: {args.input_directory}")
            
            # The recording file is the one we were given. Add it to the list.
            recording_file = args.input_directory
            full_recording_files.append(recording_file)

            # The header file has the same name but with a .hea extension.
            # We can create its path from the recording file's path.
            header_file = os.path.splitext(recording_file)[0] + '.hea'
            full_header_files.append(header_file)
            
        else:
            # If the input is neither a directory nor a .dat file, print an error.
            print(f"Error: Input path '{args.input_directory}' is not a valid directory or .dat file.")
            # The script will then gracefully exit because the file lists are empty.

        
        for header_file_path, recording_file_path in zip(full_header_files, full_recording_files):
            args.input_file = recording_file_path
            args.header_file = header_file_path
            args.start_index = -1
            # Get the base filename without path (e.g., "00001_lr")
            base_filename = os.path.splitext(os.path.basename(recording_file_path))[0]
            
            # Use the main output directory passed from the wrapper.
            # The wrapper is now responsible for creating the subdirectories.
            # We will just save everything flat in the main output folder for simplicity now.
            # The run_generation.py script will ensure the final PNG has the right name.
            args.output_directory = original_output_dir # Use the main output dir
            args.encoding = base_filename
            
            i += run_single_file(args)
            
            if(args.max_num_images != -1 and i >= args.max_num_images):
                break

if __name__=='__main__':
    # path = os.path.join(os.getcwd(), sys.argv[0])
    # parentPath = os.path.dirname(path)
    # os.chdir(parentPath)
    run(get_parser().parse_args(sys.argv[1:]))
