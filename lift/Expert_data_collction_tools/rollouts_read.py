import pickle as pkl
import matplotlib.pyplot as plt
import os
import time

def load_and_display_images_continuously(filepath):
    """
    Loads a .pkl file and continuously displays images from the rollout.
    
    Args:
        filepath (str): The path to the .pkl file.
    """
    try:
        with open(filepath, 'rb') as f:
            rollout = pkl.load(f)

        if 'camera0_obs' not in rollout or not rollout['camera0_obs']:
            print(f"No 'camera0_obs' found or it's empty in {filepath}.")
            return
        if 'camera1_obs' not in rollout or not rollout['camera1_obs']:
            print(f"No 'camera1_obs' found or it's empty in {filepath}.")
            return
            
        # Turn on interactive mode
        plt.ion()
        
        # Set up the figure and subplots
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display the first image to get the image objects
        im0 = ax0.imshow(rollout['camera0_obs'][0])
        ax0.set_title("Robot 0 Camera Image")
        ax0.axis('off')
        
        im1 = ax1.imshow(rollout['camera1_obs'][0])
        ax1.set_title("Robot 1 Camera Image")
        ax1.axis('off')
        
        fig.tight_layout()
        plt.show(block=False)

        # Loop through all images and update the plot
        num_frames = min(len(rollout['camera0_obs']), len(rollout['camera1_obs']))
        for i in range(num_frames):
            # Update the image data
            im0.set_data(rollout['camera0_obs'][i])
            im1.set_data(rollout['camera1_obs'][i])
            
            # Redraw the canvas and pause for a short time
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.01)  # Adjust delay as needed

        # Turn off interactive mode and keep the final plot displayed
        plt.ioff()
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Example Usage ---
directory = "../rollouts/newslower"
filename = "rollout_seed30_mode2.pkl"
file_path = os.path.join(directory, filename)
load_and_display_images_continuously(file_path)