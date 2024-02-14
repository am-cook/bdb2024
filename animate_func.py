import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from PIL import Image
import re

# read in paths
with open('SETTINGS.json') as f:
    settings = json.load(f)

agg_data_path = settings['AGGREGATED_DATA']
external_data_path = settings['EXTERNAL_RAW_DATA_DIR']
gifs_dir = settings['OUTPUT_GIFS']
final_destination = f'{gifs_dir}plays/'

tracking = pd.read_csv(f'{agg_data_path}tracking.csv').reset_index(drop = True)
plays = pd.read_csv(f'{agg_data_path}plays.csv').reset_index(drop = True)

def create_gif(gameplay, tracking = tracking, plays = plays, playersize = 'large', scale = False, weak_dlineman = None, weak_backer = None):

    def create_frames(tracking_df, plays_df, gameplayId, tempdir_path):

        # merge in team_colors to tracking_df
        # NFL team hex codes were manually curated from https://teamcolorcodes.com/nfl-team-color-codes/
        team_colors = pd.read_csv(f'{external_data_path}cleaned_team_color_df.csv', index_col=None)
        team_colors_dict = dict(zip(team_colors['abbrev'].values, team_colors['color'].values))

        one_play = tracking_df.loc[tracking_df['gameplayId'] == gameplayId].reset_index(drop = True)

        # add in ball carrier info to plotting dataframe
        bc = plays_df['ballCarrierDisplayName'].loc[plays_df['gameplayId'] == gameplayId].values[0]

        # specifying marker shapes: https://seaborn.pydata.org/tutorial/properties.html
        plot_shapes_dict = {'default':'o',
                            'bc':'p',
                            'weak_dlineman':'s',
                            'weak_backer':'D'}
        
        plot_shapes = ['default' for _ in range(one_play.shape[0])] # initialize default plot shapes
        
        for rownum, row in one_play.iterrows():
            if row['displayName'] == bc:
                plot_shapes[rownum] = 'bc' # ball carrier is a hexagon
            elif row['displayName'] == weak_dlineman:
                plot_shapes[rownum] = 'weak_dlineman' # weakside d-lineman is triangle
            elif row['displayName'] == weak_backer:
                plot_shapes[rownum] = 'weak_backer' # weakside backer is diamond

        one_play.insert(loc = one_play.shape[1], value = plot_shapes, column = 'plot_shapes')

        # plot_field.png was made using PowerPoint. Go Irish
        plot_background = plt.imread(f'{external_data_path}plot_field.png')

        if not os.path.exists(tempdir_path):
            os.mkdir(tempdir_path)

        los, yards_to_go = plays_df[['absoluteYardlineNumber', 'yardsToGo']].loc[plays_df['gameplayId'] == gameplayId].values[0] # get line of scrimmage for play

        play_dir = one_play['playDirection'].values[0]
        if play_dir == 'left':
            first_down_line = los - yards_to_go
        elif play_dir == 'right':
            first_down_line = los + yards_to_go



        for frame in one_play['frameId']:

            fig, ax = plt.subplots()

            if scale:
                ax.set(xlim = (one_play['x'].min()-2, one_play['x'].max()+2), ylim = (0, 53.3))
            else:
                ax.set(xlim = (0, 120), ylim = (0, 53.3))

            
            this_frame_df = one_play.loc[one_play['frameId'] == frame]

            if playersize == 'large':
                circle_sizes = this_frame_df['jerseyNumber'].apply(lambda x: 200 if not np.isnan(x) else 100) # scale football size down
                
            elif playersize == 'small':
                circle_sizes = this_frame_df['jerseyNumber'].apply(lambda x: 110 if not np.isnan(x) else 55) # scale football size down
                
            this_frame_df.insert(loc = this_frame_df.shape[1], column = 'plot_size', value = circle_sizes)
            
            ax.imshow(plot_background, extent = [0, 120, 0, 53.3])
            ax.axvline(x = los, ymin = 0, ymax = 53.3, color = 'lightblue', linewidth = 1.5, linestyle = '--', zorder = 0)
            ax.axvline(x = first_down_line, ymin = 0, ymax = 53.3, color = 'yellow', linewidth = 1.5, linestyle = '--', zorder = 0)
            ax.set_xticks([])
            ax.set_yticks([])

            if playersize == 'large':
                ax = sns.scatterplot(data = this_frame_df,
                                x = 'x',
                                y = 'y',
                                hue = 'club',
                                size = 'plot_size',
                                palette = team_colors_dict,
                                style = 'plot_shapes',
                                markers = plot_shapes_dict,
                                )
            elif playersize == 'small':
                ax = sns.scatterplot(data = this_frame_df,
                                x = 'x',
                                y = 'y',
                                hue = 'club',
                                size = 'plot_size',
                                palette = team_colors_dict,
                                style = 'plot_shapes',
                                markers = plot_shapes_dict,
                                )
                
            
            # label each point with respective player number
            for rownum, vals in this_frame_df.iterrows():
                converted_rgb = tuple(int(team_colors_dict[vals['club']].lstrip('#')[pos:pos+2], 16) for pos in (0, 2, 4)) # convert hex to rgb
                luma = np.dot(converted_rgb, [0.2126, 0.7152, 0.0722]) # luma standardizes rgbs to determine brightness
                font_color = '#ECECEC' if luma < 60 else '#000000'
        
                handles, labels = ax.get_legend_handles_labels()
                
                ax.legend(handles[1:3], labels[1:3])
                if playersize == 'large':
                    ax.text(vals['x'], vals['y'], '' if np.isnan(vals['jerseyNumber']) else str(int(vals['jerseyNumber'])),
                        fontsize = 6, color = font_color, weight = 'bold', horizontalalignment = 'center',
                        verticalalignment = 'center')
                elif playersize == 'small':
                    ax.text(vals['x'], vals['y'], '' if np.isnan(vals['jerseyNumber']) else str(int(vals['jerseyNumber'])),
                        fontsize = 5, color = font_color, weight = 'bold', horizontalalignment = 'center',
                        verticalalignment = 'center')

            title_string = plays['playDescription'].loc[plays['gameplayId'] == gameplay].values[0]
            subtitle_string = f'WSDL: {weak_dlineman}; WSLB: {weak_backer}'

            plt.suptitle(title_string, y = 0.08, fontsize = 8)
            plt.title(subtitle_string, fontsize = 10)

            plot_savename = f'gameplay{gameplayId}_frame{frame}.png'

            plt.savefig(os.path.join(tempdir_path, plot_savename), dpi = 200)

            plt.close()

            # manual override to exit if frame is the max frame from that gameplay
            if frame == one_play['frameId'].max():
                return
        

    
    pics_folder = f'{gifs_dir}temp_images/'
    
    create_frames(gameplayId = gameplay, tracking_df= tracking, plays_df = plays, tempdir_path= pics_folder)

    gif_savename = f'gameplay{gameplay}.gif'

    file_paths = glob.glob(f'{pics_folder}/*.png')
    frame_nums = [int(re.findall(pattern = '.*frame(\\d+).png', string = path)[0]) for path in file_paths]
    path_frame_dict = dict(zip(file_paths, frame_nums))
    path_frame_dict = {k:v for k, v in sorted(path_frame_dict.items(), key = lambda x: x[1])}
    sorted_paths = list(path_frame_dict.keys())

    pil_images = [Image.open(im) for im in sorted_paths]
    first_frame = pil_images[0]
    first_frame.save(os.path.join(final_destination, gif_savename), format = 'GIF', append_images = pil_images,
                     save_all = True, duration = 100, loop = 1)
    
    for img in file_paths:
        os.remove(img)
    