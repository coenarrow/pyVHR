from __future__ import print_function
import matplotlib.pyplot as plt
import PIL
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from IPython.display import display, clear_output
import cv2
import mediapipe as mp
import numpy as np
import pyVHR
from pyVHR.extraction.sig_processing import extract_frames_yield
from scipy.signal import welch
import random
from matplotlib.colors import Normalize
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


"""
This module defines classes or methods used for plotting outputs.
"""

class VisualizeParams:
    """
    This class contains usefull parameters used by this module.

    The "renderer" variable is used for rendering plots on Jupyter notebook ('notebook')
    or Colab notebook ('colab').
    """
    renderer = 'notebook'  # or 'colab'

def interactive_image_plot(images_list, scaling=1):
    """
    This method create an interactive plot of a list of images. This method must be called
    inside a Jupyter notebook or a Colab notebook.

    Args:
        images_list (list of ndarray): list of images as ndarray with shape [rows, columns, rgb_channels].
        scaling (float): scale factor useful for enlarging or decreasing the resolution of the image.
    
    """
    if images_list is None or len(images_list) == 0:
        return
    
    PIL_image = PIL.Image.fromarray(np.uint8(images_list[0]))
    width, height = PIL_image.size
    print(f'Frame Shape (W,H) = ({width, height})')

    def f(x):
        PIL_image = PIL.Image.fromarray(np.uint8(images_list[x]))
        width, height = PIL_image.size
        PIL_image = PIL_image.resize((int(width*scaling), int(height*scaling)), PIL.Image.NEAREST)
        display(PIL_image)
    
    # show interactively
    interact(f, x=widgets.IntSlider(min=0, max=len(images_list)-1, step=1, value=0));

def display_video(video_file_name, scaling=1):
    """
    This method creates an interactive plot for visualizing the frames of a video.
    It must be called inside a Jupyter or a Colab notebook.

    Args:
        video_file_name (str): video file name or path.
        scaling (float): scale factor for changing the size of the image.
    
    """
    original_frames = [cv2.cvtColor(_, cv2.COLOR_BGR2RGB)
                       for _ in extract_frames_yield(video_file_name)]
    interactive_image_plot(original_frames, scaling)

def visualize_windowed_sig(windowed_sig, window):
    """
    This method creates a plotly plot for visualizing a window of a windowed signal. This method must be called
    inside a Jupyter notebook or a Colab notebook.

    Args:
        windowed_sig (list of float32 ndarray): windowed signal as a list of length num_windows of float32 ndarray with shape [num_estimators, rgb_channels, window_frames].
        window (int): the index of the window to plot.
    
    """
    fig = go.Figure()
    i = 1
    for e in windowed_sig[window]:
        name = "sig_" + str(i) + "_r"
        r_color = random.randint(1, 255)
        g_color = random.randint(1, 255)
        b_color = random.randint(1, 255)
        fig.add_trace(go.Scatter(x=np.arange(e.shape[-1]), y=e[0, :],
                                 mode='lines', marker_color='rgba('+str(r_color)+', '+str(g_color)+', '+str(b_color)+', 1.0)', name=name))
        name = "sig_" + str(i) + "_g"
        fig.add_trace(go.Scatter(x=np.arange(e.shape[-1]), y=e[1, :],
                                 mode='lines', marker_color='rgba('+str(r_color)+', '+str(g_color)+', '+str(b_color)+', 1.0)', name=name))
        name = "sig_" + str(i) + "_b"
        fig.add_trace(go.Scatter(x=np.arange(e.shape[-1]), y=e[2, :],
                                 mode='lines', marker_color='rgba('+str(r_color)+', '+str(g_color)+', '+str(b_color)+', 1.0)', name=name))
        i += 1
    fig.update_layout(title="WIN #" + str(window))
    fig.show(renderer=VisualizeParams.renderer)

def visualize_BVPs(BVPs, window):
    """
    This method create a plotly plot for visualizing a window of a windowed BVP signal. This method must be called
    inside a Jupyter notebook or a Colab notebook.

    Args:
        BVPs (list of float32 ndarray): windowed BPM signal as a list of length num_windows of float32 ndarray with shape [num_estimators, window_frames].
        window (int): the index of the window to plot.
    
    """
    fig = go.Figure()
    i = 1
    bvp = BVPs[window]
    for e in bvp:
        name = "BVP_" + str(i)
        fig.add_trace(go.Scatter(x=np.arange(bvp.shape[1]), y=e[:],
                                 mode='lines', name=name))
        i += 1
    fig.update_layout(title="BVP #" + str(window))
    fig.show(renderer=VisualizeParams.renderer)

def visualize_multi_est_BPM_vs_BPMs_list(multi_est_BPM, BPMs_list):
    """
    This method create a plotly plot for visualizing a multi-estimator BPM signal and a list of BPM signals. 
    This is usefull when comparing Patches BPMs vs Holistic and Ground Truth BPMs. This method must be called
    inside a Jupyter notebook or a Colab notebook.

    Args:
        multi_est_BPM (list): multi-estimator BPM signal is a list that contains two elements [mul-est-BPM, times]; the first contains BPMs as a list of 
            length num_windows of float32 ndarray with shape [num_estimators, ], the second is a float32 1D ndarray that contains the time in seconds of each BPM.
        BPMs_list (list): The BPM signals is a 2D list structured as [[BPM_list, times, name_tag], ...]. The first element is a float32 ndarray that
            contains the BPM signal, the second element is a float32 1D ndarray that contains the time in seconds of each BPM, the third element is a string
            that is used in the plot's legend.    
    """
    fig = go.Figure()
    for idx, _ in enumerate(BPMs_list):
        name = str(BPMs_list[idx][2])
        fig.add_trace(go.Scatter(x=BPMs_list[idx][1], y=BPMs_list[idx][0], mode='lines', name=name))
    for w, _ in enumerate(multi_est_BPM[0]):
        name = "BPMs_" + str(w+1)
        data = multi_est_BPM[0][w]
        if data.shape == ():
            t = [multi_est_BPM[1][w], ]
            data = [multi_est_BPM[0][w], ]
        else:
            t = multi_est_BPM[1][w] * np.ones(data.shape[0])
        fig.add_trace(go.Scatter(x=t, y=data,mode='markers', marker=dict(size=2), name=name))
    fig.update_layout(title="BPMs estimators vs BPMs list", xaxis_title="Time", yaxis_title="BPM")
    fig.show(renderer=VisualizeParams.renderer)

def visualize_BPMs(BPMs_list):
    """
    This method create a plotly plot for visualizing a list of BPM signals. This method must be called
    inside a Jupyter notebook or a Colab notebook.

    Args:
        BPMs_list (list): The BPM signals is a 2D list structured as [[BPM_list, times, name_tag], ...]. The first element is a float32 ndarray that
            contains the BPM signal, the second element is a float32 1D ndarray that contains the time in seconds of each BPM, the third element is a string
            that is used in the plot's legend.    
    """
    fig = go.Figure()
    i = 1
    for e in BPMs_list:
        name = str(e[2])
        fig.add_trace(go.Scatter(x=e[1], y=e[0],
                                 mode='lines+markers', name=name))
        i += 1
    fig.update_layout(title="BPMs")
    fig.show(renderer=VisualizeParams.renderer)

def visualize_BVPs_PSD(BVPs, window, fps, minHz=0.65, maxHz=4.0):
    """
    This method create a plotly plot for visualizing the Power Spectral Density of a window of a windowed BVP signal. This method must be called
    inside a Jupyter notebook or a Colab notebook.

    Args:
        BVPs (list of float32 ndarray): windowed BPM signal as a list of length num_windows of float32 ndarray with shape [num_estimators, window_frames].
        window (int): the index of the window to plot.
        fps (float): frames per seconds.
        minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
    
    """
    data = BVPs[window]
    _, n = data.shape
    if data.shape[0] == 0:
        return np.float32(0.0)
    if n < 256:
        seglength = n
        overlap = int(0.8*n)  # fixed overlapping
    else:
        seglength = 256
        overlap = 200
    # -- periodogram by Welch
    F, P = welch(data, nperseg=seglength,
                 noverlap=overlap, fs=fps, nfft=2048)
    F = F.astype(np.float32)
    P = P.astype(np.float32)
    # -- freq subband (0.65 Hz - 4.0 Hz)
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
    Pfreqs = 60*F[band]
    Power = P[:, band]
    # -- BPM estimate by PSD
    Pmax = np.argmax(Power, axis=1)  # power max

    # plot
    fig = go.Figure()
    for idx in range(P.shape[0]):
        fig.add_trace(go.Scatter(
            x=F*60, y=P[idx], name="PSD_"+str(idx)+" no band"))
        fig.add_trace(go.Scatter(
            x=Pfreqs, y=Power[idx], name="PSD_"+str(idx)+" band"))
    fig.update_layout(title="PSD #" + str(window), xaxis_title='Beats per minute [BPM]')
    fig.show(renderer=VisualizeParams.renderer)

def visualize_landmarks_list(image_file_name=None, landmarks_list=None):
    """
    This method create a plotly plot for visualizing a list of facial landmarks on a given image. This is useful
    for studying and analyzing the available facial points of MediaPipe (https://google.github.io/mediapipe/solutions/face_mesh.html).
    This method must be called inside a Jupyter notebook or a Colab notebook.

    Args:
        image_file_name (str): image file name or path (preferred png or jpg).
        landmarks_list (list): list of positive integers between 0 and 467 that identify patches centers (landmarks).
    
    """
    PRESENCE_THRESHOLD = 0.5
    VISIBILITY_THRESHOLD = 0.5
    if image_file_name is None:
        image_file_name = pyVHR.__path__[0] + '/../img/face.png' 
    imag = cv2.imread(image_file_name, cv2.COLOR_RGB2BGR)
    imag = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:
        image = cv2.imread(image_file_name)
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return
        width = image.shape[1]
        height = image.shape[0]
        face_landmarks = results.multi_face_landmarks[0]
        ldmks = np.zeros((468, 3), dtype=np.float32)
        for idx, landmark in enumerate(face_landmarks.landmark):
            if ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD)
                    or (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                ldmks[idx, 0] = -1.0
                ldmks[idx, 1] = -1.0
                ldmks[idx, 2] = -1.0
            else:
                coords = mp_drawing._normalized_to_pixel_coordinates(
                    landmark.x, landmark.y, width, height)
                if coords:
                    ldmks[idx, 0] = coords[0]
                    ldmks[idx, 1] = coords[1]
                    ldmks[idx, 2] = idx
                else:
                    ldmks[idx, 0] = -1.0
                    ldmks[idx, 1] = -1.0
                    ldmks[idx, 2] = -1.0
    
    filtered_ldmks = []
    if landmarks_list is not None:
        for idx in landmarks_list:
            filtered_ldmks.append(ldmks[idx])
        filtered_ldmks = np.array(filtered_ldmks, dtype=np.float32)
    else:
        filtered_ldmks = ldmks

    fig = px.imshow(imag)
    for l in filtered_ldmks:
        name = 'ldmk_' + str(int(l[2]))
        fig.add_trace(go.Scatter(x=(l[0],), y=(l[1],), name=name, mode='markers', 
                                marker=dict(color='blue', size=3)))
    fig.update_xaxes(range=[0,imag.shape[1]])
    fig.update_yaxes(range=[imag.shape[0],0])
    fig.update_layout(paper_bgcolor='#eee') 
    fig.show(renderer=VisualizeParams.renderer)

from pyVHR.BPM.utils import Model, gaussian, Welch, Welch_cuda, pairwise_distances, circle_clustering, optimize_partition, gaussian_fit
def visualize_BVPs_PSD_clutering(GT_BPM, GT_times , BVPs, times, fps, minHz=0.65, maxHz=4.0, out_fact=1):
    """
    TODO: documentare
    This method create a plotly plot for visualizing the Power Spectral Density of a window of a windowed BVP signal. This method must be called
    inside a Jupyter notebook or a Colab notebook.

    Args:
        BVPs (list of float32 ndarray): windowed BPM signal as a list of length num_windows of float32 ndarray with shape [num_estimators, window_frames].
        window (int): the index of the window to plot.
        fps (float): frames per seconds.
        minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
    
    """  
    gmodel = Model(gaussian, independent_vars=['x', 'mu', 'a'])
    for i,X in enumerate(BVPs):
        if X.shape[0] == 0:
            continue
        F, PSD = Welch(X, fps, minHz, maxHz)
        W = pairwise_distances(PSD, PSD, metric='cosine')
        #W = np.corrcoef(PSD)-np.eye(W.shape[0])
        #W = dtw.distance_matrix_fast(PSD.astype(np.double))
        theta = circle_clustering(W, eps=0.01)

        # bi-partition, sum and normalization
        P, Q, Z, med_elem_P, med_elem_Q = optimize_partition(theta, out_fact=out_fact)

        P0 = np.sum(PSD[P,:], axis=0)
        max = np.max(P0, axis=0)
        max = np.expand_dims(max, axis=0)
        P0 = np.squeeze(np.divide(P0, max))

        P1 = np.sum(PSD[Q,:], axis=0)
        max = np.max(P1, axis=0)
        max = np.expand_dims(max, axis=0)
        P1 = np.squeeze(np.divide(P1, max))
        
        # peaks
        peak0_idx = np.argmax(P0) 
        P0_max = P0[peak0_idx]
        F0 = F[peak0_idx]

        peak1_idx = np.argmax(P1) 
        P1_max = P1[peak1_idx]
        F1 = F[peak1_idx]
        
        # Gaussian fitting
        result0, G1, sigma0 = gaussian_fit(gmodel, P0, F, F0, 1)  # Gaussian fit 
        result1, G2, sigma1 = gaussian_fit(gmodel, P1, F, F1, 1)  # Gaussian fit 
        chisqr0 = result0.chisqr
        chisqr1 = result1.chisqr

        print('** Processing window n. ', i)
        t = np.argmin(np.abs(times[i]-GT_times))
        GT = GT_BPM[t]
        print('GT = ', GT, '   freq0 max = ', F0, '   freq1 max = ', F1)
        # TODO: rifare con plotly
        plt.figure(figsize=(14, 4))
        plt.subplot(121)
        plt.plot(F, PSD[P,:].T, alpha=0.2)
        _, top = plt.ylim() 
        plt.plot(F, 0.5*top*P0, color='blue')
        
        plt.subplot(122)
        plt.ylim(0,1.1) 
        plt.plot(F, result0.best_fit, linestyle='-', color='magenta')
        plt.fill_between(F, result0.best_fit, alpha=0.1)
        plt.plot(F, P0, color='blue')
        plt.vlines(F0, ymin=0, ymax=1, linestyle='-.', color='blue')
        plt.vlines(GT,  ymin=0, ymax=1.1, linestyle='-.', color='black')
        plt.title('err = '+ str(np.round(np.abs(GT-F0),2)) +' -- sigma = '+ str(np.round(sigma0,2)) + ' -- chi sqr = ' + str(str(np.round(chisqr0,2))))
        plt.show()

        # second figure
        plt.figure(figsize=(14, 4))
        plt.subplot(121)
        plt.plot(F, PSD[Q,:].T, alpha=0.2)
        _, top = plt.ylim() 
        plt.plot(F, 0.5*top*P1, color='red')
        plt.subplot(122)
        plt.ylim(0,1.1) 
        plt.plot(F, result1.best_fit, linestyle='-', color='magenta')
        plt.fill_between(F, result1.best_fit, alpha=0.1)
        plt.plot(F, P1, color='red')
        plt.vlines(F1, ymin=0, ymax=1, linestyle='-.', color='red')
        plt.vlines(GT,  ymin=0, ymax=1.1, linestyle='-.', color='black')
        plt.title('err = '+ str(np.round(np.abs(GT-F1),2)) +' -- sigma = '+ str(np.round(sigma1,2)) + ' -- chi sqr = ' + str(str(np.round(chisqr1,2))))
        plt.show()

        # centers
        C1 = [np.cos(med_elem_P), np.sin(med_elem_P)]
        C2 = [np.cos(med_elem_Q), np.sin(med_elem_Q)]

        #hist, bins = histogram(theta, nbins=256)
        labels = np.zeros_like(theta)
        labels[Q] = 1
        labels[Z] = 2
        plot_circle(theta, l=labels, C1=C1, C2=C2)

def plot_circle(theta, l=None, C1=None, C2=None, radius=500):
    """
    TODO: documentare
    Produce a plot with the locations of all poles and zeros
    """

    x = np.cos(theta)
    y = np.sin(theta)

    fig = go.Figure()
    fig.add_shape(type="circle", xref="x", yref="y", x0=-1, y0=-1, x1=1, y1=1, line=dict(color="black", width=1))
    
    if l is None:
        fig.add_trace(go.Scatter(x=x, y=y,
            mode='markers',
            marker_symbol='circle',
            marker_size=10))
    else:
        ul = np.unique(l)
        cols = ['blue', 'red', 'gray']
        for c,u in zip(cols,ul):
            idx = np.where(u == l)
            fig.add_trace(go.Scatter(x=x[idx], y=y[idx],
                mode='markers',
                marker_symbol='circle',
                marker_color=c, 
                # marker_line_color=cols[c],
                marker_line_width=0, 
                marker_size=10))
        
    # separator plane
    fig.add_trace(go.Scatter(x=[0.0, C1[0]], y=[0.0, C1[1]],
        mode='lines+markers',
        marker_symbol='circle',
        marker_color='blue', 
        marker_line_color='blue',
        marker_line_width=2, 
        marker_size=2))
    fig.add_trace(go.Scatter(x=[0.0, C2[0]], y=[0.0, C2[1]],
        mode='lines+markers',
        marker_symbol='circle',
        marker_color='red', 
        marker_line_color='red',
        marker_line_width=2, 
        marker_size=2))
    
    M = 1.05
    fig.update_xaxes(title='', range=[-M, M])
    fig.update_yaxes(title='', range=[-M, M])
    fig.update_layout(title='clusters', width=radius, height=radius)
    fig.show(renderer=VisualizeParams.renderer)

#### CUSTOM VISUALIZATION FUNCTIONS ####
from scipy.ndimage import zoom
from scipy.fft import fft


def visualize_BVPs_PSD_with_IDs(BVPs,BVP_IDs, window, fps, minHz=0.65, maxHz=4.0):
    """
    This method create a plotly plot for visualizing the Power Spectral Density of a window of a windowed BVP signal. This method must be called
    inside a Jupyter notebook or a Colab notebook.

    Args:
        BVPs (list of float32 ndarray): windowed BPM signal as a list of length num_windows of float32 ndarray with shape [num_estimators, window_frames].
        window (int): the index of the window to plot.
        fps (float): frames per seconds.
        minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
    
    """
    data = BVPs[window]
    patches = BVP_IDs[window]
    _, n = data.shape
    if data.shape[0] == 0:
        return np.float32(0.0)
    if n < 256:
        seglength = n
        overlap = int(0.8*n)  # fixed overlapping
    else:
        seglength = 256
        overlap = 200
    # -- periodogram by Welch
    F, P = welch(data, nperseg=seglength,
                 noverlap=overlap, fs=fps, nfft=2048)
    F = F.astype(np.float32)
    P = P.astype(np.float32)
    # -- freq subband (0.65 Hz - 4.0 Hz)
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
    Pfreqs = 60*F[band]
    Power = P[:, band]
    # -- BPM estimate by PSD
    Pmax = np.argmax(Power, axis=1)  # power max

    # plot
    fig = go.Figure()
    for idx in range(P.shape[0]):
        fig.add_trace(go.Scatter(
            x=F*60, y=P[idx], name="PSD_"+str(patches[idx])+" no band"))
        fig.add_trace(go.Scatter(
            x=Pfreqs, y=Power[idx], name="PSD_"+str(patches[idx])+" band"))
    fig.update_layout(title="PSD #" + str(window), xaxis_title='Beats per minute [BPM]')
    fig.show(renderer=VisualizeParams.renderer)

def process_patch_BVPs(fixed_patches, patch_ids, patch_bvps, fps, minHz=0.65, maxHz=4.0):
    """
    The aim of this function is to perform Welch's algorithm just once, so we can plot the heatmap of the PSDs
    
    args:
    fixed_patches (list): list of the FixedPatch classes used for this video
    patch_ids (list): list of patch ids used in each window
    patch_bvps (list): list of patch bvps for each window
    fps (int): frames per second
    minHz (float): minimum frequency for PSD
    maxHz (float): maximum frequency for PSD

    returns:
    processed_patch_bvps ([ID,max_power,max_bpm,phase]) for each window
    """
    processed_patch_bvps = []

    for window in range(len(patch_bvps)):
        patch = 0
        windowed_patches = []
        for fixed_patch in fixed_patches:
            if fixed_patch.ID in patch_ids[window]:

                data = patch_bvps[window][patch]

                n = len(data)
                if n < 256:
                    seglength = n
                    overlap = int(0.8*n)  # fixed overlapping
                else:
                    seglength = 256
                    overlap = 200

                # Conduct Welch's method to estimate PSD
                F, P = welch(data, nperseg=seglength, noverlap=overlap, fs=fps, nfft=2048)
                # don't include the regions outside our bandpass
                band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
                Pfreqs = 60*F[band]
                Power = P[band]
                # Get max power and the corresponding frequency
                max_power = np.max(Power)
                max_frequency = Pfreqs[np.argmax(Power)]

                # Conduct FFT to get the complex Fourier coefficients
                yf = fft(data)
                xf = np.fft.fftfreq(len(yf), 1/fps)

                # Find the closest index to the max frequency in the FFT
                idx_max_freq = np.argmin(np.abs(xf - max_frequency))

                # Extract the phase at the specific frequency
                phase = np.angle(yf[idx_max_freq])

                # Append the processed data to the list
                windowed_patches.append([fixed_patch.ID, max_power, max_frequency, phase])
                
                patch += 1
            else:
                # Append a placeholder
                windowed_patches.append([fixed_patch.ID, 0, 0, 0])
                
        processed_patch_bvps.append(windowed_patches)

    return processed_patch_bvps

def interpolate_array(input_array, output_shape):
    """
    Interpolate the input 2D array to match the output shape.
    
    Args:
        input_array (ndarray): The input 2D array of shape (m, n).
        output_shape (tuple): The shape (o, p) to which the input array should be interpolated.
    
    Returns:
        ndarray: The interpolated 2D array of shape (o, p).
    """
    m, n = input_array.shape
    o, p = output_shape
    
    zoom_factor_o = o / m
    zoom_factor_p = p / n
    
    # Perform interpolation using scipy's zoom function
    interpolated_array = zoom(input_array, (zoom_factor_o, zoom_factor_p))
    
    return interpolated_array

def overlay_arrays(base_image, overlay_array, alpha=0.5):
    """
    Overlay the interpolated array on the base image.
    
    Args:
        base_image (ndarray): The base image array of shape (o, p, c), where c is the number of color channels.
        overlay_array (ndarray): The array to overlay, of shape (o, p).
        alpha (float): Weighting factor to control the transparency of the overlay.
        
    Returns:
        ndarray: The resulting image after overlaying.
    """
    # Ensure both arrays have the same shape
    if base_image.shape[:2] != overlay_array.shape:
        raise ValueError("Shape mismatch between base image and overlay array")
        
    # Normalize overlay_array to match base image intensity
    overlay_normalized = (overlay_array / np.max(overlay_array)) * 255
    
    # Convert to the same data type as base image
    overlay_normalized = overlay_normalized.astype(base_image.dtype)
    
    # Create a colored version of overlay array with 3 channels
    overlay_colored = cv2.applyColorMap(overlay_normalized, cv2.COLORMAP_JET)
    
    # Perform the overlay
    output_image = cv2.addWeighted(base_image, 1 - alpha, overlay_colored, alpha, 0)
    
    return output_image

def visualize_BVPs_heatmap(image, BVPs, BVP_IDs, window, patch_shape, overlap, fps, minHz=0.65, maxHz=4.0):
    """
    Visualize Blood Volume Pulses (BVPs) on a given image with heatmap.

    Args:
        image (np.ndarray): Input image.
        BVPs (list): List of BVPs for each patch.
        BVP_IDs (list): List of BVP IDs for each patch.
        window (int): Window index to visualize.
        patch_shape (tuple): Shape of the patch (height, width).
        overlap (int): Overlap between patches.
        fps (int): Frames per second.
        minHz (float, optional): Minimum frequency for PSD. Defaults to 0.65.
        maxHz (float, optional): Maximum frequency for PSD. Defaults to 4.0.

    Returns:
        list: landmarks with attributes (id, x_center, y_center, max_freq, max_power, 0)
        img_cv: OpenCV image with heatmap overlay.
    """

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Validate image dimensions
    image_shape = gray_image.shape
    if image_shape[0] < patch_shape[0] or image_shape[1] < patch_shape[1]:
        raise ValueError("Patch shape must be smaller than the image shape.")
    
    patch_height, patch_width = patch_shape
    
    # Select BVPs and IDs for the specific window
    BVPs_for_window = BVPs[window]
    BVP_IDs_for_window = BVP_IDs[window]
    
    # Initialize containers for landmarks and maximum powers
    ldmks = []
    max_powers = []
    
    # Calculate number of rows and columns based on the patch size and overlap
    num_rows = (image_shape[0] - patch_height) // (patch_height - overlap) + 1
    num_cols = (image_shape[1] - patch_width) // (patch_width - overlap) + 1
    
    # Initialize RGBA heatmap
    heatmap = np.zeros((*image_shape, 4), dtype=np.uint8)
    
    # Calculate the maximum power for normalization
    global_max_power = 0
    welch_results = []
    for BVP in BVPs_for_window:
        F, P = welch(BVP, nperseg=128, noverlap=64, fs=fps, nfft=2048)
        welch_results.append((F, P))
        Power = P[np.logical_and(F > minHz, F < maxHz)]
        max_power = np.max(Power)
        max_powers.append(max_power)
        global_max_power = max(global_max_power, max_power)

    # Normalize max powers for opacity scaling
    normalized_powers = np.array(max_powers) / global_max_power
    
    # Populate the heatmap and landmarks
    for idx, (BVP, id) in enumerate(zip(BVPs_for_window, BVP_IDs_for_window)):
        F, P = welch_results[idx]
        max_freq = F[np.argmax(P[np.logical_and(F > minHz, F < maxHz)])] * 60  # Convert to BPM
        
        row_idx = id // num_cols
        col_idx = id % num_cols
        
        x_start, y_start = col_idx * (patch_width - overlap), row_idx * (patch_height - overlap)
        x_center, y_center = x_start + patch_width // 2, y_start + patch_height // 2
        
        ldmks.append([id, x_center, y_center, max_freq, max_powers[idx], 0])
        
        color_value = np.interp(max_freq, [45, 200], [0, 1])
        color_map = plt.get_cmap('plasma')
        color = np.array(color_map(color_value)[:3]) * 255

        alpha = normalized_powers[idx] * 255  # Scale opacity
        
        heatmap[y_start:y_start + patch_height, x_start:x_start + patch_width, :3] = color
        heatmap[y_start:y_start + patch_height, x_start:x_start + patch_width, 3] = int(alpha)

    # Apply Gaussian blur to the heatmap
    heatmap_blurred = cv2.GaussianBlur(heatmap, (39, 39), 0)
    
    # Create figure for plotting
    fig, ax = plt.subplots()
    
    # Display the grayscale image
    ax.imshow(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB))
    
    # Display heatmap overlay, using 'plasma' colormap
    img = ax.imshow(heatmap_blurred, cmap='plasma', alpha=0.6, label='Heatmap Overlay', vmin=45, vmax=200)
    
    # Add colorbar with label, using the same 'plasma' colormap
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label('Frequency (BPM)')
    
    # Disable axis and add title
    plt.axis('off')
    plt.title('BVP Heatmap Overlay')
    
    # Convert matplotlib figure to OpenCV format
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img_arr = np.asarray(buf)
    
    # Convert from RGBA to BGR format
    img_cv = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGRA)
    
    # Close the figure to free memory
    plt.close(fig)

    return ldmks, img_cv

def visualize_BPM_Errors_heatmap(image, ldmks, timesES, wind, timesGT, bpmGT, patch_shape, overlap, vmin=-100, vmax=100):
    """
    Create a heatmap and overlay the errors over the grayscale image.
    
    Parameters:
    - image: The input image.
    - ldmks: Landmarks with shape (id, x_center, y_center, max_freq, max_power, 0).
    - timesES: Times for ES.
    - wind: Window for ES.
    - timesGT: Ground truth times.
    - bpmGT: Ground truth BPM.
    - patch_shape: Shape of the patch.
    - overlap: Overlap between patches.
    - vmin: Minimum value for colormap.
    - vmax: Maximum value for colormap.

    Returns:
    - fig_data: NumPy array containing the overlay image and colorbar.
    """

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_4channel = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGRA)
    image_shape = gray_image.shape
    patch_height, patch_width = patch_shape

    # Initialize 4-channel heatmap
    heatmap = np.zeros((*image_shape, 4), dtype=np.uint8)

    # Extract maximum power values to normalize opacity
    max_powers = np.array([max_power for _, _, _, _, max_power, _ in ldmks])
    global_min_power, global_max_power = np.min(max_powers), np.max(max_powers)

    # Get closest ground truth BPM for given time window
    time = timesES[wind]
    closest_idx = np.argmin(np.abs(np.array(timesGT) - time))
    closest_bpmGT = bpmGT[closest_idx]

    # Initialize colormap normalization
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Calculate rows and columns based on patch size and overlap
    num_rows = (image_shape[0] - patch_height) // (patch_height - overlap) + 1
    num_cols = (image_shape[1] - patch_width) // (patch_width - overlap) + 1

    # Loop through landmarks to populate heatmap
    for id, _, _, max_freq, max_power, _ in ldmks:
        row_idx = id // num_cols
        col_idx = id % num_cols
        x_start = col_idx * (patch_width - overlap)
        y_start = row_idx * (patch_height - overlap)

        # Compute BPM error
        error = max_freq - closest_bpmGT

        # Normalize maximum power for opacity
        if global_max_power == global_min_power:
            normalized_opacity = 255  # To avoid division by zero
        else:
            normalized_opacity = int((max_power - global_min_power) / (global_max_power - global_min_power) * 255)

        # Determine color based on error value
        color = (np.array(plt.cm.turbo(norm(error)))[:3] * 255).astype(np.uint8)
        color = np.append(color, normalized_opacity).astype(np.uint8)

        # Update heatmap
        heatmap[y_start:y_start + patch_height, x_start:x_start + patch_width] = color

    # Apply Gaussian blur to the heatmap
    heatmap_blurred = cv2.GaussianBlur(heatmap, (39, 39), 0)

    # Create the overlay image
    overlay_image = cv2.addWeighted(gray_4channel, 0.7, heatmap_blurred, 0.3, 0)

    # Create matplotlib figure and axes
    fig, ax = plt.subplots()

    # Display overlay image
    ax.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGRA2RGB))

    # Create colorbar
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)
    cbar = plt.colorbar(sm, ax=ax, boundaries=np.linspace(vmin, vmax, 21))
    cbar.set_label('Error Value')

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Convert figure to NumPy array
    fig.canvas.draw()
    fig_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    fig_data = fig_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)  # Close figure to free up resources

    return fig_data