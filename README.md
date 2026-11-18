# SpinCam

SpinCam is a Windows desktop webcam preview with playful PowerPoint-style slide
transitions. It can send the processed frames to OBS Virtual Camera, which you
can select as your camera in Discord.

## Setup

1. Install Python 3.10 or later.
2. Install [OBS Studio](https://obsproject.com/) and start its **Virtual Camera**
   from OBS.
3. Install the app dependencies:

   ```powershell
   py -m pip install -r requirements.txt
   ```

4. Run the app:

   ```powershell
   py spin_cam.py
   ```

5. Select your webcam from the **Webcam** list and click **Connect webcam**.
   SpinCam automatically connects its output to OBS Virtual Camera. If the
   webcam is not listed, reconnect it and click **Refresh**.
6. In Discord, open **User Settings → Voice & Video → Camera** and select
   **OBS Virtual Camera**.

Choose a transition (3D Spin, Push, Wipe, Checkerboard, Pixel Dissolve, Iris,
Zoom, or Fade), then click **Play transition** or press the selected global
function key (F6–F12). The hotkey works while SpinCam is in the background.
3D Spin makes one full turn with a stronger tilt and a small outward zoom.
SpinCam opens at its minimum window size with **Clean (1280x720)** selected.
Lower options include Retro, Crunchy, Extra crunchy, Potato, and Cursed.
Filters include Noir, Sepia, Negative, Posterize, VHS, and Thermal.

## Notes

- If a webcam cannot connect, close other apps using it and click **Refresh**.
- OBS Virtual Camera must be running before you connect the webcam in SpinCam.
- The app does not need a Discord token or access your Discord account; Discord
  receives the video through its normal camera input.
