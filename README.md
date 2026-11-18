# SpinCam

SpinCam is a Windows desktop webcam preview with  PowerPoint-style slide
transitions.

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

## Notes

- If a webcam cannot connect, close other apps using it and click **Refresh**.
- OBS Virtual Camera must be running before you connect the webcam in SpinCam.
- The app does not need a Discord token or access your Discord account; Discord
  receives the video through its normal camera input.
