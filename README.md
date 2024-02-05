# totk_ocr
OCR for TotK coordinates

- ocr_rt.py - Captures, OCR, and plotting in a single script (assumes a mac)
  - The process is slow due to OCR, it may a few frames / second
- ocr_file.py - Dumps xyz images for use in OCR from a video file
  - Usually run these through Tesseract afterwards using gnu parallel, then post process

### Notes
The pre-processing of the images assumes the minimap is focused on the Sky for a consistent background.
Be aware any movement into the menu will likely change the minimap to the current level.

Using a different level minimap (Depths or Surface) would likely require different `cv.inRange()` 
parameters or other pre-processing.

Doing this is realtime `ocr_rt.py` is slow. I have found that circumn-aviagating the rooms by climbing
on the walls does a reasonable job of capturing the outline with the least amount of data.  Running around
the room might be ok but you would likely want to use the `ocr_file.py` for a higher details.

### Requirements 
- OpenCV for image capture and processing
- Tesseract for OCR
- python3
- PIL in ocr_file.py for image conversion
- pytesseract in ocr_rt.py for OCR
- matplotlib in ocr_rt.py for plotting
- numpy

### License
BSD 2-Clause 
