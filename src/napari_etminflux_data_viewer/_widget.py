"""
Copyright: 2026 Jonatan Alvelid

napari widget for loading and visualizing data from event-triggered MINFLUX experiments.
"""

import os
from typing import TYPE_CHECKING

import numpy as np
import tifffile
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextBrowser,
    QWidget,
)

from . import obf_support

if TYPE_CHECKING:
    import napari


class LoaderWidget(QWidget):
    """A widget to load etMINFLUX event files into napari. Currently only simple etMINFLUX experiments with a single ROI per event are supported."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        self.folderField = QLineEdit()
        self.folderField.setReadOnly(True)

        self.loadFolderButton = QPushButton("Load folder")
        self.loadFolderButton.clicked.connect(self._list_events_from_folder)

        self.loadEventButton = QPushButton("Load event")
        self.loadEventButton.clicked.connect(self._load_event)

        self.localizationIterationLabel = QLabel(
            "Seq. iter. for final loc. (def.: 2D - 3; 3D - 4)"
        )
        self.localizationIterationField = QLineEdit("3")

        # add all events in a dropdown list
        self.events = []
        self.eventsPar = QComboBox()
        self.eventsParLabel = QLabel("Events in folder")

        # add recording modes in dropdown list
        self.recordingModes = [
            "Single",
            "MultiROI",
            "SingleROIFollow",
            "MultiROIFollow",
        ]
        self.recordingModesPar = QComboBox()
        self.recordingModesPar.addItems(self.recordingModes)
        self.recordingModesParLabel = QLabel("Recording mode")

        # add Imspector version in dropdown list
        self.imspectorVersions = ["m2205", "m2410"]
        self.imspectorVersionsPar = QComboBox()
        self.imspectorVersionsPar.addItems(self.imspectorVersions)
        self.imspectorVersionsParLabel = QLabel("Imspector version")

        # add data dimensions in dropdown list
        self.dataDimensions = ["2D", "3D"]
        self.dataDimensionsPar = QComboBox()
        self.dataDimensionsPar.addItems(self.dataDimensions)
        self.dataDimensionsParLabel = QLabel("Data dimensions")

        # create text browser for info from event log file
        self.logTextBrowser = QTextBrowser()

        self.eventsPath = None

        self.currentImage = None
        self.currentTracks = None

        # create grid and grid layout
        self.grid = QGridLayout()
        self.setLayout(self.grid)

        self.layout().addWidget(self.folderField, 0, 0, 1, 2)
        self.layout().addWidget(self.loadFolderButton, 1, 0, 1, 2)
        self.layout().addWidget(self.recordingModesParLabel, 2, 0)
        self.layout().addWidget(self.recordingModesPar, 2, 1)
        self.layout().addWidget(self.imspectorVersionsParLabel, 3, 0)
        self.layout().addWidget(self.imspectorVersionsPar, 3, 1)
        self.layout().addWidget(self.dataDimensionsParLabel, 4, 0)
        self.layout().addWidget(self.dataDimensionsPar, 4, 1)
        self.layout().addWidget(self.eventsParLabel, 5, 0)
        self.layout().addWidget(self.eventsPar, 5, 1)
        self.layout().addWidget(self.localizationIterationLabel, 6, 0)
        self.layout().addWidget(self.localizationIterationField, 6, 1)
        self.layout().addWidget(self.loadEventButton, 7, 0, 1, 2)
        self.layout().addWidget(self.logTextBrowser, 8, 0, 1, 2)

    def _list_events_from_folder(self):
        self.eventsPath = QFileDialog.getExistingDirectory(
            caption="Choose folder with events"
        )
        self.folderField.setText(self.eventsPath)
        self.events = []
        for msrfile in [
            file
            for file in os.listdir(self.eventsPath)
            if file.endswith(".msr")
        ]:
            if os.path.isfile(os.path.join(self.eventsPath, msrfile)):
                event_name = msrfile.split(".")[0].split("_")[0]
                self.events.append(event_name)
        self.eventsPar.clear()
        self.eventsPar.addItems(self.events)
        self.eventsPar.setCurrentIndex(0)
        print(f"Loading events from folder: {self.eventsPath}")

    def _set_logbrowser_text(self, filepath):
        with open(filepath) as f:
            text = f.read()
        self.logTextBrowser.setPlainText(text)

    def _load_event(self):
        # set default plotting params
        tail_width = 0.05
        tail_length = 30
        head_length = 3
        opacity = 0.8

        # check if an event is already loaded, and if so, remove it
        if self.currentImage is not None:
            self.viewer.layers.remove(self.currentImage)
            self.currentImage = None
        if self.currentTracks is not None:
            for trackslayer in self.currentTracks:
                self.viewer.layers.remove(trackslayer)
            self.currentTracks = None

        # list all available relevant files in the folder
        msrfiles = [
            file
            for file in os.listdir(self.eventsPath)
            if file.endswith(".msr")
        ]
        npyfiles = [
            file
            for file in os.listdir(self.eventsPath)
            if file.endswith(".npy")
        ]
        confrawfiles = [
            file
            for file in os.listdir(self.eventsPath)
            if file.endswith(".tif") and "raw" in file
        ]
        logfiles = [
            file
            for file in os.listdir(self.eventsPath)
            if file.endswith(".txt") and "log" in file
        ]

        # check which recording mode was used to record the data
        recording_mode = self.recordingModesPar.currentText()
        print(f"Recording mode: {recording_mode}")

        # check which imspector version was used to record the data (change of data structure in m2410)
        imspector_version = self.imspectorVersionsPar.currentText()
        print(f"Imspector version: {imspector_version}")

        # check how many data dimensions was recorded in the data
        if self.dataDimensionsPar.currentText() == "2D":
            dims = 2
            self.viewer.dims.ndisplay = 2
        elif self.dataDimensionsPar.currentText() == "3D":
            dims = 3
            self.viewer.dims.ndisplay = 3
        print(f"Data dimensions: {dims}")

        selected_event = self.eventsPar.currentText()
        selected_event_date = int(selected_event.split("-")[0])
        selected_event_time = int(selected_event.split("-")[1])
        selected_event_id = self.eventsPar.currentIndex()
        eventmsrfile = msrfiles[selected_event_id]
        eventlogfile = logfiles[selected_event_id]
        self._set_logbrowser_text(os.path.join(self.eventsPath, eventlogfile))

        if "Follow" in recording_mode:
            # get date and time of selected event, and find all corresponding ROI npy files and conf tiff files
            if self.eventsPar.currentIndex() > 0:
                conf_date_prev = int(
                    self.events[self.eventsPar.currentIndex() - 1].split("-")[
                        0
                    ]
                )
            else:
                conf_date_prev = 0
            if (
                self.eventsPar.currentIndex() > 0
                and selected_event_date == conf_date_prev
            ):
                conf_time_prev = int(
                    self.events[self.eventsPar.currentIndex() - 1].split("-")[
                        1
                    ]
                )
            else:
                conf_time_prev = 0
            eventnpyfiles = [
                file
                for file in npyfiles
                if int(file.split("-")[0]) == selected_event_date
                and int(file.split("-")[1].split("_")[0]) > conf_time_prev
                and int(file.split("-")[1].split("_")[0]) < selected_event_time
            ]
            eventconfrawfiles = [
                file
                for file in confrawfiles
                if int(file.split("-")[0]) == selected_event_date
                and int(file.split("-")[1].split("_")[0]) > conf_time_prev
                and int(file.split("-")[1].split("_")[0])
                <= selected_event_time
            ]
            print(eventconfrawfiles)
            print(eventnpyfiles)

            # get number of ROIs between each conf file, to know how much time to add (by copying frames as we cannot have unevenly temporally spaced confocal frames) for each confocal stack frames
            conf_time_prev = 0
            rois_between_confs = []
            for conf_file in eventconfrawfiles:
                conf_time_curr = int(conf_file.split("-")[1].split("_")[0])
                roi_files_between = [
                    file
                    for file in eventnpyfiles
                    if int(file.split("-")[0]) == selected_event_date
                    and int(file.split("-")[1].split("_")[0]) > conf_time_prev
                    and int(file.split("-")[1].split("_")[0]) < conf_time_curr
                ]
                rois_between_confs.append(len(roi_files_between))
                conf_time_prev = conf_time_curr
            print(rois_between_confs)

            # load triggering confocal raw image
            confidx = 0
            image_conf = np.swapaxes(
                tifffile.imread(
                    os.path.join(self.eventsPath, eventconfrawfiles[confidx])
                )[-1],
                1,
                0,
            )

            # load msr file and read confocal metadata
            msr_dataset = obf_support.File(
                os.path.join(self.eventsPath, eventmsrfile)
            )
            for conf_stack in msr_dataset.stacks:
                if (conf_stack.shape[0] == image_conf.shape[0]) and (
                    conf_stack.shape[1] == image_conf.shape[1]
                ):
                    # found confocal stack from msr file that matches shape of confocal image from tiff
                    break
            pxsize = conf_stack.pixel_sizes[0] * 1e6
            pxshift = pxsize / 2
            conf_size_px = (conf_stack.shape[0], conf_stack.shape[1])
            conf_size = (
                conf_stack.lengths[0] * 1e6,
                conf_stack.lengths[1] * 1e6,
            )
            conf_offset = (
                conf_stack.offsets[0] * 1e6,
                conf_stack.offsets[1] * 1e6,
            )
            print(f"MSR file loaded: {msr_dataset}")
            print(
                f"Confocal stack info - size: {conf_size_px[0]}x{conf_size_px[1]} pixels, physical size: {conf_size[0]}x{conf_size[1]} um, pixel size: {pxsize} um, offset: {conf_offset} um"
            )

            # load confocal images into stack
            confimagestack = []
            for idx, eventconfrawfile in enumerate(eventconfrawfiles):
                image_conf = tifffile.imread(
                    os.path.join(self.eventsPath, eventconfrawfile)
                )
                if idx == 0:
                    initframes = np.shape(image_conf)[0]
                if len(np.shape(image_conf)) > 2:
                    if idx == 0:
                        for image in image_conf:
                            confimagestack.append(np.swapaxes(image, 1, 0))
                        for _ in range(rois_between_confs[idx] - 1):
                            confimagestack.append(confimagestack[-1])
                    else:
                        for image in image_conf:
                            for _ in range(rois_between_confs[idx]):
                                confimagestack.append(np.swapaxes(image, 1, 0))
                else:
                    confimagestack.append(np.swapaxes(image_conf, 1, 0))
            confimagestack.append(
                confimagestack[-1]
            )  # add last confocal frame
            confimagestack = np.stack(confimagestack)
            confimagestack = confimagestack[:, np.newaxis, :, :]
            print(np.shape(confimagestack))
            print(initframes)

            # loop through event npy files and add tracks to separate layers
            self.currentTracks = []
            prevmaxtime = 0
            prevmaxtimes = []
            prevmaxtrackid = 0
            for eventnpyfile in eventnpyfiles:
                # load MINFLUX localization data
                tracksnew, track_ids, _ = self._load_MINFLUX_loc_data(
                    eventnpyfile,
                    imspector_version,
                    dims,
                    addtime=prevmaxtime,
                    addid=prevmaxtrackid,
                )
                if prevmaxtime == 0:
                    tracks = np.copy(tracksnew)
                else:
                    tracks = np.concatenate((tracks, tracksnew), axis=0)
                prevmaxtime = np.max(tracks[:, 1])
                prevmaxtimes.append(prevmaxtime)
                prevmaxtrackid = np.max(tracks[:, 0])
            print(prevmaxtimes)
            time_between_tracking_windows = np.mean(np.diff(prevmaxtimes))
            print(tracks[:, 1])
            tracks[:, 1] = tracks[:, 1] + time_between_tracking_windows * (
                initframes - 1
            )  # add initial time, where only confocal is shown
            print(tracks[:, 1])

            # add confocal image to viewer
            self.currentImage = self.viewer.add_image(
                confimagestack,
                name=f"confocal-event{selected_event_id}",
                colormap="gray",
                blending="additive",
                contrast_limits=[0, np.max(confimagestack[0])],
            )
            self.currentImage.scale = (
                time_between_tracking_windows,
                1,
                pxsize,
                pxsize,
            )
            self.currentImage.translate = (
                0,
                0,
                conf_offset[0] + pxshift,
                conf_offset[1] + pxshift,
            )
            print(
                f"Confocal image added to viewer: {image_conf.shape[0]}x{image_conf.shape[1]} pixels, pixel size: {pxsize} um, offset: {conf_offset} um"
            )

            # add all combined tracks to viewer
            self.currentTracks.append(
                self.viewer.add_tracks(
                    tracks,
                    name=f"minflux-tracks-event{selected_event_id}",
                    tail_width=tail_width,
                    tail_length=tail_length,
                    head_length=head_length,
                    opacity=opacity,
                    blending="translucent",
                    color_by="track_id",
                )
            )
            print(
                f"Tracks added to viewer: {tracks.shape[0]} vertices, {len(track_ids)} tracks"
            )
            self.viewer.dims.axis_labels = ("time", "z", "y", "x")

        else:
            eventconfrawfile = confrawfiles[selected_event_id]

            # load triggering confocal raw image
            image_conf = np.swapaxes(
                tifffile.imread(
                    os.path.join(self.eventsPath, eventconfrawfile)
                )[-1],
                1,
                0,
            )

            # load msr file and read confocal metadata
            msr_dataset = obf_support.File(
                os.path.join(self.eventsPath, eventmsrfile)
            )
            for conf_stack in msr_dataset.stacks:
                if (conf_stack.shape[0] == image_conf.shape[0]) and (
                    conf_stack.shape[1] == image_conf.shape[1]
                ):
                    # found confocal stack from msr file that matches shape of confocal image from tiff
                    break
            pxsize = conf_stack.pixel_sizes[0] * 1e6
            pxshift = pxsize / 2
            conf_size_px = (conf_stack.shape[0], conf_stack.shape[1])
            conf_size = (
                conf_stack.lengths[0] * 1e6,
                conf_stack.lengths[1] * 1e6,
            )
            conf_offset = (
                conf_stack.offsets[0] * 1e6,
                conf_stack.offsets[1] * 1e6,
            )
            print(f"MSR file loaded: {msr_dataset}")
            print(
                f"Confocal stack info - size: {conf_size_px[0]}x{conf_size_px[1]} pixels, physical size: {conf_size[0]}x{conf_size[1]} um, pixel size: {pxsize} um, offset: {conf_offset} um"
            )

            # add confocal image to viewer
            self.currentImage = self.viewer.add_image(
                image_conf,
                name=f"confocal-event{selected_event_id}",
                colormap="gray",
                blending="additive",
                contrast_limits=[0, np.max(image_conf)],
            )
            self.currentImage.scale = (pxsize, pxsize)
            self.currentImage.translate = (
                conf_offset[0] + pxshift,
                conf_offset[1] + pxshift,
            )
            print(
                f"Confocal image added to viewer: {image_conf.shape[0]}x{image_conf.shape[1]} pixels, pixel size: {pxsize} um, offset: {conf_offset} um"
            )

            if recording_mode == "Single":
                # load MINFLUX localization data of selected event
                eventnpyfile = npyfiles[selected_event_id]
                tracks, track_ids, loadsuccess = self._load_MINFLUX_loc_data(
                    eventnpyfile, imspector_version, dims
                )
                # add tracks to viewer
                self.currentTracks = []
                if loadsuccess:
                    self.currentTracks.append(
                        self.viewer.add_tracks(
                            tracks,
                            name=f"minflux-tracks-event{selected_event_id}",
                            tail_width=tail_width,
                            tail_length=tail_length,
                            head_length=head_length,
                            opacity=opacity,
                            blending="translucent",
                            color_by="track_id",
                        )
                    )
                    print(
                        f"Tracks added to viewer: {tracks.shape[0]} vertices, {len(track_ids)} tracks"
                    )
            elif recording_mode == "MultiROI":
                # get date and time of selected event, and find all corresponding ROI npy files
                if self.eventsPar.currentIndex() > 0:
                    conf_date_prev = int(
                        self.events[self.eventsPar.currentIndex() - 1].split(
                            "-"
                        )[0]
                    )
                else:
                    conf_date_prev = 0
                if (
                    self.eventsPar.currentIndex() > 0
                    and selected_event_date == conf_date_prev
                ):
                    conf_time_prev = int(
                        self.events[self.eventsPar.currentIndex() - 1].split(
                            "-"
                        )[1]
                    )
                else:
                    conf_time_prev = 0
                eventnpyfiles = [
                    file
                    for file in npyfiles
                    if int(file.split("-")[0]) == selected_event_date
                    and int(file.split("-")[1].split("_")[0]) > conf_time_prev
                    and int(file.split("-")[1].split("_")[0])
                    < selected_event_time
                ]

                # loop through event npy files and add tracks to separate layers
                self.currentTracks = []
                for eventnpyfile in eventnpyfiles:
                    roi_idx = int(eventnpyfile.split("ROI")[1].split("-")[0])
                    # load MINFLUX localization data
                    tracks, track_ids, loadsuccess = (
                        self._load_MINFLUX_loc_data(
                            eventnpyfile, imspector_version, dims
                        )
                    )
                    # add tracks to viewer
                    if loadsuccess:
                        self.currentTracks.append(
                            self.viewer.add_tracks(
                                tracks,
                                name=f"minflux-tracks-event{selected_event_id}-roi{roi_idx}",
                                tail_width=tail_width,
                                tail_length=tail_length,
                                head_length=head_length,
                                opacity=opacity,
                                blending="translucent",
                                color_by="track_id",
                            )
                        )
                        print(
                            f"Tracks from roi {roi_idx} added to viewer: {tracks.shape[0]} vertices, {len(track_ids)} tracks"
                        )
        self.viewer.reset_view()

    def _load_MINFLUX_loc_data(
        self,
        npyfile,
        imspector_version,
        dims=2,
        shuffle_ids=True,
        addtime=None,
        overheadtime=5,
        addid=None,
    ):
        # load MINFLUX localization data
        if imspector_version == "m2205":
            loc_it = int(self.localizationIterationField.text())
            mfx_dataset = np.load(os.path.join(self.eventsPath, npyfile))
            x = np.zeros((len(mfx_dataset), 1))
            y = np.zeros((len(mfx_dataset), 1))
            if dims == 3:
                z = np.zeros((len(mfx_dataset), 1))
            tid = np.zeros((len(mfx_dataset), 1))
            tim = np.zeros((len(mfx_dataset), 1))
            for i in range(len(mfx_dataset)):
                x[i] = mfx_dataset[i][0][loc_it][2][0]
                y[i] = mfx_dataset[i][0][loc_it][2][1]
                if dims == 3:
                    z[i] = mfx_dataset[i][0][loc_it][2][2]
                tid[i] = mfx_dataset[i][4]
                tim[i] = mfx_dataset[i][3]
            x_raw = x * 1e6
            y_raw = y * 1e6
            if dims == 3:
                z_raw = z * 1e6
            tid = tid.flatten()
            if addid is not None:
                tid = tid + addid
            tim = tim.flatten()
            if addtime is not None:
                tim = tim + addtime + overheadtime
            track_ids = list(map(int, set(tid)))
            track_ids.sort()
        elif imspector_version == "m2410":
            dataset = np.load(os.path.join(self.eventsPath, npyfile))
            x = dataset["loc"][:, 0]
            y = dataset["loc"][:, 1]
            if dims == 3:
                z = dataset["loc"][:, 2]
            x_raw = x * 1e6
            y_raw = y * 1e6
            if dims == 3:
                z_raw = z * 1e6
                z_raw = z_raw * 0.7  # z scaling for immersion mismatch
            tid = dataset["tid"].flatten()
            tim = dataset["tim"].flatten()
            fnl = dataset["fnl"].flatten()
            tid = tid.flatten()
            tim = tim.flatten()
            fnl = fnl.flatten()
            # sort out only final iteration localizations (not all iterations)
            x_raw = x_raw[fnl]
            y_raw = y_raw[fnl]
            if dims == 3:
                z_raw = z_raw[fnl]
            tid = tid[fnl]
            tim = tim[fnl]
            if addid is not None:
                tid = tid + addid
            if addtime is not None:
                tim = tim + addtime
            track_ids = list(map(int, set(tid)))
            track_ids.sort()
        else:
            print(
                f"Imspector version {imspector_version} not supported by widget."
            )
            return None, None, False
        print(
            f"Localization data loaded: {len(x_raw)} localizations, {len(track_ids)} tracks"
        )

        # randomize track IDs, so that plotting colors for each track will not be contiuous along time
        if shuffle_ids:
            unique_ids = np.unique(tid)
            shuffled_ids = np.random.permutation(unique_ids)
            mapping = dict(zip(unique_ids, shuffled_ids, strict=True))
            tid_randomized = np.array([mapping[tid] for tid in tid])
        else:
            tid_randomized = tid

        if dims == 3:
            tracks = np.zeros((len(x_raw), 5))
        else:
            tracks = np.zeros((len(x_raw), 4))
        tracks[:, 0] = tid_randomized
        tracks[:, 1] = tim
        if dims == 3:
            tracks[:, 2] = z_raw.flatten()
            tracks[:, 3] = x_raw.flatten()
            tracks[:, 4] = y_raw.flatten()
        else:
            tracks[:, 2] = x_raw.flatten()
            tracks[:, 3] = y_raw.flatten()

        return tracks, track_ids, True
