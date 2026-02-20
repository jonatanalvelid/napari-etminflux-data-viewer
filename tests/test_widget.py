from napari_etminflux_data_viewer._widget import EtMINFLUXDataViewerWidget


def test_image_threshold_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    my_widget = EtMINFLUXDataViewerWidget(viewer)

    assert my_widget.events == []
