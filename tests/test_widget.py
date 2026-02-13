from napari_etminflux_data_viewer._widget import LoaderWidget


def test_image_threshold_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    my_widget = LoaderWidget(viewer)

    assert my_widget.events == []
