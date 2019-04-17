import h5py
import numpy as np

class Hdf5ResultsWriter():
    """
    This class is used to write the click probabilities for a specific set of search sessions to disk. This way they
    can used in an external source to calculate metrics and do visualization
    """
    def __init__(self, output_file=None):
        self.output_file = output_file

    def write(self, click_model, search_sessions):
        with h5py.File(self.output_file, "w") as f:
            datatype = np.dtype([("predictions", "(10,)f")])
            dset = f.create_dataset("data", (len(search_sessions),), maxshape=(None,), dtype=datatype, compression='gzip',
                                    compression_opts=9)

            for i, session in enumerate(search_sessions):
                click_probs = click_model.get_click_probs(session)
                dset[i] = (np.asarray(click_probs), )
