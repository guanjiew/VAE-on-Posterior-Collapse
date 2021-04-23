def get_empty_data_dict():
    return dict(iter=[],
                recon_loss=[],
                total_kld=[],
                dim_wise_kld=[],
                mean_kld=[],
                mu=[],
                var=[],
                images=[], beta=[])


class DataGather(object):
    def __init__(self):
        self.data = get_empty_data_dict()

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = get_empty_data_dict()
