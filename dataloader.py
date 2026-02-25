import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
import ast


class PTBXLDataset:
    def __init__(self, data_path, sampling_rate=100):
        self.path = Path(data_path)

        if (self.path / 'ptbxl_database_clean.csv').exists():
            metadata_file = 'ptbxl_database_clean.csv'
        elif (self.path / 'ptbxl_database.csv').exists():
            metadata_file = 'ptbxl_database.csv'
        else:
            raise FileNotFoundError("Could not find PTB-XL metadata CSV")

        self.sampling_rate = sampling_rate

        # Load metadata
        self.metadata = pd.read_csv(self.path / metadata_file, index_col='ecg_id')
        self.metadata.scp_codes = self.metadata.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load class mappings
        self.class_dict = pd.read_csv(self.path / 'scp_statements.csv', index_col=0)

        print(f"Loaded {len(self.metadata)} ECG records")


    def load_signal(self, ecg_id):
        filename = self.metadata.loc[ecg_id,
                                     'filename_hr' if self.sampling_rate == 500 else 'filename_lr']

        base_path = self.path / filename
        dat_path = base_path.with_suffix('.dat')
        hea_path = base_path.with_suffix('.hea')

        if not dat_path.exists() or not hea_path.exists():
            return None

        signal = wfdb.rdsamp(str(base_path))[0]
        return signal  # (1000, 12) at 100 Hz


    def get_labels(self, ecg_id):
        codes = self.metadata.loc[ecg_id, 'scp_codes']

        class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        label_vec = np.zeros(len(class_names))

        for code in codes.keys():
            if code in self.class_dict.index:
                superclass = self.class_dict.loc[code, 'diagnostic_class']
                if superclass in class_names:
                    label_vec[class_names.index(superclass)] = 1

        return label_vec


    def prepare_dataset(self, split='train'):
        """
        Record-level dataset (NO windowing)
        Returns:
            signals: (N, 12, 1000)
            labels:  (N, 5)
        """

        fold_mapping = {
            'train': [1,2,3,4,5,6,7,8],
            'val': [9],
            'test': [10]
        }

        subset = self.metadata[self.metadata.strat_fold.isin(fold_mapping[split])]

        all_signals = []
        all_labels = []

        for ecg_id in subset.index:
            signal = self.load_signal(ecg_id)
            if signal is None:
                continue

            label_vec = self.get_labels(ecg_id)

            if label_vec.sum() == 0:
                continue

            all_signals.append(signal.T)  # (12, 1000)
            all_labels.append(label_vec)

        signals = np.array(all_signals, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.float32)

        print(f"{split} set shape: {signals.shape}")

        return signals, labels