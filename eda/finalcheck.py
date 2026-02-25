def prepare_dataset(self, split):

    X, y = self.load_split(split)  # or whatever your loading logic is

    print("--------------------------------------------------")
    print(f"{split.upper()} SPLIT DEBUG INFO")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Time dimension length:", X.shape[-1])
    print("Number of leads:", X.shape[1])
    print("--------------------------------------------------")

    return X, y